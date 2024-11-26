use polars::prelude::PolarsResult;
use polars_arrow::legacy::kernels::rolling::no_nulls::QuantileInterpolOptions;
use polars_core::prelude::{NamedFrom, Series};
use skiplist::OrderedSkipList;

use crate::rolling::skew::get_window_bounds;

pub fn rolling_quantile(
    input: &Series,
    quantile: f64,
    window_size: usize,
    min_periods: usize,
    center: bool,
    interpolation: QuantileInterpolOptions,
) -> PolarsResult<Series> {
    let x = input.len() as i64;

    if x == 0 {
        return Ok(Series::new("name_of_series".into(), Vec::<f64>::new()));
    }

    let (start, end) = get_window_bounds(
        window_size as i64, // window_size
        x,                  // num_values
        Some(center),       // center
        Some("right"),      // closed
        Some(1),            // step
    );

    let values: &[f64] = input.f64()?.cont_slice()?;
    let result = roll_quantile(
        values,
        &start,
        &end,
        min_periods as i64,
        quantile,
        interpolation,
    );

    let series = Series::new("name_of_series".into(), result);
    Ok(series.into())
}

#[inline]
fn is_monotonic_increasing_start_end_bounds(start: &[i64], end: &[i64]) -> bool {
    if start.len() <= 1 {
        return true;
    }

    for i in 1..start.len() {
        if start[i] < start[i - 1] || end[i] < end[i - 1] {
            return false;
        }
    }
    true
}

pub fn roll_quantile(
    values: &[f64],
    start: &[i64],
    end: &[i64],
    min_periods: i64,
    quantile: f64,
    interpolation: QuantileInterpolOptions,
) -> Vec<f64> {
    if !(0.0..=1.0).contains(&quantile) {
        panic!("quantile value {} not in [0, 1]", quantile);
    }

    let n = start.len();
    let mut output = vec![f64::NAN; n];

    // Check if bounds are monotonically increasing
    let is_monotonic = is_monotonic_increasing_start_end_bounds(start, end);
    let mut skiplist = OrderedSkipList::new();
    let mut nobs: i64 = 0;

    for i in 0..n {
        let s = start[i] as usize;
        let e = end[i] as usize;

        if i == 0 || !is_monotonic || start[i] >= end[i - 1] {
            if i != 0 {
                skiplist.clear();
                nobs = 0;
            }

            // Initialize window
            for j in s..e {
                if !values[j].is_nan() {
                    nobs += 1;
                    skiplist.insert(values[j]);
                }
            }
        } else {
            // Add new values
            for j in end[i - 1] as usize..e {
                if !values[j].is_nan() {
                    nobs += 1;
                    skiplist.insert(values[j]);
                }
            }

            // Remove old values
            for j in start[i - 1] as usize..s {
                if !values[j].is_nan() {
                    skiplist.remove(&values[j]);
                    nobs -= 1;
                }
            }
        }

        if nobs >= min_periods {
            let idx_with_fraction = quantile * ((nobs - 1) as f64);
            let idx = idx_with_fraction.floor() as usize;

            // Helper function to get value at index
            let get_value = |idx: usize| -> Option<f64> { skiplist.iter().nth(idx).copied() };

            if (idx_with_fraction - idx as f64).abs() < f64::EPSILON {
                // No interpolation needed
                output[i] = get_value(idx).unwrap_or(f64::NAN);
                continue;
            }

            match interpolation {
                QuantileInterpolOptions::Linear => {
                    let v_low = get_value(idx).unwrap_or(f64::NAN);
                    let v_high = get_value(idx + 1).unwrap_or(f64::NAN);
                    let fraction = idx_with_fraction - idx as f64;
                    output[i] = v_low + (v_high - v_low) * fraction;
                },
                QuantileInterpolOptions::Lower => {
                    output[i] = get_value(idx).unwrap_or(f64::NAN);
                },
                QuantileInterpolOptions::Higher => {
                    output[i] = get_value(idx + 1).unwrap_or(f64::NAN);
                },
                QuantileInterpolOptions::Nearest => {
                    let fraction = idx_with_fraction - idx as f64;
                    if fraction == 0.5 {
                        output[i] = if idx % 2 == 0 {
                            get_value(idx).unwrap_or(f64::NAN)
                        } else {
                            get_value(idx + 1).unwrap_or(f64::NAN)
                        };
                    } else if fraction < 0.5 {
                        output[i] = get_value(idx).unwrap_or(f64::NAN);
                    } else {
                        output[i] = get_value(idx + 1).unwrap_or(f64::NAN);
                    }
                },
                QuantileInterpolOptions::Midpoint => {
                    let v_low = get_value(idx).unwrap_or(f64::NAN);
                    let v_high = get_value(idx + 1).unwrap_or(f64::NAN);
                    output[i] = (v_low + v_high) / 2.0;
                },
            }
        }
    }

    output
}

// Example test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roll_quantile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let start = vec![0, 1, 2];
        let end = vec![3, 4, 5];
        let min_periods = 2;
        let quantile = 0.5;

        let result = roll_quantile(
            &values,
            &start,
            &end,
            min_periods,
            quantile,
            QuantileInterpolOptions::Linear,
        );

        // Add assertions based on expected results
    }
}
