use std::cmp::max;

use polars::prelude::*;
use polars_core::utils::Container;

pub fn rolling_skew(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
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
    let result = roll_skew(values, &start, &end, min_periods as i64);

    let series = Series::new("name_of_series".into(), result);
    Ok(series.into())
}

pub fn get_window_bounds(
    window_size: i64,
    mut num_values: i64,
    center: Option<bool>,
    closed: Option<&str>,
    step: Option<i64>,
) -> (Vec<i64>, Vec<i64>) {
    // Calculate offset based on center parameter and window size
    let offset = if center.unwrap_or(false) || window_size == 0 {
        (window_size - 1) / 2
    } else {
        0
    };

    // Use step size of 1 if not specified
    let step = step.unwrap_or(1);

    // Generate end points
    let mut end: Vec<i64> = (1 + offset..=num_values + offset)
        .step_by(step as usize)
        .collect();

    // Generate start points
    let mut start: Vec<i64> = end.iter().map(|x| x - window_size).collect();

    // Adjust for closed parameter
    match closed.unwrap_or("right") {
        "left" | "both" => {
            start.iter_mut().for_each(|x| *x -= 1);
        },
        "left" | "neither" => {
            end.iter_mut().for_each(|x| *x -= 1);
        },
        _ => {},
    }

    // Clip values to valid range
    start
        .iter_mut()
        .for_each(|x| *x = *x.clamp(&mut 0i64, &mut num_values));
    end.iter_mut()
        .for_each(|x| *x = *x.clamp(&mut 0i64, &mut num_values));

    (start, end)
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

#[inline]
fn calc_skew(
    minp: i64,
    nobs: i64,
    x: f64,
    xx: f64,
    xxx: f64,
    num_consecutive_same_value: i64,
) -> f64 {
    if nobs >= minp {
        let dnobs = nobs as f64;
        let a = x / dnobs;
        let b = xx / dnobs - a * a;
        let c = xxx / dnobs - a * a * a - 3.0 * a * b;

        if nobs < 3 {
            f64::NAN
        }
        // GH 42064 46431
        // uniform case, force result to be 0
        else if num_consecutive_same_value >= nobs {
            0.0
        } else if b <= 1e-14 {
            f64::NAN
        } else {
            let r = b.sqrt();
            ((dnobs * (dnobs - 1.0)).sqrt() * c) / ((dnobs - 2.0) * r * r * r)
        }
    } else {
        f64::NAN
    }
}

pub fn roll_skew(values: &[f64], start: &[i64], end: &[i64], min_periods: i64) -> Vec<f64> {
    let min_periods = max(min_periods, 3);
    let n = start.len();

    // Check if bounds are monotonically increasing
    let is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(start, end);

    let mut output = vec![0f64; n];

    // Find mean for centering
    let mut sum_val = 0f64;
    let mut nobs_mean = 0i64;
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    // Create centered values
    let mut values_copy = values.to_vec();
    for &val in values.iter() {
        if !val.is_nan() {
            nobs_mean += 1;
            sum_val += val;
        }
    }
    let mean_val = sum_val / nobs_mean as f64;

    // Center values if needed
    if min_val - mean_val > -1e5 {
        let mean_val = mean_val.round();
        for val in values_copy.iter_mut() {
            *val -= mean_val;
        }
    }

    let mut nobs = 0i64;
    let mut x = 0f64;
    let mut xx = 0f64;
    let mut xxx = 0f64;

    // Compensation variables for Kahan summation
    let mut compensation_xxx_add = 0f64;
    let mut compensation_xxx_remove = 0f64;
    let mut compensation_xx_add = 0f64;
    let mut compensation_xx_remove = 0f64;
    let mut compensation_x_add = 0f64;
    let mut compensation_x_remove = 0f64;

    let mut prev_value = values[0];
    let mut num_consecutive_same_value = 0;

    for i in 0..n {
        let s = start[i] as usize;
        let e = end[i] as usize;

        if i == 0 || !is_monotonic_increasing_bounds || s >= end[i - 1] as usize {
            prev_value = values[s]; // Just assign, don't redeclare
            num_consecutive_same_value = 0;

            // Reset accumulators
            nobs = 0;
            x = 0.0;
            xx = 0.0;
            xxx = 0.0;
            compensation_xxx_add = 0.0;
            compensation_xx_add = 0.0;
            compensation_x_add = 0.0;

            // Add values
            for j in s..e {
                let val = values_copy[j];
                add_skew(
                    val,
                    &mut nobs,
                    &mut x,
                    &mut xx,
                    &mut xxx,
                    &mut compensation_x_add,
                    &mut compensation_xx_add,
                    &mut compensation_xxx_add,
                    &mut num_consecutive_same_value,
                    &mut prev_value,
                );
            }
        } else {
            // Window is moving forward
            // Remove old values
            for j in start[i - 1] as usize..s {
                let val = values_copy[j];
                remove_skew(
                    val,
                    &mut nobs,
                    &mut x,
                    &mut xx,
                    &mut xxx,
                    &mut compensation_x_remove,
                    &mut compensation_xx_remove,
                    &mut compensation_xxx_remove,
                );
            }

            // Add new values
            for j in end[i - 1] as usize..e {
                let val = values_copy[j];
                add_skew(
                    val,
                    &mut nobs,
                    &mut x,
                    &mut xx,
                    &mut xxx,
                    &mut compensation_x_add,
                    &mut compensation_xx_add,
                    &mut compensation_xxx_add,
                    &mut num_consecutive_same_value,
                    &mut prev_value,
                );
            }
        }

        output[i] = calc_skew(min_periods, nobs, x, xx, xxx, num_consecutive_same_value);

        if !is_monotonic_increasing_bounds {
            nobs = 0;
            x = 0.0;
            xx = 0.0;
            xxx = 0.0;
        }
    }

    output
}

#[inline]
fn remove_skew(
    val: f64,
    nobs: &mut i64,
    x: &mut f64,
    xx: &mut f64,
    xxx: &mut f64,
    compensation_x: &mut f64,
    compensation_xx: &mut f64,
    compensation_xxx: &mut f64,
) {
    // Check if val is not NaN (in Rust we use is_nan() instead of val == val)
    if !val.is_nan() {
        // Decrement observation count
        *nobs -= 1;

        // Remove val from x sum with Kahan compensation
        let y = -val - *compensation_x;
        let t = *x + y;
        *compensation_x = t - *x - y;
        *x = t;

        // Remove val² from xx sum with Kahan compensation
        let y = -(val * val) - *compensation_xx;
        let t = *xx + y;
        *compensation_xx = t - *xx - y;
        *xx = t;

        // Remove val³ from xxx sum with Kahan compensation
        let y = -(val * val * val) - *compensation_xxx;
        let t = *xxx + y;
        *compensation_xxx = t - *xxx - y;
        *xxx = t;
    }
}

#[inline]
fn add_skew(
    val: f64,
    nobs: &mut i64,
    x: &mut f64,
    xx: &mut f64,
    xxx: &mut f64,
    compensation_x: &mut f64,
    compensation_xx: &mut f64,
    compensation_xxx: &mut f64,
    num_consecutive_same_value: &mut i64,
    prev_value: &mut f64,
) {
    // Check if val is not NaN
    if !val.is_nan() {
        // Increment observation count
        *nobs += 1;

        // Add val to x sum with Kahan compensation
        let y = val - *compensation_x;
        let t = *x + y;
        *compensation_x = t - *x - y;
        *x = t;

        // Add val² to xx sum with Kahan compensation
        let y = val * val - *compensation_xx;
        let t = *xx + y;
        *compensation_xx = t - *xx - y;
        *xx = t;

        // Add val³ to xxx sum with Kahan compensation
        let y = val * val * val - *compensation_xxx;
        let t = *xxx + y;
        *compensation_xxx = t - *xxx - y;
        *xxx = t;

        // Handle consecutive same values (GH#42064)
        if val == *prev_value {
            *num_consecutive_same_value += 1;
        } else {
            // Reset to 1 (include current value itself)
            *num_consecutive_same_value = 1;
        }
        *prev_value = val;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*; // For floating point comparisons

    #[test]
    fn test_simple_skew() {
        // Simple case with known skewness
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let start = vec![0, 1, 2];
        let end = vec![3, 4, 5];
        let min_periods = 3;

        let result = roll_skew(&values, &start, &end, min_periods);

        // Expected values can be calculated using pandas for verification
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10); // skew of [1,2,3]
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10); // skew of [2,3,4]
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10); // skew of [3,4,5]
    }

    #[test]
    fn test_nan_handling() {
        // Test with NaN values
        let values = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let start = vec![0, 1, 2];
        let end = vec![3, 4, 5];
        let min_periods = 3;

        let result = roll_skew(&values, &start, &end, min_periods);

        assert!(result[0].is_nan()); // Contains NaN
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10); // [3,4,5] is valid
    }

    #[test]
    fn test_min_periods() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let start = vec![0, 1];
        let end = vec![2, 3];
        let min_periods = 3; // Requiring 3 values but windows only have 2

        let result = roll_skew(&values, &start, &end, min_periods);

        // All results should be NaN as windows are too small
        assert!(result.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_uniform_values() {
        // Test with all same values - should return 0
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let start = vec![0, 1, 2];
        let end = vec![3, 4, 5];
        let min_periods = 3;

        let result = roll_skew(&values, &start, &end, min_periods);

        // Uniform distribution should have 0 skewness
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_non_monotonic_bounds() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let start = vec![2, 1, 0]; // Non-monotonic
        let end = vec![5, 4, 3]; // Non-monotonic
        let min_periods = 3;

        let result = roll_skew(&values, &start, &end, min_periods);

        // Test that it handles non-monotonic windows correctly
        // You might need to adjust these expected values based on your implementation
        assert!(!result.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_skewed_distribution() {
        // Test with known skewed distribution
        let values = vec![1.0, 1.0, 1.0, 10.0]; // Highly skewed right
        let start = vec![0];
        let end = vec![4];
        let min_periods = 3;

        let result = roll_skew(&values, &start, &end, min_periods);

        // Should be positive skew
        assert!(result[0] > 0.0);
    }

    #[test]
    fn test_rolling_skew_edge_cases() {
        // Original values from pandas test
        let values = vec![-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401];

        // Window size of 4
        let start = vec![0, 0, 0, 0, 1];
        let end = vec![1, 2, 3, 4, 5];

        let result = roll_skew(&values, &start, &end, 4);

        // Expected values from pandas:
        // [NaN, NaN, NaN, 0.177994, 1.548824]
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert_abs_diff_eq!(result[3], 0.177994, epsilon = 1e-6);
        assert_abs_diff_eq!(result[4], 1.548824, epsilon = 1e-6);
    }

    #[test]
    fn test_rolling_skew_equal_values() {
        // Create array of identical values
        let values = vec![1.1; 15];

        // Create windows
        let mut start = Vec::new();
        let mut end = Vec::new();
        for i in 0..15 {
            start.push(i64::max(0, i - 9)); // window size 10
            end.push(i + 1);
        }

        let result = roll_skew(&values, &start, &end, 10); // min_periods = 10

        // Test full windows (index >= 9)
        for i in 9..15 {
            assert_abs_diff_eq!(result[i], 0.0, epsilon = 1e-10);
        }

        // Test partial windows (index < 9)
        for i in 0..9 {
            assert!(result[i].is_nan());
        }
    }

    // Helper function to test internal calculations
    #[test]
    fn test_add_remove_skew() {
        let mut nobs = 0i64;
        let mut x = 0f64;
        let mut xx = 0f64;
        let mut xxx = 0f64;
        let mut compensation_x = 0f64;
        let mut compensation_xx = 0f64;
        let mut compensation_xxx = 0f64;
        let mut num_consecutive = 0i64;
        let mut prev_value = 0f64;

        // Add some values
        add_skew(
            1.0,
            &mut nobs,
            &mut x,
            &mut xx,
            &mut xxx,
            &mut compensation_x,
            &mut compensation_xx,
            &mut compensation_xxx,
            &mut num_consecutive,
            &mut prev_value,
        );

        assert_eq!(nobs, 1);
        assert_abs_diff_eq!(x, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(xx, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(xxx, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rolling_skew_known_values() -> PolarsResult<()> {
        // Values known to produce specific skewness
        let values = vec![-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401];
        let series = Series::new("input".into(), values);

        let result = rolling_skew(&series, 4, 4, false)?;
        let result_slice = result.f64()?.cont_slice()?;

        // First three should be NaN
        assert!(result_slice[0].is_nan());
        assert!(result_slice[1].is_nan());
        assert!(result_slice[2].is_nan());

        // Compare with known values (from pandas)
        assert_abs_diff_eq!(result_slice[3], 0.177994, epsilon = 1e-6);
        assert_abs_diff_eq!(result_slice[4], 1.548824, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_window_bounds() {
        let (start, end) = get_window_bounds(
            3,             // window_size
            5,             // num_values
            Some(false),   // center
            Some("right"), // closed
            Some(1),       // step
        );

        // For a window size of 3, not centered:
        // start should be [0, 0, 0, 1, 2]
        // end should be   [1, 2, 3, 4, 5]
        assert_eq!(start, vec![0, 0, 0, 1, 2]);
        assert_eq!(end, vec![1, 2, 3, 4, 5]);

        // Test centered window
        let (start, end) = get_window_bounds(
            3,             // window_size
            5,             // num_values
            Some(true),    // center
            Some("right"), // closed
            Some(1),       // step
        );

        // For a window size of 3, centered:
        // The offset will be 1 ((3-1)/2)
        println!("start: {:?}", start);
        println!("end: {:?}", end);
    }
}
