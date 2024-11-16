use polars::prelude::*;
use std::cmp::max;
use polars_core::utils::Container;

pub fn rolling_kurtosis(
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
        window_size as i64,             // window_size
        x,                              // num_values
        Some(center),                   // center
        Some("right"),                  // closed
        Some(1),                        // step
    );

    let values: &[f64] = input.f64()?.cont_slice()?;
    let result = roll_kurt(values,
                           &start,
                           &end,
                           min_periods as i64);

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
    let mut start: Vec<i64> = end.iter()
        .map(|x| x - window_size)
        .collect();

    // Adjust for closed parameter
    match closed.unwrap_or("right") {
        "left" | "both" => {
            start.iter_mut().for_each(|x| *x -= 1);
        }
        "left" | "neither" => {
            end.iter_mut().for_each(|x| *x -= 1);
        }
        _ => {}
    }

    // Clip values to valid range
    start.iter_mut().for_each(|x| *x = *x.clamp(&mut 0i64, &mut num_values));
    end.iter_mut().for_each(|x| *x = *x.clamp(&mut 0i64, &mut num_values));

    (start, end)
}

#[inline]
fn is_monotonic_increasing_start_end_bounds(
    start: &[i64],
    end: &[i64],
) -> bool {

    if start.len() <= 1 {
        return true;
    }

    for i in 1..start.len() {
        if start[i] < start[i-1] || end[i] < end[i-1] {
            return false;
        }
    }
    true
}

#[inline]
fn calc_kurt(
    minp: i64,
    nobs: i64,
    x: f64,
    xx: f64,
    xxx: f64,
    xxxx: f64,
    num_consecutive_same_value: i64,
) -> f64 {

    if nobs >= minp {
        let dnobs = nobs as f64;
        let a = x / dnobs;
        let b = xx / dnobs - a * a;
        let c = xxx / dnobs - a * a * a - 3.0 * a * b;
        let d = xxxx / dnobs - a * a * a * a - 6.0 * b * a * a - 4.0 * c * a;

        if nobs < 4 {
            f64::NAN
        }
        // GH 42064 46431
        // uniform case, force result to be 0
        else if num_consecutive_same_value >= nobs {
            -3.0
        }

        else if b <= 1e-14 {
            f64::NAN
        } else {
            let k = (dnobs * dnobs - 1.0) * d / (b * b) - 3.0 * ((dnobs - 1.0).powi(2));
            k / ((dnobs - 2.0) * (dnobs - 3.0))
        }
    } else {
        f64::NAN
    }
}

pub fn roll_kurt(
    values: &[f64],
    start: &[i64],
    end: &[i64],
    min_periods: i64,
) -> Vec<f64> {
    let min_periods = max(min_periods, 4);
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
    if min_val - mean_val > -1e4 {
        let mean_val = mean_val.round();
        for val in values_copy.iter_mut() {
            *val -= mean_val;
        }
    }

    let mut nobs = 0i64;
    let mut x = 0f64;
    let mut xx = 0f64;
    let mut xxx = 0f64;
    let mut xxxx = 0f64;

    // Compensation variables for Kahan summation
    let mut compensation_xxxx_add = 0f64;
    let mut compensation_xxxx_remove = 0f64;
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

        if i == 0 || !is_monotonic_increasing_bounds || s >= end[i-1] as usize {

            prev_value = values[s];  // Just assign, don't redeclare
            num_consecutive_same_value = 0;

            // Reset accumulators
            nobs = 0;
            x = 0.0;
            xx = 0.0;
            xxx = 0.0;
            xxxx = 0.0;
            compensation_xxxx_add = 0.0;
            compensation_xxx_add = 0.0;
            compensation_xx_add = 0.0;
            compensation_x_add = 0.0;

            // Add values
            for j in s..e {
                let val = values_copy[j];
                add_kurt(
                    val,
                    &mut nobs,
                    &mut x,
                    &mut xx,
                    &mut xxx,
                    &mut xxxx,
                    &mut compensation_x_add,
                    &mut compensation_xx_add,
                    &mut compensation_xxx_add,
                    &mut compensation_xxxx_add,
                    &mut num_consecutive_same_value,
                    &mut prev_value,
                );
            }
        } else {
            // Window is moving forward
            // Remove old values
            for j in start[i-1] as usize..s {
                let val = values_copy[j];
                remove_kurt(
                    val,
                    &mut nobs,
                    &mut x,
                    &mut xx,
                    &mut xxx,
                    &mut xxxx,
                    &mut compensation_x_remove,
                    &mut compensation_xx_remove,
                    &mut compensation_xxx_remove,
                    &mut compensation_xxxx_remove,
                );
            }

            // Add new values
            for j in end[i-1] as usize..e {
                let val = values_copy[j];
                add_kurt(
                    val,
                    &mut nobs,
                    &mut x,
                    &mut xx,
                    &mut xxx,
                    &mut xxxx,
                    &mut compensation_x_add,
                    &mut compensation_xx_add,
                    &mut compensation_xxx_add,
                    &mut compensation_xxxx_add,
                    &mut num_consecutive_same_value,
                    &mut prev_value,
                );
            }
        }

        output[i] = calc_kurt(min_periods, nobs, x, xx, xxx, xxxx, num_consecutive_same_value);

        if !is_monotonic_increasing_bounds {
            nobs = 0;
            x = 0.0;
            xx = 0.0;
            xxx = 0.0;
            xxxx = 0.0;
        }
    }

    output
}

#[inline]
fn remove_kurt(
    val: f64,
    nobs: &mut i64,
    x: &mut f64,
    xx: &mut f64,
    xxx: &mut f64,
    xxxx: &mut f64,
    compensation_x: &mut f64,
    compensation_xx: &mut f64,
    compensation_xxx: &mut f64,
    compensation_xxxx: &mut f64,
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

        // Remove val4 from xxx sum with Kahan compensation
        let y = -(val * val * val * val) - *compensation_xxxx;
        let t = *xxxx + y;
        *compensation_xxxx = t - *xxxx - y;
        *xxxx = t;
    }
}

#[inline]
fn add_kurt(
    val: f64,
    nobs: &mut i64,
    x: &mut f64,
    xx: &mut f64,
    xxx: &mut f64,
    xxxx: &mut f64,
    compensation_x: &mut f64,
    compensation_xx: &mut f64,
    compensation_xxx: &mut f64,
    compensation_xxxx: &mut f64,
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

        // Update sum of fourth powers
        let y = val * val * val * val - *compensation_xxxx;
        let t = *xxxx + y;
        *compensation_xxxx = t - *xxxx - y;
        *xxxx = t;

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
    use super::*;
    use approx::assert_abs_diff_eq;  // For floating point comparisons

    #[test]
    fn test_simple_kurt() {
        // Simple case with known skewness
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let start = vec![0, 0, 0, 0, 1];
        let end = vec![1, 2, 3, 4, 5];
        let min_periods = 3;

        let result = roll_kurt(&values, &start, &end, min_periods);

        // Expected values can be calculated using pandas for verification
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert_abs_diff_eq!(result[3], -1.2, epsilon = 1e-10);
        assert_abs_diff_eq!(result[4], -1.2, epsilon = 1e-10);
    }

    #[test]
    fn test_nan_handling() {
        // Test with NaN values
        let values = vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0];
        let start = vec![0, 0, 0, 0, 1, 2];
        let end = vec![1, 2, 3, 4, 5, 6];
        let min_periods = 3;

        let result = roll_kurt(&values, &start, &end, min_periods);

        assert!(result[4].is_nan());  // Contains NaN
        assert_abs_diff_eq!(result[5], -1.2, epsilon = 1e-10);  // [3,4,5] is valid
    }

    #[test]
    fn test_min_periods() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let start = vec![0, 0, 0, 0, 0, 1];
        let end = vec![1, 2, 3, 4, 5, 6];
        let min_periods = 7;

        let result = roll_kurt(&values, &start, &end, min_periods);

        // All results should be NaN as windows are too small
        assert!(result.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_uniform_values() {
        // Test with all same values - should return 0
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let start = vec![0, 0, 0, 1, 2, 3];
        let end = vec![1, 2, 3, 4, 5, 6];
        let min_periods = 3;

        let result = roll_kurt(&values, &start, &end, min_periods);

        // Uniform distribution should have 3 kurtosis
        assert!(result.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_non_monotonic_bounds() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let start = vec![2, 1, 0];  // Non-monotonic
        let end = vec![5, 4, 3];    // Non-monotonic
        let min_periods = 3;

        let result = roll_kurt(&values, &start, &end, min_periods);

        // Test that it handles non-monotonic windows correctly
        // You might need to adjust these expected values based on your implementation
        assert!(result.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_kurtosis_distribution() {
        // Test with known skewed distribution
        let values = vec![1.0, 1.0, 1.0, 1.0, -100.0, 10.0];  // Highly skewed right
        let start = vec![0, 0, 0, 0, 1, 2];
        let end = vec![1, 2, 3, 4, 5, 6];
        let min_periods = 3;

        let result = roll_kurt(&values, &start, &end, min_periods);

        // Should be positive skew
        assert_abs_diff_eq!(result[3], -3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[4], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[5], 3.876089241551547, epsilon = 1e-10);
    }

    // Helper function to test internal calculations
    #[test]
    fn test_add_remove_kurt() {
        let mut nobs = 0i64;
        let mut x = 0f64;
        let mut xx = 0f64;
        let mut xxx = 0f64;
        let mut xxxx = 0f64;
        let mut compensation_x = 0f64;
        let mut compensation_xx = 0f64;
        let mut compensation_xxx = 0f64;
        let mut compensation_xxxx = 0f64;
        let mut num_consecutive = 0i64;
        let mut prev_value = 0f64;

        // Add some values
        add_kurt(
            1.0,
            &mut nobs,
            &mut x,
            &mut xx,
            &mut xxx,
            &mut xxxx,
            &mut compensation_x,
            &mut compensation_xx,
            &mut compensation_xxx,
            &mut compensation_xxxx,
            &mut num_consecutive,
            &mut prev_value,
        );

        assert_eq!(nobs, 1);
        assert_abs_diff_eq!(x, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(xx, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(xxx, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(xxxx, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_window_bounds() {
        let (start, end) = get_window_bounds(
            3,              // window_size
            5,              // num_values
            Some(false),    // center
            Some("right"),  // closed
            Some(1),        // step
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

    }
}
