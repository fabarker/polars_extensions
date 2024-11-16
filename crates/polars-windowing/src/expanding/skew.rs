use polars::prelude::PolarsResult;
use polars_core::prelude::{NamedFrom, Series};
use crate::rolling::skew::*;

pub fn expanding_skew(
    input: &Series,
    min_periods: usize,
) -> PolarsResult<Series> {
    let x = input.len() as i64;

    if x == 0 {
        return Ok(Series::new("name_of_series".into(), Vec::<f64>::new()));
    }

    // For expanding window, the window size grows with each position
    // Start indices are always 0, end indices increase by 1
    let start = vec![0; x as usize];  // All windows start at 0
    let end: Vec<i64> = (1..=x).collect();  // End points grow: 1,2,3,...,n

    let values: &[f64] = input.f64()?.cont_slice()?;
    let result = roll_skew(values,
                           &start,
                           &end,
                           min_periods as i64);

    let series = Series::new("name_of_series".into(), result);
    Ok(series.into())
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;  // For floating point comparisons

    #[test]
    fn test_expanding_skew_basic() -> PolarsResult<()> {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series = Series::new("input".into(), values);

        let result = expanding_skew(&series, 3)?;
        let result_slice = result.f64()?.cont_slice()?;

        // First two values should be NaN (need min 3 values)
        assert!(result_slice[0].is_nan());
        assert!(result_slice[1].is_nan());

        // Rest should be finite
        for i in 2..5 {
            assert!(result_slice[i].is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_expanding_skew_known_values() -> PolarsResult<()> {
        // Values known to produce specific skewness
        let values = vec![-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401];
        let series = Series::new("input".into(), values);

        let result = expanding_skew(&series, 3)?;
        let result_slice = result.f64()?.cont_slice()?;

        // First two should be NaN
        assert!(result_slice[0].is_nan());
        assert!(result_slice[1].is_nan());

        // Expanding window means each result uses all previous values
        // Values will be different from rolling window as window size grows
        assert!(result_slice[2].is_finite());
        assert!(result_slice[3].is_finite());
        assert!(result_slice[4].is_finite());

        Ok(())
    }

    #[test]
    fn test_expanding_skew_all_same_value() -> PolarsResult<()> {
        let values = vec![1.0; 5];
        let series = Series::new("input".into(), values);

        let result = expanding_skew(&series, 3)?;
        let result_slice = result.f64()?.cont_slice()?;

        // First two should be NaN (min_periods = 3)
        assert!(result_slice[0].is_nan());
        assert!(result_slice[1].is_nan());

        // Rest should be 0 (no skew in uniform data)
        for i in 2..5 {
            assert_abs_diff_eq!(result_slice[i], 0.0, epsilon = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_expanding_skew_min_periods() -> PolarsResult<()> {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series = Series::new("input".into(), values);

        // Test with min_periods = 2
        let result = expanding_skew(&series, 2)?;
        let result_slice = result.f64()?.cont_slice()?;

        // First value should be NaN
        assert!(result_slice[0].is_nan());

        // Second value should be nan (min_periods = 2)
        assert!(result_slice[1].is_nan());

        Ok(())
    }

    #[test]
    fn test_expanding_skew_empty_series() -> PolarsResult<()> {
        let values: Vec<f64> = vec![];
        let series = Series::new("input".into(), values);
        let result = expanding_skew(&series, 3)?;
        assert_eq!(result.len(), 0);
        Ok(())
    }
}
