use polars::prelude::PolarsResult;
use polars_arrow::legacy::prelude::QuantileInterpolOptions;
use polars_core::prelude::{NamedFrom, Series};

use crate::rolling::quantile::*;

pub fn expanding_quantile(
    input: &Series,
    quantile: f64,
    min_periods: usize,
    interpolation: QuantileInterpolOptions,
) -> PolarsResult<Series> {
    let x = input.len() as i64;

    if x == 0 {
        return Ok(Series::new("name_of_series".into(), Vec::<f64>::new()));
    }

    // For expanding window, the window size grows with each position
    // Start indices are always 0, end indices increase by 1
    let start = vec![0; x as usize]; // All windows start at 0
    let end: Vec<i64> = (1..=x).collect(); // End points grow: 1,2,3,...,n

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
