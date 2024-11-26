use polars::prelude::{PolarsResult};
use polars_core::prelude::Series;
use polars_custom_utils::utils::ts::ReturnsType;
use crate::expanding::sum::expanding_sum;

pub fn expanding_cagr(
    input: &Series,
    min_periods: usize,
    weights: Option<Vec<f64>>,
    returns_type: ReturnsType,
) -> PolarsResult<Series> {

    let log_rtn = ReturnsType::from(returns_type).to_log(&input)?;
    let cmrtn = expanding_sum(&log_rtn, min_periods, weights)?;
    Ok(ReturnsType::log_to_linear(&cmrtn)?)
}

/*
pub fn expanding_cagr(
    input: &Series,
    min_periods: usize,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let s = &input.as_series().add(1.0);
    with_match_physical_float_polars_type!(
    s.dtype(),
    |T U| {
        let ca: &ChunkedArray<U> = s.as_ref().as_ref().as_ref();
        Ok(expanding_aggregator::<ProdWindow<T>, T, U>(
        ca,
        min_periods,
        weights)?.sub(1.0))
        }
    )
}
*/
