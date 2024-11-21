use std::ops::{Add, Sub};
use polars::prelude::{PolarsResult, SeriesSealed};
use polars_core::prelude::Series;
use crate::expanding::prod::expanding_prod;

pub fn expanding_cagr(
    input: &Series,
    min_periods: usize,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let s = &input.as_series().add(1.0);
    let mut prod = expanding_prod(s, min_periods, weights)?;
    Ok(prod.sub(1.0))
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