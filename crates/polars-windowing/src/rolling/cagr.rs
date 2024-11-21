use std::ops::{Add, Sub};
use polars::prelude::{PolarsResult, SeriesSealed};
use polars_core::prelude::{RollingOptionsFixedWindow, Series};
use crate::rolling::prod::rolling_prod;

pub fn rolling_cagr_with_opts(
    input: &Series,
    options: RollingOptionsFixedWindow,
) -> PolarsResult<Series> {
    rolling_cagr(
        input,
        options.window_size,
        options.min_periods,
        options.center,
        options.weights,
    )
}

pub fn rolling_cagr(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let s = &input.as_series().add(1.0);
    let mut prod = rolling_prod(s, window_size, min_periods, center, weights)?;
    Ok(prod.sub(1.0))
}
