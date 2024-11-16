use std::ops::Add;
use std::ops::Sub;
use polars::prelude::{PolarsResult, SeriesSealed};
use polars_core::prelude::{ChunkedArray, RollingOptionsFixedWindow, Series};
use crate::rolling::{rolling_aggregator, ProdWindow};
use crate::with_match_physical_float_polars_type;
use crate::{Float32Type, Float64Type};
use crate::DataType;

pub fn rolling_cagr_with_opts(input: &Series,
                              options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
    rolling_cagr(input, options.window_size, options.min_periods, options.center, options.weights)
}

pub fn rolling_cagr(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {

    let s = &input.as_series().add(1.0);
    with_match_physical_float_polars_type!(
        s.dtype(),
        |T U| {
            let ca: &ChunkedArray<U> = s.as_ref().as_ref().as_ref();
            Ok(rolling_aggregator::<ProdWindow<T>, T, U>(
            ca,
            window_size,
            min_periods,
            center,
            weights)?.sub(1.0))
            }
        )
}
