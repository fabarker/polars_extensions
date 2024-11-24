use std::ops::{Add, Sub};

use polars::prelude::{PolarsResult, SeriesSealed};
use polars_core::prelude::{RollingOptionsFixedWindow, Series};

use crate::rolling::sum::rolling_sum;
use polars_custom_utils::utils::ts::ReturnsType;


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
        "simple",
    )
}


pub fn rolling_cagr(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
    returns_type: &str,
) -> PolarsResult<Series> {

    // Convert input returns series to log returns...
    let log_rtn = ReturnsType::from(returns_type).to_log(&input)?;
    let cmrtn = rolling_sum(&log_rtn, window_size, min_periods, center, weights)?;
    Ok(ReturnsType::log_to_linear(&cmrtn)?)
}
