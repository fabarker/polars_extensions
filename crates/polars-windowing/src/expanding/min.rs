use polars::prelude::{PolarsResult, *};
use polars_core::prelude::Series;

pub fn expanding_min(series: &Series, min_periods: usize) -> PolarsResult<Series> {
    // Cast to Float64 ChunkedArray
    let arr = series.f64()?;
    let len = arr.len();
    let mut result = Vec::with_capacity(len);
    let mut current_min = f64::MAX;

    // Handle null values in the input array
    for (idx, opt_val) in arr.into_iter().enumerate() {
        if let Some(val) = opt_val {
            current_min = current_min.min(val);
            result.push(Some(current_min));
        } else {
            // If we encounter a null value, keep the previous min
            result.push(if idx > 0 { Some(current_min) } else { None });
        }

        if idx < (min_periods - 1) {
            if let Some(last) = result.last_mut() {
                *last = None;
            }
        }
    }

    // Convert back to Series with same name as input
    Ok(Series::new(series.name().clone(), result))
}
