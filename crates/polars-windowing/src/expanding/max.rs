use polars::prelude::*;

pub fn expanding_max(series: &Series,
                     min_periods: usize) -> PolarsResult<Series> {

    // Cast to Float64 ChunkedArray
    let arr = series.f64()?;
    let len = arr.len();
    let mut result = Vec::with_capacity(len);
    let mut current_max = f64::MIN;

    // Handle null values in the input array
    for (idx, opt_val) in arr.into_iter().enumerate() {
        if let Some(val) = opt_val {
            current_max = current_max.max(val);
            result.push(Some(current_max));
        } else {
            // If we encounter a null value, keep the previous max
            result.push(if idx > 0 {
                Some(current_max)
            } else {
                None
            });
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