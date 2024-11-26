use super::*;
use crate::SeriesRollingExt;

pub fn rolling_ewm_std(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
    decay: ExponentialDecayType,
) -> PolarsResult<Series> {
    input
        .rolling(window_size)
        .with_min_periods(min_periods)
        .with_center(center)
        .ewm(decay, false)
        .var()
        .map(|mut s| {
            // 2. Match on the data type of the series
            match s.dtype().clone() {
                // 3. If it's Float32
                Float32 => {
                    // Get mutable reference to inner Float32 array
                    let ca: &mut ChunkedArray<Float32Type> = s._get_inner_mut().as_mut();
                    // Apply square root (0.5 power) to each value in place
                    ca.apply_mut(|v| v.powf(0.5))
                },
                // 4. If it's Float64
                Float64 => {
                    // Get mutable reference to inner Float64 array
                    let ca: &mut ChunkedArray<Float64Type> = s._get_inner_mut().as_mut();
                    // Apply square root to each value in place
                    ca.apply_mut(|v| v.powf(0.5))
                },
                // 5. This shouldn't happen as variance should always be float
                _ => unreachable!(),
            }
            // 6. Return the modified series
            s
        })
}

pub fn rolling_std(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    input
        .rolling(window_size)
        .with_min_periods(min_periods)
        .with_center(center)
        .with_weights(weights)
        .var()
        .map(|mut s| {
            // 2. Match on the data type of the series
            match s.dtype().clone() {
                // 3. If it's Float32
                Float32 => {
                    // Get mutable reference to inner Float32 array
                    let ca: &mut ChunkedArray<Float32Type> = s._get_inner_mut().as_mut();
                    // Apply square root (0.5 power) to each value in place
                    ca.apply_mut(|v| v.powf(0.5))
                },
                // 4. If it's Float64
                Float64 => {
                    // Get mutable reference to inner Float64 array
                    let ca: &mut ChunkedArray<Float64Type> = s._get_inner_mut().as_mut();
                    // Apply square root to each value in place
                    ca.apply_mut(|v| v.powf(0.5))
                },
                // 5. This shouldn't happen as variance should always be float
                _ => unreachable!(),
            }
            // 6. Return the modified series
            s
        })
}
