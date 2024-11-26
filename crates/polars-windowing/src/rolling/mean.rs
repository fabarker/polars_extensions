use std::ops::{Add, Div, Mul, Sub};
use crate::rolling::sum::SumWindowType;
use num_traits::Float;
use polars::prelude::series::AsSeries;
use polars_arrow::array::{ArrayRef, PrimitiveArray};
use polars_custom_utils::utils::weights::ExponentialDecayType;
use super::*;

pub fn ew_rolling_mean(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
    decay: &ExponentialDecayType,
) -> PolarsResult<Series> {
    let s = input.as_series().to_float()?;
    polars_core::with_match_physical_float_polars_type!(s.dtype(), |$T| {
        let chk_arr: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        apply_ew_rolling_aggregator_chunked(
            chk_arr,
            window_size,
            min_periods,
            center,
            decay,
            &calc_ew_rolling_mean,
        )
    })
}

pub fn rolling_mean(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let s = input.as_series().to_float()?;
    polars_core::with_match_physical_float_polars_type!(s.dtype(), |$T| {
        let chk_arr: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        apply_rolling_aggregator_chunked(
            chk_arr,
            window_size,
            min_periods,
            center,
            weights,
            &calc_rolling_mean,
        )
    })
}

fn calc_ew_rolling_mean<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    mut decay_type: ExponentialWeights<T>,
) -> ArrayRef
where
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat + std::ops::DivAssign,
{
    calc_ew_rolling_generic::<T, SumWindowType>(arr, window_size, min_periods, center, &decay_type.normalize())
}

fn calc_ew_rolling_generic<'a, T, W>(
    arr: &'a PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    decay_type: &'a ExponentialWeights<T>,
) -> ArrayRef
where
    T: NativeType + Float + iter::Sum<T> + SubAssign + AddAssign + IsFloat,
    W: WindowType<'a, T>,
{
    let offsets: OffsetFn = if center {
        det_offsets_center
    } else {
        det_offsets
    };

    match arr.has_nulls() {
        true => nulls::calc_ew_rolling_aggregator::<W::EWindow, T, OffsetFn>(
            arr.values().as_slice(),
            arr.validity(),
            window_size,
            min_periods,
            offsets,
            &decay_type,
        ),
        false => no_nulls::calc_ew_rolling_aggregator::<W::EWindow, T, OffsetFn>(
            arr.values().as_slice(),
            arr.validity(),
            window_size,
            min_periods,
            offsets,
            &decay_type,
        ),
    }
}

fn calc_rolling_mean<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat + std::ops::DivAssign,
{
    calc_rolling_generic::<T, MeanWindowType>(arr, window_size, min_periods, center, weights)
}

pub(crate) fn compute_sum_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: Sum<T> + Copy + Mul<Output = T>,
{
    values.iter().zip(weights).map(|(v, w)| *v * *w).sum()
}

// Implement for Mean
struct MeanWindowType;
impl<'a, T> WindowType<'a, T> for MeanWindowType
where
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat + std::ops::DivAssign,
{
    type Window = MeanWindow<'a, T>;
    type EWindow = ExponentialMeanWindow<'a, T>;
    fn get_weight_computer() -> fn(&[T], &[T]) -> T {
        compute_sum_weights
    }

    fn prepare_weights(weights: Vec<T>) -> Vec<T> {
        <MeanWindowType as WindowType<T>>::normalize_weights(weights)
    }
}

impl<'a,
    T: NativeType
    + IsFloat
    + Div<Output = T>
    + Add<Output = T>
    + Sub<Output = T>
    + iter::Sum
    + NumCast,
    > RollingAggWindow<'a, T> for MeanWindow<'a, T>
{
    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize) -> Self {
        Self {
            sum: SumWindow::new(slice, validity, start, end),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        if self.sum.null_count > 0 {
            self.update_nulls(start, end)
        } else {
            self.update_no_nulls(start, end)
        }
    }

    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        //let sum = self.sum.update_nulls(start, end).unwrap_unchecked();
        //Some(sum / NumCast::from(end - start - self.sum.null_count).unwrap())
        let sum = self.sum.update_nulls(start, end);
        sum.map(|sum| sum / NumCast::from(end - start - self.sum.null_count).unwrap())
    }

    unsafe fn update_no_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let sum = self.sum.update_no_nulls(start, end).unwrap_unchecked();
        Some(sum / NumCast::from(end - start).unwrap())
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.sum.is_valid(min_periods)
    }

    fn window_type() -> &'static str {
        "mean"
    }
}



impl<'a,
    T: NativeType
    + IsFloat
    + Div<Output = T>
    + Add<Output = T>
    + Sub<Output = T>
    + Sum
    + NumCast
    + Mul<Output = T>
    + DivAssign
    + num_traits::Zero,
> EWRollingAggWindow<'a, T> for ExponentialMeanWindow<'a, T>
{
    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize, weights: &'a ExponentialWeights<T>) -> Self {
        Self {
            sum: ExponentialSumWindow::new(slice, validity, start, end, weights),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        if self.sum.null_count > 0 {
            self.update_nulls(start, end)
        } else {
            self.update_no_nulls(start, end)
        }
    }

    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        //let sum = self.sum.update_nulls(start, end).unwrap_unchecked();
        //Some(sum / NumCast::from(end - start - self.sum.null_count).unwrap())
        let sum = self.sum.update_nulls(start, end);
        sum.map(|sum| sum / NumCast::from(end - start - self.sum.null_count).unwrap())
    }

    unsafe fn update_no_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let sum = self.sum.update_no_nulls(start, end).unwrap_unchecked();
        Some(sum / NumCast::from(end - start).unwrap())
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.sum.is_valid(min_periods)
    }

    fn window_type() -> &'static str {
        "mean"
    }
}