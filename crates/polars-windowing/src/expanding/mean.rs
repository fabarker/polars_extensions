use std::ops::{Add, Div, Mul, Sub};

use polars::prelude::series::AsSeries;
use polars_core::datatypes::{Float32Type, Float64Type};

use super::*;

pub fn expanding_mean(
    input: &Series,
    min_periods: usize,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let s = input.as_series().to_float()?;
    polars_core::with_match_physical_float_polars_type!(s.dtype(), |$T| {
        let chk_arr: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        apply_expanding_aggregator_chunked(
            chk_arr,
            min_periods,
            weights,
            &calc_expanding_mean,
        )
    })
}

fn calc_expanding_mean<T>(
    arr: &PrimitiveArray<T>,
    min_periods: usize,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + Float + iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    calc_expanding_generic::<T, MeanWindowType>(arr, min_periods, weights)
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
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat,
{
    type Window = MeanWindow<'a, T>;
    fn get_weight_computer() -> fn(&[T], &[T]) -> T {
        compute_sum_weights
    }

    fn prepare_weights(weights: Vec<T>) -> Vec<T> {
        <MeanWindowType as WindowType<T>>::normalize_weights(weights)
    }
}

/*
pub fn expanding_mean(
    input: &Series,
    min_periods: usize,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let s = input.as_series().to_float()?;
    with_match_physical_float_polars_type!(
    s.dtype(),
    |T U| {
        let ca: &ChunkedArray<U> = s.as_ref().as_ref().as_ref();
        expanding_aggregator::<MeanWindow<T>, T, U>(  // Note: using $_ as per macro definition
        ca,
        min_periods,
        weights)
        }
    )
}

 */

impl<
        'a,
        T: NativeType + IsFloat + Div<Output = T> + Add<Output = T> + Sub<Output = T> + Sum + NumCast,
    > ExpandingAggWindow<'a, T> for MeanWindow<'a, T>
{
    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        if self.sum.null_count > 0 {
            self.update_nulls(start, end)
        } else {
            self.update_no_nulls(start, end)
        }
    }

    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let sum = self.sum.update_nulls(start, end).unwrap_unchecked();
        Some(sum / NumCast::from(end - start - self.sum.null_count).unwrap())
    }

    unsafe fn update_no_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let sum = self.sum.update_no_nulls(start, end).unwrap_unchecked();
        Some(sum / NumCast::from(end - start).unwrap())
    }

    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize) -> Self {
        Self {
            sum: SumWindow::new(slice, validity, start, end),
        }
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.sum.is_valid(min_periods)
    }

    fn window_type() -> &'static str {
        "mean"
    }
}
