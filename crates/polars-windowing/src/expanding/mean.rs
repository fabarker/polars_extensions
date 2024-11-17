use std::ops::{Add, Div, Sub};

use polars::prelude::series::AsSeries;

use super::*;
use crate::{with_match_physical_float_polars_type, DataType};

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

impl<
        'a,
        T: NativeType
            + IsFloat
            + Div<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + iter::Sum
            + NumCast,
    > ExpandingAggWindow<'a, T> for MeanWindow<'a, T>
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

    unsafe fn update_no_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let sum = self.sum.update_no_nulls(start, end).unwrap_unchecked();
        Some(sum / NumCast::from(end - start).unwrap())
    }

    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let sum = self.sum.update_nulls(start, end).unwrap_unchecked();
        Some(sum / NumCast::from(end - start - self.sum.null_count).unwrap())
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.sum.is_valid(min_periods)
    }

    fn window_type() -> &'static str {
        "mean"
    }
}
