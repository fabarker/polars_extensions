use std::ops::{Add, Mul, Sub};
use num_traits::Zero;
use polars::prelude::series::AsSeries;
use super::*;

pub fn ew_rolling_sum(
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
            &calc_ew_rolling_sum,
        )
    })
}

pub fn rolling_sum(
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
            &calc_rolling_sum,
        )
    })
}

fn calc_ew_rolling_sum<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    decay: ExponentialWeights<T>,
) -> ArrayRef
where
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat + DivAssign,
{
    calc_ew_rolling_generic::<T, SumWindowType>(arr, window_size, min_periods, center, &decay)
}

fn calc_rolling_sum<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat + std::ops::DivAssign,
{
    calc_rolling_generic::<T, SumWindowType>(arr, window_size, min_periods, center, weights)
}

pub(crate) fn compute_sum_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: Sum<T> + Copy + Mul<Output = T>,
{
    values.iter().zip(weights).map(|(v, w)| *v * *w).sum()
}

// Implement for Mean
pub(crate) struct SumWindowType;
impl<'a, T> WindowType<'a, T> for SumWindowType
where
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat + DivAssign,
{
    type Window = SumWindow<'a, T>;
    type EWindow = ExponentialSumWindow<'a, T>;
    fn get_weight_computer() -> fn(&[T], &[T]) -> T {
        compute_sum_weights
    }

    fn prepare_weights(weights: Vec<T>) -> Vec<T> {
        weights
    }
}

/*
pub fn rolling_sum(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let s = input.as_series().to_float()?;
    with_match_physical_float_polars_type!(
    s.dtype(),
    |T U| {
        let ca: &ChunkedArray<U> = s.as_ref().as_ref().as_ref();
        rolling_aggregator::<SumWindow<T>, T, U>(  // Note: using $_ as per macro definition
        ca,
        window_size,
        min_periods,
        center,
        weights)
        }
    )
}
 */

impl<'a, T: NativeType + IsFloat + Add<Output = T> + Sub<Output = T>> SumWindow<'a, T> {
    // compute sum from the entire window
    unsafe fn compute_sum_and_null_count(&mut self, start: usize, end: usize) -> Option<T> {
        let mut sum = None;
        let mut idx = start;
        self.null_count = 0;
        for value in &self.slice[start..end] {
            let valid = match self.validity {
                None => true,
                Some(bitmap) => bitmap.get_bit_unchecked(idx),
            };

            if valid {
                match sum {
                    None => sum = Some(*value),
                    Some(current) => sum = Some(*value + current),
                }
            } else {
                self.null_count += 1;
            }
            idx += 1;
        }
        self.sum = sum;
        sum
    }
}

impl<'a, T: NativeType + IsFloat + Add<Output = T> + Sub<Output = T> + std::iter::Sum>
    RollingAggWindow<'a, T> for SumWindow<'a, T>
{
    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize) -> Self {
        let mut out = Self {
            slice,
            validity,
            sum: None,
            last_start: start,
            last_end: end,
            null_count: 0,
        };
        out.compute_sum_and_null_count(start, end);
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        if self.null_count > 0 {
            self.update_nulls(start, end)
        } else {
            self.update_no_nulls(start, end)
        }
    }

    unsafe fn update_no_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        // if we exceed the end, we have a completely new window
        // so we recompute
        let recompute_sum = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_sum = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let leaving_value = self.slice.get_unchecked(idx);

                if T::is_float() && !leaving_value.is_finite() {
                    recompute_sum = true;
                    break;
                }

                self.sum = self.sum.map(|v| v - *leaving_value);
            }
            recompute_sum
        };
        self.last_start = start;

        // we traverse all values and compute
        if recompute_sum {
            self.sum = Some(
                self.slice
                    .get_unchecked(start..end)
                    .iter()
                    .copied()
                    .sum::<T>(),
            );
        }
        // remove leaving values.
        else {
            for idx in self.last_end..end {
                let value = *self.slice.get_unchecked(idx);
                self.sum = Some(self.sum.unwrap() + value)
            }
        }
        self.last_end = end;
        Some(self.sum?)
    }

    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let recompute_sum = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_sum = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let valid = self.validity?.get_bit_unchecked(idx);
                if valid {
                    let leaving_value = self.slice.get_unchecked(idx);

                    // if the leaving value is nan we need to recompute the window
                    if T::is_float() && !leaving_value.is_finite() {
                        recompute_sum = true;
                        break;
                    }
                    self.sum = self.sum.map(|v| v - *leaving_value)
                } else {
                    // null value leaving the window
                    self.null_count -= 1;

                    // self.sum is None and the leaving value is None
                    // if the entering value is valid, we might get a new sum.
                    if self.sum.is_none() {
                        recompute_sum = true;
                        break;
                    }
                }
            }
            recompute_sum
        };

        self.last_start = start;

        // we traverse all values and compute
        if recompute_sum {
            self.compute_sum_and_null_count(start, end);
        } else {
            for idx in self.last_end..end {
                let valid = match self.validity {
                    None => true,
                    Some(bitmap) => bitmap.get_bit_unchecked(idx),
                };

                if valid {
                    let value = *self.slice.get_unchecked(idx);
                    match self.sum {
                        None => self.sum = Some(value),
                        Some(current) => self.sum = Some(current + value),
                    }
                } else {
                    // null value entering the window
                    self.null_count += 1;
                }
            }
        }
        self.last_end = end;
        self.sum
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }

    fn window_type() -> &'static str {
        "sum"
    }
}

/////////////////////////////


impl<'a, T> ExponentialSumWindow<'a, T>
where
    T: NativeType
    + IsFloat
    + Mul<Output = T>
    + Div<Output = T>
    + Add<Output = T>
    + Sub<Output = T>
    + DivAssign
    + NumCast
    + Zero
    + Sum,
{
    unsafe fn compute_sum_and_null_count(&mut self, start: usize, end: usize) -> Option<T> {

        let mut sum = None;
        let mut idx = start;
        let mut wdx = self.weights.len();
        self.null_count = 0;

        for value in &self.slice[start..end] {

            // Check if we have a bitmap in place,
            // should we have it, we are dealing with nulls,
            // so check if the val in idx is a null
            let valid = match self.validity {
                None => true,
                Some(bitmap) => bitmap.get_bit_unchecked(idx),
            };

            // If we have a valid value
            if valid {
                match sum {
                    None => sum = Some(*value * *&self.weights.get_weight_unchecked(wdx-1)),
                    Some(current) => sum = Some(*value * *&self.weights.get_weight_unchecked(wdx-1) + current),
                }
            } else {
                self.null_count += 1;
            }
            idx += 1;
            wdx -= 1;
        }
        self.sum = sum;
        sum
    }
}

impl<'a, T> EWRollingAggWindow<'a, T> for ExponentialSumWindow<'a, T>
where
    T: NativeType
    + IsFloat
    + Add<Output = T>
    + Sub<Output = T>
    + Mul<Output = T>
    + Div<Output = T>
    + DivAssign
    + NumCast
    + Zero
    + Sum,
{
    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize, weights: &'a ExponentialWeights<T>) -> Self {
        let mut out = Self {
            slice,
            validity,
            sum: None,
            last_start: start,
            last_end: end,
            null_count: 0,
            weights
        };
        out.compute_sum_and_null_count(start, end);
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        if self.null_count > 0 {
            self.update_nulls(start, end)
        } else {
            self.update_no_nulls(start, end)
        }
    }

    unsafe fn update_no_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        // if we exceed the end, we have a completely new window
        // so we recompute
        let recompute_sum = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_sum = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let leaving_value = self.slice.get_unchecked(idx);
                let leaving_wt = self.weights.get_weight_unchecked(idx - self.last_start);

                if T::is_float() && !leaving_value.is_finite() {
                    recompute_sum = true;
                    break;
                }
                self.sum = self.sum.map(|v| v - *leaving_value * leaving_wt);
            }
            recompute_sum
        };
        self.last_start = start;

        // we traverse all values and compute
        if recompute_sum {
            self.compute_sum_and_null_count(start, end);
        }
        // remove leaving values.
        else {
            for idx in self.last_end..end {
                let value = *self.slice.get_unchecked(idx) * self.weights.leading_weight();
                self.sum = Some(self.sum.unwrap() / self.weights.normalizer + value)
            }
        }
        self.last_end = end;
        Some(self.sum?)
    }

    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let recompute_sum = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_sum = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let valid = self.validity?.get_bit_unchecked(idx);
                if valid {
                    let leaving_value = self.slice.get_unchecked(idx);
                    let leaving_wt = self.weights.get_weight_unchecked(idx - self.last_start);

                    // if the leaving value is nan we need to recompute the window
                    if T::is_float() && !leaving_value.is_finite() {
                        recompute_sum = true;
                        break;
                    }
                    self.sum = self.sum.map(|v| v - *leaving_value * leaving_wt);
                } else {
                    // null value leaving the window
                    self.null_count -= 1;

                    // self.sum is None and the leaving value is None
                    // if the entering value is valid, we might get a new sum.
                    if self.sum.is_none() {
                        recompute_sum = true;
                        break;
                    }
                }
            }
            recompute_sum
        };

        self.last_start = start;

        // we traverse all values and compute
        if recompute_sum {
            self.compute_sum_and_null_count(start, end);
        } else {
            for idx in self.last_end..end {
                let valid = match self.validity {
                    None => true,
                    Some(bitmap) => bitmap.get_bit_unchecked(idx),
                };

                if valid {
                    let value = *self.slice.get_unchecked(idx) * self.weights.leading_weight();
                    match self.sum {
                        None => self.sum = Some(value),
                        Some(current) => self.sum = Some(current / self.weights.normalizer + value),
                    }
                } else {
                    // null value entering the window
                    self.null_count += 1;
                }
            }
        }
        self.last_end = end;
        self.sum
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }

    fn window_type() -> &'static str {
        "sum"
    }
}

