use std::ops::{Add, Mul, Sub};

use num::{Float, One, Zero};
use polars::prelude::series::AsSeries;

use super::*;
use crate::{with_match_physical_float_polars_type, DataType};

pub fn expanding_var(
    input: &Series,
    min_periods: usize,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let s = input.as_series().to_float()?;
    with_match_physical_float_polars_type!(
    s.dtype(),
    |T U| {
        let ca: &ChunkedArray<U> = s.as_ref().as_ref().as_ref();
        expanding_aggregator::<VarWindow<T>, T, U>(  // Note: using $_ as per macro definition
        ca,
        min_periods,
        weights)
        }
    )
}

impl<'a, T: NativeType + IsFloat + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>
    SumSquaredWindow<'a, T>
{
    // compute sum from the entire window
    unsafe fn compute_sum_and_null_count(&mut self, start: usize, end: usize) -> Option<T> {
        let mut sum_of_squares = None;
        let mut idx = start;
        self.null_count = 0;
        for value in &self.slice[start..end] {
            let valid = match self.validity {
                None => true,
                Some(bitmap) => bitmap.get_bit_unchecked(idx),
            };

            if valid {
                match sum_of_squares {
                    None => sum_of_squares = Some(*value * *value),
                    Some(current) => sum_of_squares = Some(*value * *value + current),
                }
            } else {
                self.null_count += 1;
            }
            idx += 1;
        }
        self.sum_of_squares = sum_of_squares;
        sum_of_squares
    }
}

impl<
        'a,
        T: NativeType + IsFloat + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + iter::Sum,
    > ExpandingAggWindow<'a, T> for SumSquaredWindow<'a, T>
{
    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize) -> Self {
        let mut out = Self {
            slice,
            validity,
            sum_of_squares: None,
            last_start: start,
            last_end: end,
            null_count: 0,
            last_recompute: 0,
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
        let recompute_sum = if start >= self.last_end || self.last_recompute > 128 {
            self.last_recompute = 0;
            true
        } else {
            self.last_recompute += 1;
            // remove elements that should leave the window
            let mut recompute_sum = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let leaving_value = *self.slice.get_unchecked(idx);
                if T::is_float() && !leaving_value.is_finite() {
                    recompute_sum = true;
                    break;
                }

                self.sum_of_squares = self
                    .sum_of_squares
                    .map(|v| v - (leaving_value * leaving_value))
            }
            recompute_sum
        };

        self.last_start = start;

        // we traverse all values and compute
        if T::is_float() && recompute_sum {
            self.sum_of_squares = Some(
                self.slice
                    .get_unchecked(start..end)
                    .iter()
                    .map(|v| *v * *v)
                    .sum::<T>(),
            );
        } else {
            for idx in self.last_end..end {
                let entering_value = *self.slice.get_unchecked(idx);
                self.sum_of_squares =
                    Some(self.sum_of_squares.unwrap() + (entering_value * entering_value))
            }
        }
        self.last_end = end;
        Some(self.sum_of_squares?)
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
                    let leaving_value = *self.slice.get_unchecked(idx);

                    // if the leaving value is nan we need to recompute the window
                    if T::is_float() && !leaving_value.is_finite() {
                        recompute_sum = true;
                        break;
                    }
                    self.sum_of_squares = self
                        .sum_of_squares
                        .map(|v| v - leaving_value * leaving_value)
                } else {
                    // null value leaving the window
                    self.null_count -= 1;

                    // self.sum is None and the leaving value is None
                    // if the entering value is valid, we might get a new sum.
                    if self.sum_of_squares.is_none() {
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
                    let value = value * value;
                    match self.sum_of_squares {
                        None => self.sum_of_squares = Some(value),
                        Some(current) => self.sum_of_squares = Some(current + value),
                    }
                } else {
                    // null value entering the window
                    self.null_count += 1;
                }
            }
        }
        self.last_end = end;
        self.sum_of_squares
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }

    fn window_type() -> &'static str {
        ""
    }
}

impl<
        'a,
        T: Zero
            + One
            + Float
            + NativeType
            + IsFloat
            + Mul<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + iter::Sum,
    > ExpandingAggWindow<'a, T> for VarWindow<'a, T>
{
    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize) -> Self {
        Self {
            mean: MeanWindow::new(slice, validity, start, end),
            sum_of_squares: SumSquaredWindow::new(slice, validity, start, end),
            ddof: 1u8,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        if self.sum_of_squares.null_count > 0 {
            self.update_nulls(start, end)
        } else {
            self.update_no_nulls(start, end)
        }
    }

    unsafe fn update_no_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let count: T = NumCast::from(end - start).unwrap();
        let sum_of_squares = self.sum_of_squares.update(start, end).unwrap_unchecked();
        let mean = self.mean.update(start, end).unwrap_unchecked();

        let denom = count - NumCast::from(self.ddof).unwrap();
        if denom <= T::zero() {
            None
        } else if end - start == 1 {
            Some(T::zero())
        } else {
            let out = (sum_of_squares - count * mean * mean) / denom;
            // variance cannot be negative.
            // if it is negative it is due to numeric instability
            if out < T::zero() {
                Some(T::zero())
            } else {
                Some(out)
            }
        }
    }

    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let sum_of_squares = self.sum_of_squares.update(start, end)?;
        let null_count = self.sum_of_squares.null_count;
        let count: T = NumCast::from(end - start - null_count).unwrap();

        let mean = self.mean.update(start, end)?;
        let ddof = NumCast::from(self.ddof).unwrap();

        let denom = count - ddof;

        if count == T::zero() {
            None
        } else if count == T::one() {
            NumCast::from(0)
        } else if denom <= T::zero() {
            Some(T::infinity())
        } else {
            let var = (sum_of_squares - count * mean * mean) / denom;
            Some(if var < T::zero() { T::zero() } else { var })
        }
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.mean.is_valid(min_periods)
    }

    fn window_type() -> &'static str {
        "variance"
    }
}
