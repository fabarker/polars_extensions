use std::ops::{Div, Mul};

use polars::prelude::series::AsSeries;

use super::*;
use crate::{with_match_physical_float_polars_type, DataType};

pub fn rolling_prod(
    input: &Series,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series> {
    let s = input.as_series().clone();
    with_match_physical_float_polars_type!(
    input.dtype(),
    |T U| {
        let ca: &ChunkedArray<U> = s.as_ref().as_ref().as_ref();
        rolling_aggregator::<ProdWindow<T>, T, U>(
        ca,
        window_size,
        min_periods,
        center,
        weights)
        }
    )
}

impl<'a, T: NativeType + IsFloat + Mul<Output = T> + Div<Output = T>> ProdWindow<'a, T> {
    // compute sum from the entire window
    unsafe fn compute_prod_and_null_count(&mut self, start: usize, end: usize) -> Option<T> {
        let mut prod = None;
        let mut idx = start;
        self.null_count = 0;
        for value in &self.slice[start..end] {
            let valid = match self.validity {
                None => true,
                Some(bitmap) => bitmap.get_bit_unchecked(idx),
            };

            if valid {
                match prod {
                    None => prod = Some(*value),
                    Some(current) => prod = Some(*value * current),
                }
            } else {
                self.null_count += 1;
            }
            idx += 1;
        }
        self.prod = prod;
        prod
    }
}

impl<'a, T: NativeType + IsFloat + Mul<Output = T> + Div<Output = T> + iter::Product>
    RollingAggWindow<'a, T> for ProdWindow<'a, T>
{
    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize) -> Self {
        let mut out = Self {
            slice,
            validity,
            prod: None,
            last_start: start,
            last_end: end,
            null_count: 0,
        };
        out.compute_prod_and_null_count(start, end);
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
        let recompute_prod = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_prod = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let leaving_value = self.slice.get_unchecked(idx);

                if T::is_float() && !leaving_value.is_finite() {
                    recompute_prod = true;
                    break;
                }

                self.prod = self.prod.map(|v| v / *leaving_value);
            }
            recompute_prod
        };
        self.last_start = start;

        // we traverse all values and compute
        if recompute_prod {
            self.prod = Some(
                self.slice
                    .get_unchecked(start..end)
                    .iter()
                    .copied()
                    .product::<T>(),
            );
        }
        // remove leaving values.
        else {
            for idx in self.last_end..end {
                let value = *self.slice.get_unchecked(idx);
                self.prod = Some(self.prod.unwrap() * value)
            }
        }
        self.last_end = end;
        Some(self.prod?)
    }

    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T> {
        let recompute_prod = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_prod = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let valid = self.validity?.get_bit_unchecked(idx);
                if valid {
                    let leaving_value = self.slice.get_unchecked(idx);

                    // if the leaving value is nan we need to recompute the window
                    if T::is_float() && !leaving_value.is_finite() {
                        recompute_prod = true;
                        break;
                    }
                    self.prod = self.prod.map(|v| v / *leaving_value)
                } else {
                    // null value leaving the window
                    self.null_count -= 1;

                    // self.sum is None and the leaving value is None
                    // if the entering value is valid, we might get a new sum.
                    if self.prod.is_none() {
                        recompute_prod = true;
                        break;
                    }
                }
            }
            recompute_prod
        };

        self.last_start = start;

        // we traverse all values and compute
        if recompute_prod {
            self.compute_prod_and_null_count(start, end);
        } else {
            for idx in self.last_end..end {
                let valid = self.validity?.get_bit_unchecked(idx);

                if valid {
                    let value = *self.slice.get_unchecked(idx);
                    match self.prod {
                        None => self.prod = Some(value),
                        Some(current) => self.prod = Some(current * value),
                    }
                } else {
                    // null value entering the window
                    self.null_count += 1;
                }
            }
        }
        self.last_end = end;
        self.prod
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }

    fn window_type() -> &'static str {
        "prod"
    }
}
