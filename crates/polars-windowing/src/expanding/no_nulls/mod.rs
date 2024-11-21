use std::fmt::Debug;
use std::iter;
use std::ops::Mul;
use arrow::array::Array;
use num_traits::{Num, NumCast, One, Zero};
use polars::prelude::PolarsResult;
use polars_arrow::array::{ArrayRef, PrimitiveArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::legacy::utils::CustomIterTools;
use polars_arrow::types::NativeType;
use crate::rolling::sum::compute_sum_weights;
use super::*;

/*
pub fn expanding_aggregator_no_nulls<'a, Agg, T>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    min_periods: usize,
    weights: Option<&[f64]>,
) -> PolarsResult<ArrayRef>
where
    Agg: ExpandingAggWindow<'a, T>,
    T: NativeType + iter::Sum + NumCast + Mul<Output = T> + AddAssign + SubAssign + Num,
{
    match weights {
        None => expanding_apply_agg_window::<Agg, _>(&values, validity, min_periods),
        Some(weights) => {
            let weights = coerce_weights(weights);
            expanding_apply_weights(&values, min_periods, compute_sum_weights, &weights)
        },
    }
}

pub(super) fn expanding_apply_agg_window<'a, Agg, T>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    min_periods: usize,
) -> PolarsResult<ArrayRef>
where
    Agg: ExpandingAggWindow<'a, T>,
    T: Debug + NativeType + Num,
{
    let len = values.len();
    // Instantiate new window struct
    let mut agg_window = unsafe { Agg::new(values, validity, 0, 1) };
    let out = (1..=len).map(|idx| {
        if idx < min_periods {
            None
        } else {
            // SAFETY:
            // we are in bounds
            unsafe { agg_window.update(0, idx) }
        }
    });
    let arr = PrimitiveArray::from_trusted_len_iter(out);
    Ok(Box::new(arr))
}

pub(super) fn expanding_apply_weights<T, Fa>(
    values: &[T],
    min_periods: usize,
    aggregator: Fa,
    weights: &[T],
) -> PolarsResult<ArrayRef>
where
    T: NativeType,
    Fa: Fn(&[T], &[T]) -> T,
{
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let vals = unsafe { values.get_unchecked(0..idx) };

            aggregator(vals, weights)
        })
        .collect_trusted::<Vec<T>>();

    let validity = get_validity(len, min_periods);
    Ok(Box::new(PrimitiveArray::new(
        ArrowDataType::from(T::PRIMITIVE),
        out.into(),
        validity.map(|b| b.into()),
    )))
}
 */

fn get_validity(n: usize, t: usize) -> Option<MutableBitmap> {
    let mut bits = MutableBitmap::with_capacity(n);
    bits.extend_constant(t, false); // First T bits as false
    bits.extend_constant(n - t, true); // Remaining N-T bits as true
    Some(bits)
}

pub(super) fn calc_expanding_aggregator<'a, Agg, T>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    min_periods: usize,
) -> ArrayRef
where
    Agg: ExpandingAggWindow<'a, T>,
    T: Debug + NativeType + Num,
{
    let len = values.len();
    // Instantiate new window struct
    let mut agg_window = unsafe { Agg::new(values, validity, 0, 1) };
    let out = (1..=len).map(|idx| {
        if idx < min_periods {
            None
        } else {
            // SAFETY:
            // we are in bounds
            unsafe { agg_window.update(0, idx) }
        }
    });
    let arr = PrimitiveArray::from_trusted_len_iter(out);
    Box::new(arr)
}

pub(super) fn calc_expanding_weighted_aggregator<T, Fa>(
    values: &[T],
    min_periods: usize,
    aggregator: Fa,
    weights: &[T],
) -> ArrayRef
where
    T: NativeType,
    Fa: Fn(&[T], &[T]) -> T,
{
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let vals = unsafe { values.get_unchecked(0..idx) };

            aggregator(vals, weights)
        })
        .collect_trusted::<Vec<T>>();

    let validity = get_validity(len, min_periods);
    Box::new(PrimitiveArray::new(
        ArrowDataType::from(T::PRIMITIVE),
        out.into(),
        validity.map(|b| b.into()),
    ))
}