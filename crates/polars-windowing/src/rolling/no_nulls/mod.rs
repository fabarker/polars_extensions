use std::fmt::Debug;
use std::iter;
use std::ops::{Add, AddAssign, Mul, Sub};

use num_traits::{Num, NumCast, One, Zero};
use polars::prelude::PolarsResult;
use polars_arrow::array::{ArrayRef, PrimitiveArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::legacy::utils::CustomIterTools;
use polars_arrow::types::NativeType;

use super::*;

pub fn rolling_aggregator_no_nulls<'a, Agg, T>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> PolarsResult<ArrayRef>
where
    Agg: RollingAggWindow<'a, T>,
    T: NativeType
        + iter::Sum
        + NumCast
        + Mul<Output = T>
        + AddAssign
        + SubAssign
        + Num
        + iter::Product
        + One,
{
    let aggregator = {
        match Agg::window_type() {
            "prod" => compute_prod_weights,
            _ => compute_sum_weights,
        }
    };

    match (center, weights) {
        (true, None) => rolling_apply_agg_window::<Agg, _, _>(
            &values,
            validity,
            window_size,
            min_periods,
            det_offsets_center,
        ),
        (false, None) => rolling_apply_agg_window::<Agg, _, _>(
            &values,
            validity,
            window_size,
            min_periods,
            det_offsets,
        ),
        (true, Some(weights)) => {
            let weights = coerce_weights(weights);
            rolling_apply_weights(
                &values,
                window_size,
                min_periods,
                det_offsets_center,
                aggregator,
                &weights,
            )
        },
        (false, Some(weights)) => {
            let weights = coerce_weights(weights);
            rolling_apply_weights(
                &values,
                window_size,
                min_periods,
                det_offsets,
                aggregator,
                &weights,
            )
        },
    }
}

pub(super) fn rolling_apply_agg_window<'a, Agg, T, Fo>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
) -> PolarsResult<ArrayRef>
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Agg: RollingAggWindow<'a, T>,
    T: Debug + NativeType + Num,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);

    // Instantiate new window struct
    let mut agg_window = unsafe { Agg::new(values, validity, start, end) };

    if let Some(validity) = create_validity(min_periods, len, window_size, &det_offsets_fn) {
        if validity.iter().all(|x| !x) {
            return Ok(Box::new(PrimitiveArray::<T>::new_null(
                T::PRIMITIVE.into(),
                len,
            )));
        }
    }

    let out = (0..len).map(|idx| {
        let (start, end) = det_offsets_fn(idx, window_size, len);
        if end - start < min_periods {
            None
        } else {
            // SAFETY:
            // we are in bounds
            unsafe { agg_window.update(start, end) }
        }
    });
    let arr = PrimitiveArray::from_trusted_len_iter(out);
    Ok(Box::new(arr))
}

pub(super) fn rolling_apply_weights<T, Fo, Fa>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[T],
) -> PolarsResult<ArrayRef>
where
    T: NativeType,
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[T], &[T]) -> T,
{
    assert_eq!(weights.len(), window_size);
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };

            aggregator(vals, weights)
        })
        .collect_trusted::<Vec<T>>();

    let validity = create_validity(min_periods, len, window_size, det_offsets_fn);
    Ok(Box::new(PrimitiveArray::new(
        ArrowDataType::from(T::PRIMITIVE),
        out.into(),
        validity.map(|b| b.into()),
    )))
}

pub(crate) fn compute_prod_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: iter::Product<T> + Copy + Mul<Output = T> + One + Sub<Output = T> + Add<Output = T>,
{
    values
        .iter()
        .zip(weights)
        .map(|(v, w)| (*v - T::one()) * *w + T::one())
        .product()
}

pub(crate) fn compute_sum_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: iter::Sum<T> + Copy + Mul<Output = T>,
{
    values.iter().zip(weights).map(|(v, w)| *v * *w).sum()
}

pub(crate) fn compute_sum_weights_normalized<T>(values: &[T], weights: &[T]) -> T
where
    T: iter::Sum<T> + Copy + Mul<Output = T> + Zero + One + std::ops::Div<Output = T>,
{
    assert!(!weights.is_empty(), "Weights array cannot be empty");
    assert_eq!(
        values.len(),
        weights.len(),
        "Values and weights must have the same length"
    );

    // Calculate sum of weights
    let sum: T = weights.iter().copied().fold(T::zero(), |acc, x| acc + x);
    assert!(!sum.is_zero(), "Sum of weights cannot be zero");

    // Calculate inverse of sum for normalization
    let inv_sum = T::one() / sum;

    // Multiply each value by its normalized weight and sum
    values
        .iter()
        .zip(weights.iter())
        .map(|(v, w)| *v * (*w * inv_sum))
        .sum()
}

pub(super) fn coerce_weights<T: NumCast>(weights: &[f64]) -> Vec<T>
where
{
    weights
        .iter()
        .map(|v| NumCast::from(*v).unwrap())
        .collect::<Vec<_>>()
}

pub(super) fn calc_rolling_aggregator<'a, Agg, T, Fo>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Agg: RollingAggWindow<'a, T>,
    T: Debug + NativeType + Num,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    let mut agg_window = unsafe { Agg::new(values, validity, start, end) };
    if let Some(validity) = create_validity(min_periods, len, window_size, &det_offsets_fn) {
        if validity.iter().all(|x| !x) {
            return Box::new(PrimitiveArray::<T>::new_null(
                T::PRIMITIVE.into(),
                len,
            ));
        }
    }

    let out = (0..len).map(|idx| {
        let (start, end) = det_offsets_fn(idx, window_size, len);
        if end - start < min_periods {
            None
        } else {
            // SAFETY:
            // we are in bounds
            unsafe { agg_window.update(start, end) }
        }
    });
    let arr = PrimitiveArray::from_trusted_len_iter(out);
    Box::new(arr)
}

pub(super) fn calc_rolling_weighted_aggregator<T, Fo, Fa>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[T],
) -> ArrayRef
where
    T: NativeType,
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[T], &[T]) -> T,
{
    assert_eq!(weights.len(), window_size);
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };

            aggregator(vals, weights)
        })
        .collect_trusted::<Vec<T>>();

    let validity = create_validity(min_periods, len, window_size, det_offsets_fn).unwrap();
    Box::new(PrimitiveArray::new(
        T::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    ))
}


