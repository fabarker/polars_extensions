use std::ops::{Add, Sub};

use num_traits::Num;
use polars_arrow::array::{ArrayRef, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::legacy::utils::CustomIterTools;
use polars_arrow::types::NativeType;
use crate::BitmapExt;
use super::*;
use crate::rolling::{End, Idx, Len, Start, WindowSize};

/*
pub fn rolling_aggregator_nulls<'a, Agg, T>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> PolarsResult<ArrayRef>
where
    Agg: RollingAggWindow<'a, T>,
    T: NativeType + Num + PartialOrd + Add<Output = T> + Sub<Output = T>,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }

    if center {
        rolling_apply_agg_window::<Agg, _, _>(
            values,
            validity,
            window_size,
            min_periods,
            det_offsets_center,
        )
    } else {
        rolling_apply_agg_window::<Agg, _, _>(
            values,
            validity,
            window_size,
            min_periods,
            det_offsets,
        )
    }
}

 */

// Use an aggregation window that maintains the state
/*
pub(super) fn rolling_apply_agg_window<'a, Agg, T, Fo>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
) -> PolarsResult<ArrayRef>
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End) + Copy,
    Agg: RollingAggWindow<'a, T>,
    T: Num + NativeType,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    // SAFETY; we are in bounds
    let mut agg_window = unsafe { Agg::new(values, validity, start, end) };

    let mut validity = create_validity(min_periods, len, window_size, det_offsets_fn)
        .unwrap_or_else(|| {
            let mut validity = MutableBitmap::with_capacity(len);
            validity.extend_constant(len, true);
            validity
        });

    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            // SAFETY:
            // we are in bounds
            let agg = unsafe { agg_window.update(start, end) };
            match agg {
                Some(val) => {
                    if agg_window.is_valid(min_periods) {
                        Some(val)
                    } else {
                        // SAFETY: we are in bounds
                        unsafe { validity.set_unchecked(idx, false) };
                        None
                    }
                },
                None => {
                    // SAFETY: we are in bounds
                    unsafe { validity.set_unchecked(idx, false) };
                    None
                },
            }
        })
        .collect_trusted::<Vec<Option<T>>>();

    let arr = PrimitiveArray::from(out);
    Ok(Box::new(arr))
}

 */



pub(super) fn calc_rolling_aggregator<'a, Agg, T, Fo>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End) + Copy,
    Agg: RollingAggWindow<'a, T>,
    T: IsFloat + NativeType,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    // SAFETY; we are in bounds
    let mut agg_window = unsafe { Agg::new(values, validity, start, end) };

    let mut validity = create_validity(min_periods, len, window_size, det_offsets_fn)
        .unwrap_or_else(|| {
            let mut validity = MutableBitmap::with_capacity(len);
            validity.extend_constant(len, true);
            validity
        });

    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            // SAFETY:
            // we are in bounds
            let agg = unsafe { agg_window.update(start, end) };
            match agg {
                Some(val) => {
                    if agg_window.is_valid(min_periods) {
                        val
                    } else {
                        // SAFETY: we are in bounds
                        unsafe { validity.set_unchecked(idx, false) };
                        T::default()
                    }
                },
                None => {
                    // SAFETY: we are in bounds
                    unsafe { validity.set_unchecked(idx, false) };
                    T::default()
                },
            }
        })
        .collect_trusted::<Vec<_>>();

    Box::new(PrimitiveArray::new(
        T::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    ))
}


pub(super) fn calc_rolling_weighted_aggregator<T, Fo, Fa>(
    values: &[T],
    validity: &Bitmap,
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

            // If start is less than min_periods, return None
            if start < min_periods {
                None
            } else {
                // Check for nulls and perform aggregation if none
                (!validity.has_nulls_in_range(start, end))
                    .then(|| unsafe { aggregator(values.get_unchecked(start..end), weights) })
            }
        })
        .collect_trusted::<PrimitiveArray<_>>();
    Box::new(out)
}
