use num_traits::Num;
use polars_arrow::array::{ArrayRef, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::legacy::utils::CustomIterTools;
use polars_arrow::types::NativeType;

use super::*;
use crate::BitmapExt;

/*
pub fn expanding_aggregator_nulls<'a, Agg, T>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    min_periods: usize,
    weights: Option<&[f64]>,
) -> PolarsResult<ArrayRef>
where
    Agg: ExpandingAggWindow<'a, T>,
    T: NativeType + Num + PartialOrd + Add<Output = T> + Sub<Output = T>,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }

    expanding_apply_agg_window::<Agg, _>(values, validity, min_periods)
}

// Use an aggregation window that maintains the state
pub(super) fn expanding_apply_agg_window<'a, Agg, T>(
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    min_periods: usize,
) -> PolarsResult<ArrayRef>
where
    Agg: ExpandingAggWindow<'a, T>,
    T: Num + NativeType,
{
    let len = values.len();
    let mut agg_window = unsafe { Agg::new(values, validity, 0, 0) };

    let mut validity = MutableBitmap::with_capacity(len);
    validity.extend_constant(len, true);

    let out = (0..=len)
        .map(|idx| {
            // SAFETY:
            // we are in bounds
            let agg = unsafe { agg_window.update(0, idx) };
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

pub(super) fn calc_expanding_aggregator<'a, Agg, T>(
    values: &'a [T],
    validmap: Option<&'a Bitmap>,
    min_periods: usize,
) -> ArrayRef
where
    Agg: ExpandingAggWindow<'a, T>,
    T: Num + NativeType,
{
    let len = values.len();
    let mut agg_window = unsafe { Agg::new(values, validmap, 0, 0) };

    let mut validity = MutableBitmap::with_capacity(len);
    validity.extend_constant(len, true);
    if let Some(validmap) = validmap {
        for i in 0..len {
            if !validmap.get_bit(i) {
                unsafe { validity.set_unchecked(i, false) };
            }
        }
    }

    let out = (0..len)
        .map(|idx| {
            let agg = unsafe { agg_window.update(0, idx) };
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

pub(super) fn calc_expanding_weighted_aggregator<T, Fa>(
    values: &[T],
    validity: &Bitmap,
    min_periods: usize,
    aggregator: Fa,
    weights: &[T],
) -> ArrayRef
where
    T: NativeType,
    Fa: Fn(&[T], &[T]) -> T,
{
    let len = values.len();
    let out = (0..=len)
        .map(|idx| {
            if idx < min_periods {
                None
            } else {
                (!validity.has_nulls_in_range(0, idx))
                    .then(|| unsafe { aggregator(values.get_unchecked(0..idx), weights) })
            }
        })
        .collect_trusted::<PrimitiveArray<_>>();
    Box::new(out)
}
