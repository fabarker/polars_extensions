use std::ops::{Add, Sub};
use num_traits::Num;
use polars_arrow::array::{ArrayRef, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::types::NativeType;
use polars_arrow::legacy::utils::CustomIterTools;
use super::*;

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

    expanding_apply_agg_window::<Agg, _>(
            values,
            validity,
            min_periods,
        )
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
