use std::fmt::Debug;
use std::iter;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_core::series::Series;
use std::ops::{AddAssign, SubAssign, DivAssign, MulAssign};
use num_traits::{NumCast, Num};
use polars_utils::float::IsFloat;
use polars::prelude::{PolarsResult};
use polars_arrow::types::NativeType;
use polars_core::datatypes::PolarsNumericType;
use polars_core::datatypes::DataType::{Float64, Float32};
use polars_core::prelude::ChunkedArray;
use polars::datatypes::{Float32Type, Float64Type};
use crate::rolling::no_nulls::rolling_aggregator_no_nulls;
use crate::rolling::nulls::rolling_aggregator_nulls;

pub mod no_nulls;
pub mod nulls;
pub mod sum;
pub mod prod;
pub mod mean;
pub mod variance;
pub mod stdev;
pub mod skew;
pub mod kurtosis;
pub mod quantile;
pub mod cagr;
type Start = usize;
type End = usize;
type Idx = usize;
type WindowSize = usize;
type Len = usize;

#[derive(Clone, Debug)]
pub struct RollingOptions {
    pub window_size: usize,
    pub min_periods: usize,
    pub centred: bool,
    pub weights: Option<Vec<f64>>,
}


pub(super) struct SumSquaredWindow<'a, T> {
    slice: &'a [T],
    sum_of_squares: Option<T>,
    last_start: usize,
    last_end: usize,
    validity: Option<&'a Bitmap>,
    // if we don't recompute every 'n' iterations
    // we get a accumulated error/drift
    last_recompute: u8,
    pub(super) null_count: usize,
}

pub struct VarWindow<'a, T> {
    mean: MeanWindow<'a, T>,
    sum_of_squares: SumSquaredWindow<'a, T>,
    ddof: u8,
}

pub struct SumWindow<'a, T> {
    slice: &'a [T],
    validity: Option<&'a Bitmap>,
    sum: Option<T>,
    last_start: usize,
    last_end: usize,
    pub(super) null_count: usize,
}

pub struct MeanWindow<'a, T> {
    sum: SumWindow<'a, T>,
}

pub struct ProdWindow<'a, T> {
    slice: &'a [T],
    validity: Option<&'a Bitmap>,
    prod: Option<T>,
    last_start: usize,
    last_end: usize,
    pub(super) null_count: usize,
}


pub trait RollingAggWindow<'a, T: NativeType> {

    unsafe fn update(
        &mut self,
        start: usize,
        end: usize) -> Option<T>;

    unsafe fn update_nulls(
        &mut self,
        start: usize,
        end: usize) -> Option<T>;

    unsafe fn update_no_nulls(
        &mut self,
        start: usize,
        end: usize) -> Option<T>;

    unsafe fn new(
        slice: &'a [T],
        validity: Option<&'a Bitmap>,
        start: usize,
        end: usize,
    ) -> Self;

    fn is_valid(
        &self,
        min_periods: usize) -> bool;

    fn window_type() -> &'static str;

}

fn det_offsets(i: Idx, window_size: WindowSize, _len: Len) -> (usize, usize) {
    (i.saturating_sub(window_size - 1), i + 1)
}

fn det_offsets_center(i: Idx, window_size: WindowSize, len: Len) -> (usize, usize) {
    let right_window = (window_size + 1) / 2;
    (
        i.saturating_sub(window_size - right_window),
        std::cmp::min(len, i + right_window),
    )
}

fn create_validity<Fo>(
    min_periods: usize,
    len: usize,
    window_size: usize,
    det_offsets_fn: Fo,
) -> Option<MutableBitmap>
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
{
    if min_periods > 1 {
        let mut validity = MutableBitmap::with_capacity(len);
        validity.extend_constant(len, true);

        // Set the null values at the boundaries

        // Head.
        for i in 0..len {
            let (start, end) = det_offsets_fn(i, window_size, len);
            if (end - start) < min_periods {
                validity.set(i, false)
            } else {
                break;
            }
        }
        // Tail.
        for i in (0..len).rev() {
            let (start, end) = det_offsets_fn(i, window_size, len);
            if (end - start) < min_periods {
                validity.set(i, false)
            } else {
                break;
            }
        }

        Some(validity)
    } else {
        None
    }
}


fn rolling_aggregator<'a, Agg, T, U>(
    ca: &'a ChunkedArray<U>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>) -> PolarsResult<Series>
where
    Agg: RollingAggWindow<'a, T>,
    U: PolarsNumericType<Native = T>,
    T: NativeType +
    iter::Sum +
    iter::Product +
    NumCast +
    AddAssign +
    SubAssign +
    DivAssign +
    MulAssign +
    Num +
    PartialOrd,
{

    //let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    let arr = match ca.null_count() {
        0 => rolling_aggregator_no_nulls::<Agg, _>(
            arr.values().as_slice(),
            arr.validity(),
            window_size,
            min_periods,
            center,
            weights.as_deref(),
        )?,
        _ => rolling_aggregator_nulls::<Agg, _>(
            arr.values().as_slice(),
            arr.validity(),
            window_size,
            min_periods,
            center,
            weights.as_deref(),
        )?,
    };
    Series::try_from((ca.name().clone(), arr))

}