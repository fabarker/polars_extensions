use std::fmt::Debug;
use std::iter;
use std::iter::Sum;
use std::ops::{AddAssign, Div, DivAssign, MulAssign, SubAssign};

use num_traits::{Float, Num, NumCast};
use polars::datatypes::{Float32Type, Float64Type};
use polars::prelude::PolarsResult;
use polars_arrow::array::{Array, ArrayRef, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::types::NativeType;
use polars_core::datatypes::DataType::{Float32, Float64};
use polars_core::datatypes::PolarsNumericType;
use polars_core::prelude::ChunkedArray;
use polars_core::series::Series;
use polars_custom_utils::utils::weights::coerce_weights;
use polars_utils::float::IsFloat;
use polars_utils::index::NullCount;

use crate::MyArrayExt;

pub mod cagr;
pub mod kurtosis;
pub mod mean;
pub mod no_nulls;
pub mod nulls;
pub mod prod;
pub mod quantile;
pub mod skew;
pub mod stdev;
pub mod sum;
pub mod variance;
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
    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T>;

    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T>;

    unsafe fn update_no_nulls(&mut self, start: usize, end: usize) -> Option<T>;

    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize) -> Self;

    fn is_valid(&self, min_periods: usize) -> bool;

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

/*
fn rolling_aggregator<'a, Agg, T, U>(
    ca: &'a ChunkedArray<U>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series>
where
    Agg: RollingAggWindow<'a, T>,
    U: PolarsNumericType<Native = T>,
    T: NativeType
    + iter::Sum
    + iter::Product
    + NumCast
    + AddAssign
    + SubAssign
    + DivAssign
    + MulAssign
    + Num
    + PartialOrd,
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

 */

fn apply_rolling_aggregator_chunked<T>(
    ca: &ChunkedArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
    aggregator_fn: &dyn Fn(
        &PrimitiveArray<T::Native>,
        usize,
        usize,
        bool,
        Option<&[f64]>,
    ) -> ArrayRef,
) -> PolarsResult<Series>
where
    T: PolarsNumericType,
    <T as PolarsNumericType>::Native: Float,
{
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    let arr = aggregator_fn(&arr, window_size, min_periods, center, weights.as_deref());
    Series::try_from((ca.name().clone(), arr))
}

trait WindowType<'a, T: NativeType> {
    type Window: 'a + RollingAggWindow<'a, T>;
    fn get_weight_computer() -> fn(&[T], &[T]) -> T;

    fn prepare_weights(weights: Vec<T>) -> Vec<T>;
    fn normalize_weights(mut weights: Vec<T>) -> Vec<T>
    where
        T: Float + Sum + Div<Output = T> + Copy, // Ensure T supports division and floating-point operations
    {
        let wsum = weights.iter().fold(T::zero(), |acc, x| acc + *x);
        weights.iter_mut().for_each(|w| *w = *w / wsum);
        weights
    }
}

type OffsetFn = fn(usize, usize, usize) -> (usize, usize);

fn calc_rolling_generic<'a, T, W>(
    arr: &'a PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + Float + iter::Sum<T> + SubAssign + AddAssign + IsFloat,
    W: WindowType<'a, T>,
{
    let offsets: OffsetFn = if center {
        det_offsets_center
    } else {
        det_offsets
    };

    match (arr.has_nulls(), weights) {
        (true, None) => nulls::calc_rolling_aggregator::<W::Window, T, OffsetFn>(
            arr.values().as_slice(),
            arr.validity(),
            window_size,
            min_periods,
            offsets,
        ),
        (false, None) => no_nulls::calc_rolling_aggregator::<W::Window, T, OffsetFn>(
            arr.values().as_slice(),
            arr.validity(),
            window_size,
            min_periods,
            offsets,
        ),
        (true, Some(weights)) => {
            let weights = coerce_weights(weights);
            nulls::calc_rolling_weighted_aggregator(
                arr.values().as_slice(),
                arr.validity().unwrap(),
                window_size,
                min_periods,
                offsets,
                W::get_weight_computer(),
                &W::prepare_weights(weights),
            )
        },
        (false, Some(weights)) => {
            let weights = coerce_weights(weights);
            no_nulls::calc_rolling_weighted_aggregator(
                arr.values().as_slice(),
                window_size,
                min_periods,
                offsets,
                W::get_weight_computer(),
                &W::prepare_weights(weights),
            )
        },
    }
}
