pub mod cagr;
pub mod ewm;
pub mod kurtosis;
pub mod max;
pub mod mean;
pub mod min;
pub mod no_nulls;
pub mod nulls;
pub mod prod;
pub mod quantile;
pub mod skew;
pub mod stdev;
pub mod sum;
pub mod variance;

// Grouped imports
use std::iter;
use std::iter::Sum;
use std::ops::{AddAssign, Div, DivAssign, MulAssign, SubAssign};

use arrow::array::Array;
use num_traits::{Float, Num, NumCast};
use polars::datatypes::{Float32Type, Float64Type};
use polars::prelude::PolarsResult;
use polars_arrow::array::{ArrayRef, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::types::NativeType;
use polars_core::datatypes::DataType::{Float32, Float64};
use polars_core::datatypes::PolarsNumericType;
use polars_core::prelude::ChunkedArray;
use polars_core::series::Series;
use polars_custom_utils::utils::weights::coerce_weights;
use polars_utils::float::IsFloat;
use polars_utils::index::NullCount;

use crate::rolling::RollingAggWindow;
use crate::MyArrayExt;

pub(super) struct SumSquaredWindow<'a, T> {
    slice: &'a [T],
    sum_of_squares: Option<T>,
    last_start: usize,
    last_end: usize,
    validity: Option<&'a Bitmap>,
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

pub trait ExpandingAggWindow<'a, T: NativeType> {
    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T>;
    unsafe fn update_nulls(&mut self, start: usize, end: usize) -> Option<T>;
    unsafe fn update_no_nulls(&mut self, start: usize, end: usize) -> Option<T>;
    unsafe fn new(slice: &'a [T], validity: Option<&'a Bitmap>, start: usize, end: usize) -> Self;
    fn is_valid(&self, min_periods: usize) -> bool;
    fn window_type() -> &'static str;
}

trait WindowType<'a, T: NativeType> {
    type Window: 'a + ExpandingAggWindow<'a, T>;
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

fn calc_expanding_generic<'a, T, W>(
    arr: &'a PrimitiveArray<T>,
    min_periods: usize,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat,
    W: WindowType<'a, T>,
{
    match (arr.has_nulls(), weights) {
        (true, None) => nulls::calc_expanding_aggregator::<W::Window, T>(
            arr.values().as_slice(),
            arr.validity(),
            min_periods,
        ),
        (false, None) => no_nulls::calc_expanding_aggregator::<W::Window, T>(
            arr.values().as_slice(),
            arr.validity(),
            min_periods,
        ),
        (true, Some(weights)) => {
            let weights = coerce_weights(weights);
            nulls::calc_expanding_weighted_aggregator(
                arr.values().as_slice(),
                arr.validity().unwrap(),
                min_periods,
                W::get_weight_computer(),
                &W::prepare_weights(weights),
            )
        },
        (false, Some(weights)) => {
            let weights = coerce_weights(weights);
            no_nulls::calc_expanding_weighted_aggregator(
                arr.values().as_slice(),
                min_periods,
                W::get_weight_computer(),
                &W::prepare_weights(weights),
            )
        },
    }
}

fn apply_expanding_aggregator_chunked<T>(
    ca: &ChunkedArray<T>,
    min_periods: usize,
    weights: Option<Vec<f64>>,
    aggregator_fn: &dyn Fn(&PrimitiveArray<T::Native>, usize, Option<&[f64]>) -> ArrayRef,
) -> PolarsResult<Series>
where
    T: PolarsNumericType,
    <T as PolarsNumericType>::Native: Float,
{
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    let arr = aggregator_fn(&arr, min_periods, weights.as_deref());
    Series::try_from((ca.name().clone(), arr))
}

/*
fn expanding_aggregator<'a, Agg, T, U>(
    ca: &'a ChunkedArray<U>,
    min_periods: usize,
    weights: Option<Vec<f64>>,
) -> PolarsResult<Series>
where
    Agg: ExpandingAggWindow<'a, T>,
    U: PolarsNumericType<Native = T>,
    T: NativeType
        + Sum
        + iter::Product
        + NumCast
        + AddAssign
        + SubAssign
        + DivAssign
        + MulAssign
        + Num
        + PartialOrd,
{
    let arr = ca.downcast_iter().next().unwrap();
    let arr = match ca.null_count() {
        0 => expanding_aggregator_no_nulls::<Agg, _>(
            arr.values().as_slice(),
            arr.validity(),
            min_periods,
            weights.as_deref(),
        )?,
        _ => expanding_aggregator_nulls::<Agg, _>(
            arr.values().as_slice(),
            arr.validity(),
            min_periods,
            weights.as_deref(),
        )?,
    };
    Series::try_from((ca.name().clone(), arr))
}

 */
