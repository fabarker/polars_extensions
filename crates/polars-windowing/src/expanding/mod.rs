pub mod no_nulls;
pub mod nulls;
pub mod mean;
pub mod prod;
pub mod sum;
pub mod variance;
pub mod skew;
pub mod kurtosis;
pub mod stdev;
pub mod max;
pub mod min;
pub mod quantile;
pub mod ewm;
pub mod cagr;

// Grouped imports
use polars::{
    datatypes::{Float32Type, Float64Type},
    prelude::PolarsResult,
};

use polars_arrow::{
    bitmap::{Bitmap, MutableBitmap},
    types::NativeType,
};

use polars_core::{
    datatypes::{
        DataType::{Float32, Float64},
        PolarsNumericType,
    },
    prelude::ChunkedArray,
    series::Series,
};

use polars_utils::float::IsFloat;

use std::{
    iter,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{Num, NumCast};

// Local imports
use crate::expanding::{
    no_nulls::expanding_aggregator_no_nulls,
    nulls::expanding_aggregator_nulls,
};

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


fn expanding_aggregator<'a, Agg, T, U>(
    ca: &'a ChunkedArray<U>,
    min_periods: usize,
    weights: Option<Vec<f64>>) -> PolarsResult<Series>
where
    Agg: ExpandingAggWindow<'a, T>,
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
