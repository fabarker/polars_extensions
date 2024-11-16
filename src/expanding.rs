#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
pub use polars_windowing::*;
use pyo3::prelude::*;

fn output_mapper(input_fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).map_to_float_dtype()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_mean(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).mean()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_sum(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).sum()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_std(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).std()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_skew(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).skew()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_kurt(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).kurt()
}


#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_cagr(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).cagr()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_var(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).var()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_median(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).median()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_min(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).min()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn expanding_max(
    inputs: &[Series],
    kwargs: ExpandingKwargs) -> PolarsResult<Series> {
    Expanding::from_kwargs(&inputs[0], kwargs).max()
}


