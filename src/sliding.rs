#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
pub use polars_windowing::*;
use pyo3::prelude::*;

fn output_mapper(input_fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).map_to_float_dtype()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_mean(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).mean()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_sum(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).sum()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_std(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).std()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_skew(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).skew()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_kurt(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).kurt()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_prod(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).prod()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_cagr(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).cagr()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_var(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).var()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_median(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).median()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_min(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).min()
}

#[polars_expr(output_type_func=output_mapper)]
pub fn rolling_max(
    inputs: &[Series],
    kwargs: RollingKwargs) -> PolarsResult<Series> {
    Rolling::from_kwargs(&inputs[0], kwargs).max()
}


