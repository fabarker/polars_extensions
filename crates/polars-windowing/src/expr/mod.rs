use std::sync::Arc;

use polars::prelude::{Expr, SpecialEq};
use polars_core::datatypes::DataType::Float64;
use polars_core::error::PolarsResult;
use polars_core::prelude::RollingOptionsFixedWindow;
use polars_core::series::Series;
use pyo3_polars::export::polars_plan::prelude::{
    ApplyOptions, FunctionFlags, FunctionOptions, GetOutput,
};

use crate::rolling::cagr::rolling_cagr_with_opts;

pub trait MyCustomTrait {
    fn rolling_cagr(self, options: RollingOptionsFixedWindow) -> Expr;
    fn map_custom<F>(self, function: F, output_type: GetOutput) -> Expr
    where
        F: Fn(Series) -> PolarsResult<Series> + 'static + Send + Sync;
}

impl MyCustomTrait for Expr {
    fn map_custom<F>(self, function: F, output_type: GetOutput) -> Self
    where
        F: Fn(Series) -> PolarsResult<Series> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| {
            let result = function(std::mem::take(&mut s[0]))?;
            Ok(Some(result)) // Wrap the result in Some
        };

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ElementWise,
                fmt_str: "map",
                flags: FunctionFlags::default() | FunctionFlags::OPTIONAL_RE_ENTRANT,
                ..Default::default()
            },
        }
    }

    fn rolling_cagr(self, options: RollingOptionsFixedWindow) -> Self {
        self.map_custom(
            move |s| rolling_cagr_with_opts(&s, options.clone()),
            GetOutput::from_type(Float64),
        )
    }
}
