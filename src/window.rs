#![allow(clippy::unused_unit)]
use polars::prelude::*;
use polars_custom_utils::utils::weights::ExponentialDecayType;
pub use polars_windowing::{
    Expanding, ExpandingKwargs, ExponentiallyWeighted, Rolling, RollingKwargs, WindowParams,
};
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;

fn output_mapper(input_fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).map_to_float_dtype()
}

#[derive(Debug)]
pub enum WindowType<'a> {
    Expanding(Expanding<'a>),
    Rolling(Rolling<'a>),
}

impl<'a> WindowType<'a> {
    pub fn sum(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.sum(),
            WindowType::Expanding(e) => e.sum(),
        }
    }

    pub fn mean(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.mean(),
            WindowType::Expanding(e) => e.mean(),
        }
    }

    pub fn std(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.std(),
            WindowType::Expanding(e) => e.std(),
        }
    }

    pub fn var(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.var(),
            WindowType::Expanding(e) => e.var(),
        }
    }

    pub fn skew(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.skew(),
            WindowType::Expanding(e) => e.skew(),
        }
    }

    pub fn kurt(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.kurt(),
            WindowType::Expanding(e) => e.kurt(),
        }
    }

    pub fn cgr(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.cagr("simple"),
            WindowType::Expanding(e) => e.cagr("simple"),
        }
    }

    pub fn prod(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.prod(),
            _ => Err(PolarsError::ComputeError(
                "Method not implemented for this window type".into(),
            )),
        }
    }

    pub fn min(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.min(),
            WindowType::Expanding(e) => e.min(),
        }
    }

    pub fn max(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.max(),
            WindowType::Expanding(e) => e.max(),
        }
    }

    pub fn median(&self) -> PolarsResult<Series> {
        match self {
            WindowType::Rolling(r) => r.median(),
            WindowType::Expanding(e) => e.median(),
        }
    }

    pub fn ewm(
        self,
        decay_type: ExponentialDecayType,
        bias: bool,
    ) -> PolarsResult<ExponentiallyWeighted<'a>> {
        match self {
            WindowType::Rolling(r) => Ok(r.ewm(decay_type, bias)),
            WindowType::Expanding(e) => Ok(e.ewm(decay_type, bias)),
        }
    }
}

fn initialize_window(inputs: &Series, kwargs: WindowParams) -> PolarsResult<WindowType> {
    match kwargs.get_win_type() {
        "expanding" => Ok(WindowType::Expanding(Expanding::from_kwargs(
            &inputs,
            ExpandingKwargs::from_kwargs(&kwargs),
        ))),
        "rolling" => Ok(WindowType::Rolling(Rolling::from_kwargs(
            &inputs,
            RollingKwargs::from_kwargs(&kwargs),
        ))),
        win_type => Err(PolarsError::ComputeError(
            format!("Unknown window type: {}", win_type).into(),
        )),
    }
}

#[polars_expr(output_type_func=output_mapper)]
pub fn windowed_stats(inputs: &[Series], kwargs: WindowParams) -> PolarsResult<Series> {

    dbg!(&inputs);
    let win = initialize_window(&inputs[0], kwargs)?;
    let result = match inputs[1].str_value(0)?.as_ref() {
        "mean" => win.mean(),
        "std" => win.std(),
        "var" => win.var(),
        "prod" => win.prod(),
        "cgr" => win.cgr(),
        "min" => win.min(),
        "max" => win.max(),
        "median" => win.median(),
        other => Err(PolarsError::ComputeError(
            format!("Unknown function: {}", other).into(),
        )),
    };
    result
}

fn initialize_window_ewm(
    inputs: &Series,
    kwargs: WindowParams,
) -> PolarsResult<ExponentiallyWeighted> {
    match kwargs.get_win_type() {
        "expanding" => Ok(
            Expanding::from_kwargs(&inputs, ExpandingKwargs::from_kwargs(&kwargs))
                .ewm(kwargs.decay.unwrap(), kwargs.bias.unwrap_or(false)),
        ),
        "rolling" => Ok(
            Rolling::from_kwargs(&inputs, RollingKwargs::from_kwargs(&kwargs))
                .ewm(kwargs.decay.unwrap(), kwargs.bias.unwrap_or(false)),
        ),
        win_type => Err(PolarsError::ComputeError(
            format!("Unknown window type: {}", win_type).into(),
        )),
    }
}

#[polars_expr(output_type_func=output_mapper)]
pub fn exponentially_weighted(inputs: &[Series], kwargs: WindowParams) -> PolarsResult<Series> {
    dbg!(&inputs);
    let ewm = initialize_window(&inputs[0], kwargs.clone())?
        .ewm(kwargs.decay.unwrap(), kwargs.bias.unwrap_or(false))?;
    let result = match inputs[1].str_value(0)?.as_ref() {
        "mean" => ewm.mean(),
        "std" => ewm.std(),
        "var" => ewm.var(),
        "prod" => ewm.prod(),
        "cgr" => ewm.cagr("simple"),
        other => Err(PolarsError::ComputeError(
            format!("Unknown function: {}", other).into(),
        )),
    };
    result
}
