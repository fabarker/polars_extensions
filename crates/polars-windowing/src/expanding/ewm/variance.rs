use polars_arrow::legacy::kernels::ewm::ewm_var as kernel_ewm_var;
pub use polars_arrow::legacy::kernels::ewm::EWMOptions;
use polars_core::prelude::*;

fn check_alpha(alpha: f64) -> PolarsResult<()> {
    polars_ensure!((0.0..=1.0).contains(&alpha), ComputeError: "alpha must be in [0; 1]");
    Ok(())
}

pub fn ewm_var(
    s: &Series,
    alpha: f64,
    adjust: bool,
    min_periods: usize,
    ignore_nulls: bool,
    bias: bool,
) -> PolarsResult<Series> {
    check_alpha(alpha)?;
    match s.dtype() {
        DataType::Float32 => {
            let xs = s.f32().unwrap();
            let result = kernel_ewm_var(xs, alpha as f32, adjust, bias, min_periods, ignore_nulls);
            Series::try_from((s.name().clone(), Box::new(result) as ArrayRef))
        },
        DataType::Float64 => {
            let xs = s.f64().unwrap();
            let result = kernel_ewm_var(xs, alpha, adjust, bias, min_periods, ignore_nulls);
            Series::try_from((s.name().clone(), Box::new(result) as ArrayRef))
        },
        _ => ewm_var(
            &s.cast(&DataType::Float64)?,
            alpha,
            adjust,
            min_periods,
            ignore_nulls,
            bias,
        ),
    }
}
