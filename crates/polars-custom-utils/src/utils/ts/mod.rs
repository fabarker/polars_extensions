use polars_core::prelude::*;
use serde::Deserialize;
use std::ops::{Add};

#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(from = "&str")]
pub enum ReturnsType {
    Simple,
    Log,
}

impl From<&str> for ReturnsType {
    fn from(string: &str) -> Self {
        match string.to_lowercase().as_str() {
            "simple" | "linear" => ReturnsType::Simple,
            "log" => ReturnsType::Log,
            _ => panic!("Invalid returns type: {}", string), // Use panic sparingly
        }
    }
}

impl ReturnsType {
    // Method to convert any type to half_life
    pub fn linear_to_log(linear_returns: &Series) -> PolarsResult<Series> {
        let float_series = linear_returns.cast(&DataType::Float64)?;
        Ok(float_series.add(1.0).f64()?.apply_values(|v| v.ln()).into_series())
    }
    pub fn log_to_linear(log_returns: &Series) -> PolarsResult<Series> {
        let float_series = log_returns.cast(&DataType::Float64)?;
        Ok(float_series.f64()?.apply_values(|v| v.exp() - 1.0).into_series())
    }

    pub fn to_linear(&self, returns: &Series) -> PolarsResult<Series> {
        match self {
            ReturnsType::Log => Self::log_to_linear(returns),
            ReturnsType::Simple => Ok(returns.clone()),
        }
    }

    pub fn to_log(&self, returns: &Series) -> PolarsResult<Series> {
        match self {
            ReturnsType::Simple => Self::linear_to_log(returns),
            ReturnsType::Log => Ok(returns.clone()),
        }
    }

    pub fn from_to_type(returns: &Series,
                       from_type: ReturnsType,
                       to_type: ReturnsType
    ) -> PolarsResult<Series> {
        match (from_type, to_type) {
            (ReturnsType::Simple, ReturnsType::Log) => Self::linear_to_log(returns),
            (ReturnsType::Log, ReturnsType::Simple) => Self::log_to_linear(returns),
            _ => Ok(returns.clone())
        }
    }
}
