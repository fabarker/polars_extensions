use ndarray::{s, Array1};
use pyo3::{pyfunction, PyResult};
use serde::Deserialize;
use thiserror::Error;

use crate::Utils;

#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(from = "(String, f32)")]
pub enum ExponentialDecayType {
    HalfLife(f32),
    Com(f32),   // Center of mass
    Alpha(f32), // Smoothing factor
    Span(f32),  // Span size
}

impl From<(String, f32)> for ExponentialDecayType {
    fn from(tuple: (String, f32)) -> Self {
        match tuple.0.as_str() {
            "halflife" => ExponentialDecayType::HalfLife(tuple.1),
            "com" => ExponentialDecayType::Com(tuple.1),
            "alpha" => ExponentialDecayType::Alpha(tuple.1),
            "span" => ExponentialDecayType::Span(tuple.1),
            _ => panic!("Invalid decay type"), // or handle error differently
        }
    }
}

impl ExponentialDecayType {
    // Method to convert any type to half_life
    pub fn get_alpha(&self) -> Result<f32, ExpWeightsError> {
        match self {
            ExponentialDecayType::Alpha(alpha) => Ok(*alpha),
            ExponentialDecayType::HalfLife(hl) => Ok(1.0 - f32::exp(-f32::ln(2.0) / *hl)),
            ExponentialDecayType::Span(s) => Ok(2.0 / (1.0 + *s)),
            ExponentialDecayType::Com(com) => Ok(1.0 / (1.0 + *com)),
        }
    }
    pub fn get_half_life(&self) -> Result<f32, ExpWeightsError> {
        match self {
            ExponentialDecayType::HalfLife(hl) => Ok(*hl),

            ExponentialDecayType::Com(com) => Ok(f32::ln(2.0) / -f32::ln(*com / (1.0 + *com))),

            ExponentialDecayType::Alpha(alpha) => Ok(-f32::ln(2.0) / f32::ln(1.0 - *alpha)),

            ExponentialDecayType::Span(span) => {
                Ok(f32::ln(2.0) / (f32::ln(*span + 1.0) - f32::ln(*span - 1.0)))
            },
        }
    }

    // Constructor methods
    pub fn from_half_life(hl: f32) -> Self {
        Self::HalfLife(hl)
    }

    pub fn from_com(com: f32) -> Self {
        Self::Com(com)
    }

    pub fn from_alpha(alpha: f32) -> Self {
        Self::Alpha(alpha)
    }

    pub fn from_span(span: f32) -> Self {
        Self::Span(span)
    }
}

#[derive(Error, Debug)]
pub enum ExpWeightsError {
    #[error("window must be a strictly positive integer, got {0}")]
    InvalidWindow(i32),
    #[error("half_life must be a strictly positive integer, got {0}")]
    InvalidHalfLife(f32),
}

/// Generate exponentially decaying weights over `window` trailing values,
/// decaying by half each `half_life` index.
///
/// # Arguments
///
/// * `window` - Integer number of points in the trailing lookback period
/// * `half_life` - Integer decay rate (defaults to 126)
///
/// # Returns
///
/// * `Result<Array1<f64>, ExpWeightsError>` - A 1-dimensional array of weights
///
/// # Examples
///
/// ```
/// use polars_custom_utils::utils::weights::exp_weights;
///
/// let weights = exp_weights(5, Some(2.0)).unwrap();
/// assert_eq!(weights.len(), 5);
/// ```

pub fn exp_weights(window: i32, half_life: Option<f32>) -> Result<Vec<f64>, ExpWeightsError> {
    // Validate window
    if window <= 0 {
        return Err(ExpWeightsError::InvalidWindow(window));
    }

    // Validate half_life
    let half_life = half_life.unwrap_or(126.0);
    if half_life <= 0.0 {
        return Err(ExpWeightsError::InvalidHalfLife(half_life));
    }

    // Calculate decay factor
    let decay = std::f64::consts::LN_2 / half_life as f64;

    // Create array of indices and calculate weights
    let weights: Array1<f64> =
        Array1::linspace(0., (window - 1) as f64, window as usize).map(|x| (-decay * x).exp());

    // Reverse the array to match Python's [::-1] behavior
    Ok(weights.slice(s![..;-1]).to_owned().to_vec())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_exp_weights_basic() {
        let weights = exp_weights(5, Some(2)).unwrap();
        assert_eq!(weights.len(), 5);

        // First weight should be closest to 1.0
        assert_relative_eq!(*weights.last().unwrap(), 1.0, epsilon = 1e-10);

        // Each subsequent weight should be smaller
        for i in 1..weights.len() {
            assert!(weights[i] > weights[i - 1]);
        }
    }

    #[test]
    fn test_exp_weights_default_half_life() {
        let weights = exp_weights(5, None).unwrap();
        assert_eq!(weights.len(), 5);
    }

    #[test]
    fn test_exp_weights_invalid_window() {
        assert!(matches!(
            exp_weights(0, Some(2)),
            Err(ExpWeightsError::InvalidWindow(0))
        ));

        assert!(matches!(
            exp_weights(-1, Some(2)),
            Err(ExpWeightsError::InvalidWindow(-1))
        ));
    }

    #[test]
    fn test_exp_weights_invalid_half_life() {
        assert!(matches!(
            exp_weights(5, Some(0)),
            Err(ExpWeightsError::InvalidHalfLife(0))
        ));

        assert!(matches!(
            exp_weights(5, Some(-1)),
            Err(ExpWeightsError::InvalidHalfLife(-1))
        ));
    }
}
