use std::iter;
use std::ops::{Add, Div, DivAssign, Mul, Sub};
use ndarray::{s, Array1};
use num_traits::{NumCast, Zero};
use serde::Deserialize;
use thiserror::Error;

pub fn coerce_weights<T: NumCast>(weights: &[f64]) -> Vec<T>
where
{
    weights
        .iter()
        .map(|v| NumCast::from(*v).unwrap())
        .collect::<Vec<_>>()
}

#[derive(Debug, Clone)]
pub struct ExponentialWeights<T> {
    weights: Vec<T>, // Use Vec<T> instead of Vec<f64>
    decay_type: ExponentialDecayType,
    normalized: bool,
    pub normalizer: T,
}

impl<T> ExponentialWeights<T>
where
    T: Copy
    + Add<Output = T>
    + Sub<Output = T>
    + Mul<Output = T>
    + Div<Output = T>
    + DivAssign
    + NumCast
    + Zero
    + iter::Sum
    + PartialEq,
{
    pub fn new(
        decay_type: ExponentialDecayType,
        periods: i32, normalized: bool)
        -> Self {
        let mut weights = Self {
            weights: Vec::new(),
            decay_type,
            normalized,
            normalizer: NumCast::from(decay_type.get_normalizer()).unwrap(),
        };
        weights.construct_weights(periods);
        weights
    }

    pub fn normalize(&mut self) -> &mut Self {
        let wsum: &T = &self.weights.iter().copied().sum();
        if *wsum != T::zero() {
            self.weights.iter_mut().for_each(|w| *w /= *wsum);
        }
        self.normalized = true;
        self
    }

    pub fn coerce_weights<U: NumCast>(weights: &[f64]) -> Vec<U> {
        weights
            .iter()
            .map(|v| NumCast::from(*v).unwrap()) // Convert `f64` to `U`
            .collect::<Vec<_>>()
    }

    pub fn construct_weights(&mut self, periods: i32) {
        let half_life = self.decay_type.get_half_life().unwrap();
        let weights_f64 = exp_weights(periods, Some(half_life)).unwrap();
        let weights = Self::coerce_weights::<T>(&weights_f64);
        self.weights = weights;

        if self.normalized {
           self.normalize();
        }
    }

    pub fn get_weights(&self) -> &[T] {
        &self.weights
    }

    /// Get weight at a specific index without bounds checking
    #[inline(always)]
    pub fn get_weight_unchecked(&self, index: usize) -> T {
        self.weights[index]
    }

    /// Get a slice of weights without bounds checking
    pub fn get_slice_unchecked(&self, from: usize, to: usize) -> &[T] {
        &self.weights[from..to]
    }


    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn leading_weight(&self) -> T {
        self.weights[self.weights.len() - 1]
    }

    pub fn trailing_weight(&self) -> T {
        self.weights[0]
    }

}

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
    pub fn get_normalizer(&self) -> f64 {
        f64::exp(self.get_decay().unwrap())
    }

    pub fn get_decay(&self) -> Result<f64, ExpWeightsError> {
        Ok(std::f64::consts::LN_2 / self.get_half_life()? as f64)
    }
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
        let weights = exp_weights(5, Some(2.0)).unwrap();
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
            exp_weights(0, Some(2.0)),
            Err(ExpWeightsError::InvalidWindow(0))
        ));

        assert!(matches!(
            exp_weights(-1, Some(2.0)),
            Err(ExpWeightsError::InvalidWindow(-1))
        ));
    }

    #[test]
    fn test_exp_weights_invalid_half_life() {
        assert!(matches!(
            exp_weights(5, Some(0.0)),
            Err(ExpWeightsError::InvalidHalfLife(0.0))
        ));

        assert!(matches!(
            exp_weights(5, Some(-1.0)),
            Err(ExpWeightsError::InvalidHalfLife(-1.0))
        ));
    }
}
