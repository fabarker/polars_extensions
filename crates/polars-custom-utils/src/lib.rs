pub mod utils;

use pyo3::{pyfunction, PyResult};

use crate::utils::weights::*;

#[derive(Debug)]
pub struct Utils; // Unit struct

impl Utils {
    pub fn exponential_weights(
        window: i32,
        weight_type: &ExponentialDecayType,
    ) -> Result<Vec<f64>, ExpWeightsError> {
        let half_life = weight_type.get_half_life()?;
        exp_weights(window, Some(half_life))
    }
}
