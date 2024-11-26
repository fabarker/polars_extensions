pub mod utils;
use crate::utils::weights::*;

#[derive(Debug)]
pub struct Utils; // Unit struct

impl Utils {
    pub fn exponential_weights(
        window: i32,
        weight_type: &ExponentialDecayType,
        normalize: bool,
    ) -> Result<Vec<f64>, ExpWeightsError> {
        let half_life = weight_type.get_half_life()?;
        let mut weights = exp_weights(window, Some(half_life))?;

        if normalize {
            let wsum: f64 = weights.iter().sum();
            if wsum != 0.0 {
                weights.iter_mut().for_each(|w| *w /= wsum);
            }
        }
        Ok(weights)
    }
}
