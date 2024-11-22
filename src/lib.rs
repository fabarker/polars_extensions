mod window;
use polars_custom_utils::utils::weights::ExponentialDecayType;
use polars_custom_utils::Utils;
pub use polars_custom_utils::*;
pub use polars_windowing::*;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{pyfunction, pymodule, wrap_pyfunction, Bound, PyResult};
use pyo3_polars::PolarsAllocator;

#[pyfunction]
pub fn exponential_weights(window: i32, half_life: f32, normalize: bool) -> PyResult<Vec<f64>> {
    Ok(Utils::exponential_weights(
        window,
        &ExponentialDecayType::HalfLife(half_life),
        normalize,
    )
    .unwrap())
}

#[pymodule]
fn pypolars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(exponential_weights, m)?)?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
