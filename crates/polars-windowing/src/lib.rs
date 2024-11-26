pub mod expanding;
pub mod expr;
pub mod rolling;

use std::iter;
use std::iter::Sum;
use std::ops::{AddAssign, Div, SubAssign};

use arrow::datatypes::*;
use num_traits::{Float, NumCast};
use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray}; // Change this import
use polars_arrow::bitmap::Bitmap;
use polars_arrow::legacy::kernels::rolling::no_nulls::QuantileInterpolOptions;
use polars_arrow::types::NativeType;
use polars_core::prelude::UInt64Type;
use polars_custom_utils::utils::weights::ExponentialDecayType;
use polars_custom_utils::Utils;
use polars_utils::float::IsFloat;
use serde::Deserialize;

use crate::expanding::cagr::expanding_cagr;
use crate::expanding::ewm::mean::ewm_mean;
use crate::expanding::ewm::stdev::ewm_std;
use crate::expanding::ewm::variance::ewm_var;
use crate::expanding::kurtosis::expanding_kurtosis;
use crate::expanding::max::expanding_max;
use crate::expanding::mean::expanding_mean;
use crate::expanding::min::expanding_min;
use crate::expanding::prod::expanding_prod;
use crate::expanding::quantile::expanding_quantile;
use crate::expanding::skew::expanding_skew;
use crate::expanding::stdev::expanding_std;
use crate::expanding::sum::expanding_sum;
use crate::expanding::variance::expanding_var;
use crate::rolling::cagr::{rolling_cagr, ew_rolling_cagr};
use crate::rolling::kurtosis::rolling_kurtosis;
use crate::rolling::mean::{rolling_mean, ew_rolling_mean};
use crate::rolling::prod::rolling_prod;
use crate::rolling::quantile::rolling_quantile;
use crate::rolling::skew::rolling_skew;
use crate::rolling::stdev::{rolling_std, rolling_ewm_std};
use crate::rolling::sum::{rolling_sum, ew_rolling_sum};
use crate::rolling::variance::{rolling_var, ew_rolling_var};
use crate::rolling::RollingAggWindow;

trait MyArrayExt<T>
where
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat,
{
    fn has_nulls(&self) -> bool;
    fn has_nulls_in_range(&self, start: usize, end: usize) -> bool;
}

impl<T> MyArrayExt<T> for PrimitiveArray<T>
where
    T: NativeType + Float + Sum<T> + SubAssign + AddAssign + IsFloat,
{
    fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    fn has_nulls_in_range(&self, start: usize, end: usize) -> bool {
        if let Some(valid) = self.validity() {
            valid
                .iter()
                .skip(start)
                .take(end - start)
                .any(|is_valid| !is_valid)
        } else {
            false
        }
    }
}

trait BitmapExt {
    fn has_nulls_in_range(&self, start: usize, end: usize) -> bool;
}

impl BitmapExt for Bitmap {
    fn has_nulls_in_range(&self, start: usize, end: usize) -> bool {
        self.iter()
            .skip(start)
            .take(end - start)
            .any(|is_valid| !is_valid)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct RollingParams {
    pub window: usize,
    pub center: bool,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ExpandingParams {
    pub ignore_nans: bool,
}

#[derive(Deserialize, Debug, Clone)]
pub enum RollingExpandingType {
    Rolling(RollingParams),
    Expanding(ExpandingParams),
}

impl RollingExpandingType {
    pub fn get_window(&self) -> Result<usize, PolarsError> {
        match self {
            RollingExpandingType::Rolling(params) => Ok(params.window),
            RollingExpandingType::Expanding(_) => Err(PolarsError::ComputeError(
                "window is only available for Rolling windows".into(),
            )),
        }
    }

    pub fn get_center(&self) -> Result<bool, PolarsError> {
        match self {
            RollingExpandingType::Rolling(params) => Ok(params.center),
            RollingExpandingType::Expanding(_) => Err(PolarsError::ComputeError(
                "center is only available for Rolling windows".into(),
            )),
        }
    }

    pub fn get_ignore_nans(&self) -> Result<bool, PolarsError> {
        match self {
            RollingExpandingType::Rolling(_) => Err(PolarsError::ComputeError(
                "ignore_nans is only available for Expanding windows".into(),
            )),
            RollingExpandingType::Expanding(params) => Ok(params.ignore_nans),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct WindowParams {
    pub decay: Option<ExponentialDecayType>,
    pub min_periods: usize,
    pub weights: Option<Vec<f64>>,
    pub adjust: bool,
    pub window_type: RollingExpandingType,
    pub bias: Option<bool>,
}

impl WindowParams {
    pub fn get_win_type(&self) -> &str {
        match &self.window_type {
            RollingExpandingType::Rolling(_) => "rolling",
            RollingExpandingType::Expanding(_) => "expanding",
        }
    }
}

#[derive(Deserialize)]
pub struct RollingKwargs {
    window: usize,
    min_periods: usize,
    center: bool,
    weights: Option<Vec<f64>>,
    adjust: bool,
}

impl RollingKwargs {
    pub fn from_kwargs(kwargs: &WindowParams) -> Self {
        Self {
            window: kwargs.window_type.get_window().unwrap(),
            min_periods: kwargs.min_periods,
            weights: kwargs.weights.clone(),
            adjust: kwargs.adjust,
            center: kwargs.window_type.get_center().unwrap(),
        }
    }
}

#[derive(Deserialize)]
pub struct ExpandingKwargs {
    min_periods: usize,
    weights: Option<Vec<f64>>,
    adjust: bool,
    ignore_nans: bool,
}

impl ExpandingKwargs {
    pub fn from_kwargs(kwargs: &WindowParams) -> Self {
        Self {
            min_periods: kwargs.min_periods,
            weights: kwargs.weights.clone(),
            adjust: kwargs.adjust,
            ignore_nans: kwargs.window_type.get_ignore_nans().unwrap(),
        }
    }
}

#[derive(Debug)]
pub enum WindowType<'a> {
    Rolling(Rolling<'a>),
    Expanding(Expanding<'a>),
}

impl<'a> WindowType<'a> {
    pub fn min_periods(&self) -> usize {
        match self {
            WindowType::Rolling(rolling) => rolling.min_periods,
            WindowType::Expanding(expanding) => expanding.min_periods,
        }
    }
}

#[derive(Debug)]
pub struct ExponentiallyWeighted<'a> {
    window_type: WindowType<'a>,
    decay_type: ExponentialDecayType,
    adjust: bool,
    ignore_nans: bool,
    bias: bool,
}

impl<'a> ExponentiallyWeighted<'a> {
    pub fn with_adjust(mut self, adjust: bool) -> Self {
        self.adjust = adjust;
        self
    }

    pub fn with_ignore_nans(mut self, ignore_nans: bool) -> Self {
        self.ignore_nans = ignore_nans;
        self
    }

    pub fn sum(&self) -> PolarsResult<Series> {
        match &self.window_type {
            WindowType::Expanding(expanding) => panic!("Invalid window type"),
            WindowType::Rolling(rolling) => {
                ew_rolling_sum(
                    rolling.series,
                    rolling.window,
                    rolling.min_periods,
                    rolling.center,
                    &self.decay_type,
                )
            },
        }
    }

    pub fn mean(&self) -> PolarsResult<Series> {
        match &self.window_type {
            WindowType::Expanding(expanding) => {
                ewm_mean(
                    expanding.series, // or expanding.get_series() if that's a method
                    self.decay_type.get_alpha().unwrap() as f64,
                    self.adjust,
                    expanding.min_periods, // or expanding.min_periods() if that's a method
                    self.ignore_nans,
                )
            },
            WindowType::Rolling(rolling) => {
                ew_rolling_mean(
                    rolling.series,
                    rolling.window,
                    rolling.min_periods,
                    rolling.center,
                    &self.decay_type,
                )
            },
        }
    }

    pub fn std(&self) -> PolarsResult<Series> {
        match &self.window_type {
            WindowType::Expanding(expanding) => ewm_std(
                expanding.series,
                self.decay_type.get_alpha().unwrap() as f64,
                self.adjust,
                expanding.min_periods,
                self.ignore_nans,
                self.bias,
            ),
            WindowType::Rolling(rolling) => {
                rolling_ewm_std(
                    rolling.series,
                    rolling.window,
                    rolling.min_periods,
                    rolling.center,
                    self.decay_type,
                )
            },
        }
    }

    pub fn var(&self) -> PolarsResult<Series> {
        match &self.window_type {
            WindowType::Expanding(expanding) => ewm_var(
                expanding.series,
                self.decay_type.get_alpha().unwrap() as f64,
                self.adjust,
                expanding.min_periods,
                self.ignore_nans,
                self.bias,
            ),
            WindowType::Rolling(rolling) => {
                ew_rolling_var(
                    rolling.series,
                    rolling.window,
                    rolling.min_periods,
                    rolling.center,
                    &self.decay_type,
                )
            },
        }
    }

    pub fn prod(&self) -> PolarsResult<Series> {
        match &self.window_type {
            WindowType::Expanding(expanding) => {
                let wts = Utils::exponential_weights(
                    expanding.series.len() as i32,
                    &self.decay_type,
                    false,
                )
                .unwrap();
                expanding_prod(expanding.series, expanding.min_periods, Some(wts))
            },
            WindowType::Rolling(rolling) => {
                let wts =
                    Utils::exponential_weights(rolling.window as i32, &self.decay_type, false)
                        .unwrap();
                rolling_prod(
                    rolling.series,
                    rolling.window,
                    rolling.min_periods,
                    rolling.center,
                    Some(wts),
                )
            },
        }
    }

    pub fn cagr(&self, returns_type: &str) -> PolarsResult<Series> {
        match &self.window_type {
            WindowType::Expanding(expanding) => {
                let wts = Utils::exponential_weights(
                    expanding.series.len() as i32,
                    &self.decay_type,
                    false,
                )
                .unwrap();
                expanding_cagr(expanding.series, expanding.min_periods, Some(wts), returns_type)
            },
            WindowType::Rolling(rolling) => {
                ew_rolling_cagr(
                    rolling.series,
                    rolling.window,
                    rolling.min_periods,
                    rolling.center,
                    &self.decay_type,
                    returns_type
                )
            },
        }
    }
}

#[derive(Debug)]
pub struct Expanding<'a> {
    series: &'a Series,
    min_periods: usize,
    adjust: bool,
    ignore_nans: bool,
    weights: Option<Vec<f64>>,
}

impl<'a> Expanding<'a> {
    fn new(series: &'a Series, min_periods: usize) -> Self {
        Self {
            series,
            min_periods,
            adjust: false,
            ignore_nans: false,
            weights: None,
        }
    }

    pub fn from_kwargs(series: &'a Series, kwargs: ExpandingKwargs) -> Self {
        Self {
            series,
            min_periods: kwargs.min_periods,
            weights: kwargs.weights,
            adjust: kwargs.adjust,
            ignore_nans: kwargs.ignore_nans,
        }
    }

    pub fn ewm(self, decay_type: ExponentialDecayType, bias: bool) -> ExponentiallyWeighted<'a> {
        ExponentiallyWeighted {
            window_type: WindowType::Expanding(self),
            decay_type,
            adjust: false,
            ignore_nans: true,
            bias,
        }
    }

    pub fn with_adjust(mut self, adjust: bool) -> Self {
        self.adjust = adjust;
        self
    }

    pub fn with_weights(mut self, weights: Option<Vec<f64>>) -> Self {
        self.weights = weights;
        self
    }

    pub fn sum(&self) -> PolarsResult<Series> {
        expanding_sum(self.series, self.min_periods, self.weights.clone())
    }

    pub fn mean(&self) -> PolarsResult<Series> {
        expanding_mean(self.series, self.min_periods, self.weights.clone())
    }

    pub fn prod(&self) -> PolarsResult<Series> {
        expanding_prod(self.series, self.min_periods, self.weights.clone())
    }

    pub fn std(&self) -> PolarsResult<Series> {
        expanding_std(self.series, self.min_periods, self.weights.clone())
    }

    pub fn var(&self) -> PolarsResult<Series> {
        expanding_var(self.series, self.min_periods, self.weights.clone())
    }

    pub fn kurt(&self) -> PolarsResult<Series> {
        expanding_kurtosis(self.series, self.min_periods)
    }

    pub fn skew(&self) -> PolarsResult<Series> {
        expanding_skew(self.series, self.min_periods)
    }

    pub fn min(&self) -> PolarsResult<Series> {
        expanding_min(self.series, self.min_periods)
    }

    pub fn max(&self) -> PolarsResult<Series> {
        expanding_max(self.series, self.min_periods)
    }

    pub fn quantile(&self, quantile: f64) -> PolarsResult<Series> {
        expanding_quantile(
            self.series,
            quantile,
            self.min_periods,
            QuantileInterpolOptions::Linear,
        )
    }

    pub fn cagr(&self, returns_type: &str) -> PolarsResult<Series> {
        expanding_cagr(self.series, self.min_periods, self.weights.clone(), returns_type)
    }

    pub fn median(&self) -> PolarsResult<Series> {
        expanding_quantile(
            self.series,
            0.5,
            self.min_periods,
            QuantileInterpolOptions::Linear,
        )
    }
}

#[derive(Debug)]
pub struct Rolling<'a> {
    series: &'a Series,
    window: usize,
    min_periods: usize,
    adjust: bool,
    ignore_nans: bool,
    center: bool,
    weights: Option<Vec<f64>>,
}

impl<'a> Rolling<'a> {
    fn new(series: &'a Series, window: usize) -> Self {
        Self {
            series,
            window,
            min_periods: window,
            adjust: false,
            ignore_nans: false,
            center: false,
            weights: None,
        }
    }

    pub fn from_kwargs(series: &'a Series, kwargs: RollingKwargs) -> Self {
        Self {
            series,
            window: kwargs.window,
            min_periods: kwargs.min_periods,
            center: kwargs.center,
            weights: kwargs.weights,
            adjust: kwargs.adjust,
            ignore_nans: false,
        }
    }

    pub fn ewm(self, decay_type: ExponentialDecayType, bias: bool) -> ExponentiallyWeighted<'a> {
        ExponentiallyWeighted {
            window_type: WindowType::Rolling(self),
            decay_type,
            adjust: false,
            ignore_nans: false,
            bias,
        }
    }

    pub(self) fn get_options(&self) -> RollingOptionsFixedWindow {
        RollingOptionsFixedWindow {
            window_size: self.window, // note: use self not Self
            min_periods: self.min_periods,
            weights: self.weights.clone(),
            center: self.center,
            fn_params: None,
        }
    }

    pub fn with_min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = min_periods;
        self
    }

    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    pub fn with_adjust(mut self, adjust: bool) -> Self {
        self.adjust = adjust;
        self
    }

    pub fn with_weights(mut self, weights: Option<Vec<f64>>) -> Self {
        self.weights = weights;
        self
    }

    pub fn sum(&self) -> PolarsResult<Series> {
        rolling_sum(
            self.series,
            self.window,
            self.min_periods,
            self.center,
            self.weights.clone(),
        )
    }

    pub fn prod(&self) -> PolarsResult<Series> {
        rolling_prod(
            self.series,
            self.window,
            self.min_periods,
            self.center,
            self.weights.clone(),
        )
    }

    pub fn mean(&self) -> PolarsResult<Series> {
        rolling_mean(
            self.series,
            self.window,
            self.min_periods,
            self.center,
            self.weights.clone(),
        )
    }

    pub fn var(&self) -> PolarsResult<Series> {
        rolling_var(
            self.series,
            self.window,
            self.min_periods,
            self.center,
            self.weights.clone(),
        )
    }

    pub fn std(&self) -> PolarsResult<Series> {
        rolling_std(
            self.series,
            self.window,
            self.min_periods,
            self.center,
            self.weights.clone(),
        )
    }

    pub fn quantile(&self, quantile: f64) -> PolarsResult<Series> {
        let params = RollingQuantileParams {
            interpol: QuantileInterpolOptions::Linear,
            prob: quantile,
        };

        let mut opts = self.get_options();
        opts.fn_params = Some(Arc::new(params));
        self.series.rolling_quantile(opts)
    }

    pub fn median(&self) -> PolarsResult<Series> {
        self.quantile(0.5)
    }

    pub fn min(&self) -> PolarsResult<Series> {
        self.series.rolling_min(self.get_options())
    }

    pub fn max(&self) -> PolarsResult<Series> {
        self.series.rolling_max(self.get_options())
    }

    pub fn skew(&self) -> PolarsResult<Series> {
        rolling_skew(self.series, self.window, self.min_periods, self.center)
    }

    pub fn kurt(&self) -> PolarsResult<Series> {
        rolling_kurtosis(self.series, self.window, self.min_periods, self.center)
    }

    pub fn cagr(&self, returns_type: &str) -> PolarsResult<Series> {
        rolling_cagr(
            self.series,
            self.window,
            self.min_periods,
            self.center,
            self.weights.clone(),
            returns_type,
        )
    }
}

// Define trait to add .rolling() method to Series
pub trait SeriesRollingExt {
    fn rolling(&self, window: usize) -> Rolling;
    fn expanding(&self, min_periods: usize) -> Expanding;
}

// Implement the extension trait for Series
impl SeriesRollingExt for Series {
    fn rolling(&self, window: usize) -> Rolling {
        Rolling::new(self, window)
    }
    fn expanding(&self, min_periods: usize) -> Expanding {
        Expanding::new(self, min_periods)
    }
}

#[macro_export]
macro_rules! with_match_physical_float_polars_type {(
    $key_type:expr, | $T:ident $U:ident | $($body:tt)*
) => ({
    match $key_type {
        DataType::Float32 => {
            type T = f32;
            type U = Float32Type;
            $($body)*
        },
        DataType::Float64 => {
            type T = f64;
            type U = Float64Type;
            $($body)*
        },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}
