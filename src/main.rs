use polars::prelude::{DataFrame, SerReader, *};
use polars_arrow::legacy::kernels::rolling::no_nulls::QuantileInterpolOptions;
use polars_arrow::legacy::kernels::rolling::RollingQuantileParams;
use polars_core::prelude::*;
use polars_custom_utils::utils::weights::ExponentialDecayType;
use polars_custom_utils::Utils;
use polars_windowing::expr::MyCustomTrait;
use polars_windowing::{ExpandingParams, RollingExpandingType, SeriesRollingExt, WindowParams};

fn load_returns() -> PolarsResult<DataFrame> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(
            "/Users/francisbarker/Desktop/PUT_History.csv".into(),
        ))?
        .finish();

    df
}

fn parquet_polars() -> PolarsResult<DataFrame> {
    let mut file = std::fs::File::open("/Users/francisbarker/repo/toraniko/US Equity").unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();
    dbg!(&df);
    Ok(df)
}

fn main() -> PolarsResult<()> {
    let mut ts = load_returns()?;
    let puts = Series::new("symbol".into(), vec!["PUT"; ts.height()]);
    let ts = ts.with_column(puts)?;
    let new_names = vec!["date", "asset_returns", "symbol"];
    ts.set_column_names(new_names)?;
    dbg!(&ts);

    let kw = WindowParams {
        decay: Some(ExponentialDecayType::Alpha(0.5)),
        min_periods: 1,
        weights: None,
        adjust: false,
        window_type: RollingExpandingType::Expanding(ExpandingParams { ignore_nans: false }),
        bias: None,
    };

    let decay = ExponentialDecayType::HalfLife(126.0);
    let wts = Utils::exponential_weights(504, &decay, false).unwrap();
    dbg!(&wts);

    let mut chunked = ts
        .column("asset_returns")?
        .f64()?
        .into_iter()
        .collect::<Vec<Option<f64>>>();
    let new_series = Series::new("data".into(), chunked);
    dbg!(&new_series);
    let s = new_series
        .shift(20)
        .rolling(504)
        .with_weights(Option::from(wts))
        .cagr();
    dbg!(&s);

    Ok(())
}
