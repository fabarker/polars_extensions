use polars::prelude::{DataFrame, SerReader, *};
use polars_arrow::legacy::kernels::rolling::no_nulls::QuantileInterpolOptions;
use polars_arrow::legacy::kernels::rolling::RollingQuantileParams;
use polars_core::prelude::*;
use polars_custom_utils::utils::weights::ExponentialDecayType;
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

    let res = ts.column("asset_returns")?.clone();
    let binding = [res];

    let params = RollingQuantileParams {
        interpol: QuantileInterpolOptions::Linear,
        prob: 0.5,
    };

    let opts = RollingOptionsFixedWindow {
        window_size: 504,
        min_periods: 504,
        weights: None,
        center: false,
        fn_params: Some(Arc::new(params)),
    };

    let s = ts.column("asset_returns")?.rolling(252).quantile(0.5)?;
    dbg!(&s);

    //let s = ts.column("asset_returns")?.rolling_quantile(opts)?;

    //let decay = ExponentialDecayType::HalfLife(126.0);
    //let wts = Utils::exponential_weights(504, &decay).unwrap();

    //dbg!(&wts);
    let opts = RollingOptionsFixedWindow {
        window_size: 504,
        min_periods: 504,
        weights: None,
        center: false,
        fn_params: None,
    };

    let lagged = ts
        .clone()
        .lazy()
        .with_columns([col("asset_returns")
            .shift(lit(20))
            .over([col("symbol")])
            .alias("lagged_asset_returns")])
        .collect()?;
    dbg!(&lagged);

    let mom = lagged
        .lazy()
        .with_columns([col("lagged_asset_returns")
            .fill_null(0)
            .rolling_cagr(opts)
            .over([col("symbol")])
            .alias("mom_score")])
        .collect()?;
    dbg!(&mom);

    //let duration = start.elapsed();

    // Print the time taken
    //println!("Time taken for rolling_mean: {:?}", duration);

    // Add rolling sum columns
    //let df = s.clone().into_frame()
    //    .lazy()
    //    .with_columns([col("PUT").ewm_std(opts).alias("ewm_rolling_mean")])
    //    .collect()?;
    //dbg!(&df);

    // Print the first and last few rows of the DataFrame
    Ok(())
}
