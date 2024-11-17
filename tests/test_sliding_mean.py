import py_polars_ext  # noqa: F401
import polars as pl


def test_sliding_mean():
    df = pl.DataFrame(
        {
            "asset_returns": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    result = df.with_columns(pl.col("asset_returns").sliding(window=2).mean())

    expected_df = pl.DataFrame(
        {
            "asset_returns": [None, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        }
    )

    assert result.equals(expected_df)
