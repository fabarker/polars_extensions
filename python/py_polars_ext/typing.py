from typing import TYPE_CHECKING, Union
from py_polars_ext._utils import pl_plugin
from py_polars_ext._utils import *

if TYPE_CHECKING:
    import sys
    import polars as pl

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    import DataType, DataTypeClass

    IntoExprColumn: TypeAlias = Union[pl.Expr, str, pl.Series]
    PolarsDataType: TypeAlias = Union[DataType, DataTypeClass]
