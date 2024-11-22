from abc import ABC, abstractmethod
import polars as pl
from py_polars_ext.enums.windowing import WinType, UDF
import numpy.typing as npt
from py_polars_ext._utils import pl_plugin
from typing import Dict, Any, Optional
from typing_extensions import Self
from enum import Enum

_DROP_FIELDS = [
    "_expr",
    "_window_type",
    "_ignore_nans",
    "_center",
    "_window",
    "_symbol",
    "_winType",
]


class ABCWindow(ABC):
    @abstractmethod
    def pars(self):
        return NotImplemented


class Window(ABCWindow):
    def __init__(self, expr: pl.Expr, winType: WinType):
        """Initialize a Rolling window computation object.
        Args:
            expr (pl.Expr): The Polars expression to perform rolling computations on.
        """
        self._expr = expr
        self._winType = winType
        self._min_periods = 1
        self._weights = None
        self._ignore_nans = False
        self._adjust = False
        self._symbol = "windowed_stats"

    @property
    def min_periods(self) -> int:
        return self._min_periods

    @abstractmethod
    def pars(self):
        return NotImplemented

    @abstractmethod
    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)

    def params(self):
        return self.kwargs() | self.pars()

    def kwargs(self):
        return {
            k.lstrip("_"): v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
            if k not in _DROP_FIELDS
        }

    def __call(self, func: UDF) -> pl.Expr:
        return pl_plugin(
            args=[self._expr, pl.lit(func)],
            symbol=self._symbol,
            is_elementwise=True,
            kwargs=self.params(),
        )

    def ewm(self, **kwargs):
        from py_polars_ext.ewm import ExponentialMoving

        return ExponentialMoving(
            window_type=self._winType, expr=self._expr
        ).__call__(**kwargs | self.__dict__)

    def mean(self) -> pl.Expr:
        """Calculate the rolling mean (average) over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling mean calculation.
        """
        return self.__call(UDF.MEAN)

    def std(self) -> pl.Expr:
        """Calculate the rolling standard deviation over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling standard deviation calculation.
        """
        return self.__call(UDF.STD)

    def cgr(self) -> pl.Expr:
        """Calculate the rolling compound growth rate over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling CGR calculation.
        """
        return self.__call(UDF.CGR)

    def min(self) -> pl.Expr:
        """Calculate the rolling minimum over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling minimum calculation.
        """
        return self.__call(UDF.MIN)

    def max(self) -> pl.Expr:
        """Calculate the rolling maximum over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling maximum calculation.
        """
        return self.__call(UDF.MAX)

    def median(self) -> pl.Expr:
        """Calculate the rolling median over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling median calculation.
        """
        return self.__call(UDF.MEDIAN)

    def var(self) -> pl.Expr:
        """Calculate the rolling variance over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling variance calculation.
        """
        return self.__call(UDF.VAR)

    def sum(self) -> pl.Expr:
        """Calculate the rolling sum over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling sum calculation.
        """
        return self.__call(UDF.SUM)

    def skew(self) -> pl.Expr:
        """Calculate the rolling skewness over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling skewness calculation.
        """
        return self.__call(UDF.SKEW)

    def kurt(self) -> pl.Expr:
        """Calculate the rolling kurtosis over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling kurtosis calculation.
        """
        return self.__call(UDF.KURTOSIS)

    def prod(self) -> pl.Expr:
        """Calculate the rolling product over the specified window.

        Returns:
            pl.Expr: A Polars expression representing the rolling product calculation.
        """
        return self.__call(UDF.PRODUCT)


@pl.api.register_expr_namespace("sliding")
class Sliding(Window):
    def __init__(self, expr: pl.Expr):
        super(Sliding, self).__init__(expr, WinType.ROLLING)

        # Set the internal properties of an sliding window
        self._window: Optional[int] = None
        self._center: bool = False

    @property
    def window(self) -> Optional[int]:
        return self._window

    @property
    def center(self) -> bool:
        return self._center

    def __call__(  # type: ignore[override]
        self,
        window: int,
        min_periods: Optional[int] = None,
        center: bool = False,
        weights: Optional[npt.ArrayLike] = None,
        adjust: bool = False,
    ) -> Self:
        """Configure the rolling window parameters.

        Args:
            window (int): Size of the moving window. Must be positive.
            min_periods (Optional[int]): Minimum number of observations required to have a value.
                Defaults to None, which is equivalent to the window size.
            center (bool): If True, set the window labels at the center of the window.
                Defaults to False.
            weights (Optional[npt.ArrayLike]): An array of weights with the same length as the window.
                Defaults to None, which gives equal weights to all observations.
            adjust (bool): If True, divide by the decaying adjustment factor in beginning periods
                to account for imbalanced weights. Defaults to False.

        Returns:
            Self: The Rolling object with updated parameters for method chaining.

        Raises:
            ValueError: If window size is not positive or min_periods is larger than window size.
        """

        if isinstance(window, int) and window <= 0:
            raise ValueError("Window size must be positive")
        if min_periods is not None and min_periods > window:
            raise ValueError("min_periods cannot be larger than window")

        super(Sliding, self).__call__(
            _window=window,
            _min_periods=min_periods if min_periods else window,
            _center=center,
            _weights=weights,
            _adjust=adjust,
        )
        return self

    def pars(self):
        return self.cls_pars(self.window, self.center)

    @staticmethod
    def cls_pars(window: int, center: bool) -> Dict[str, Any]:
        return {
            "window_type": {"Rolling": {"window": window, "center": center}},
        }


@pl.api.register_expr_namespace("expanding")
class Expanding(Window):
    _MIN_PERIODS: int = 1

    def __init__(self, expr: pl.Expr):
        super(Expanding, self).__init__(expr, WinType.EXPANDING)

        # Set the internal properties of an expanding window
        self._min_periods = Expanding._MIN_PERIODS
        self._ignore_nans = False

    @property
    def ignore_nans(self) -> bool:
        return self._ignore_nans

    def __call__(  # type: ignore[override]
        self,
        min_periods: Optional[int] = 1,
        weights: Optional[npt.ArrayLike] = None,
        adjust: bool = False,
        ignore_nans: bool = False,
    ) -> Self:
        """Configure the expanding window parameters.

        Args:
            window (int): Size of the moving window. Must be positive.
            min_periods (Optional[int]): Minimum number of observations required to have a value.
                Defaults to None, which is equivalent to the window size.
            center (bool): If True, set the window labels at the center of the window.
                Defaults to False.
            weights (Optional[npt.ArrayLike]): An array of weights with the same length as the window.
                Defaults to None, which gives equal weights to all observations.
            adjust (bool): If True, divide by the decaying adjustment factor in beginning periods
                to account for imbalanced weights. Defaults to False.

        Returns:
            Self: The Expanding object with updated parameters for method chaining.

        Raises:
            ValueError: If window size is not positive or min_periods is larger than window size.
        """

        super(Expanding, self).__call__(
            _min_periods=min_periods,
            _weights=weights,
            _adjust=adjust,
            _ignore_nans=ignore_nans,
        )
        return self

    def pars(self):
        return self.cls_pars(self.ignore_nans)

    @staticmethod
    def cls_pars(ignore_nans: bool) -> Dict[str, Any]:
        return {
            "window_type": {"Expanding": {"ignore_nans": ignore_nans}},
        }
