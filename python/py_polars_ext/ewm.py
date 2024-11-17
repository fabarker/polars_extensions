import polars as pl
from typing import Self
from py_polars_ext.window import Window, Expanding, Sliding
from py_polars_ext.enums.windowing import WinType



class ExponentialMoving(Window):
    _expr: pl.Expr

    def __init__(self,
                 expr: pl.Expr,
                 window_type: WinType):


        super(ExponentialMoving, self).__init__(expr, window_type)

        # Exponentially weighted specific properties
        self._bias = False
        self._symbol = 'exponentially_weighted'


    def __call__(self,
                 **kwargs) -> Self:

        if ((bias := kwargs.get('bias')) is not None
                and isinstance(bias, bool)):
            self._bias = kwargs.pop('bias')

        if ((bias := kwargs.get('adjust')) is not None
                and isinstance(bias, bool)):
            self._adjust = kwargs.pop('adjust')

        if 'com' in kwargs.keys():
            self._decay = ('com', float(kwargs.pop('com')))
        elif 'alpha' in kwargs.keys():
            self._decay = ('alpha', float(kwargs.pop('alpha')))
        elif 'half_life' in kwargs.keys():
            self._decay = ('half_life', float(kwargs.pop('half_life')))
        elif 'span' in kwargs.keys():
            self._decay = ('span', float(kwargs.pop('span')))

        # add the other kwargs to the
        kwargs.update({'_symbol': self._symbol})
        super(ExponentialMoving, self).__call__(**kwargs)
        return self


    def pars(self):
        if self._winType == WinType.EXPANDING:
            return Expanding.cls_pars(self._ignore_nans)
        else:
            return Sliding.cls_pars(self._window, self._center)