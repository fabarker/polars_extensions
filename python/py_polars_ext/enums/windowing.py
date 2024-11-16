from enum import Enum

class DecayType(Enum):

    COM = 'COM'
    ALPHA = 'ALPHA'
    HALF_LIFE = 'HALF_LIFE'
    SPAN = 'SPAN'

    def __call__(self, value):
        self._val = value
        return self

    def tup(self):
        if hasattr(self, '_val'):
            return self.value.lower(), self._val
        raise ValueError('Error - no value set for {}'.format(self.value))

class WinType(Enum):

    ROLLING = 'rolling'
    EXPANDING = 'expanding'

class UDF(Enum):

    MEAN =     'mean'
    CGR =      'cgr'
    STD =      'std'
    MIN =      'min'
    MAX =      'max'
    SUM =      'sum'
    MEDIAN =   'median'
    SKEW =     'skew'
    KURTOSIS = 'kurt'
    VAR =      'var'
    PRODUCT =  'product'

    def f(self,
          win_type: WinType,
          exponentially_weighted: bool = False) -> str:

        if exponentially_weighted:
           return 'exp_wt_' + self.name.lower()
        else:
            return win_type.name.lower() + '_' + self.name.lower()