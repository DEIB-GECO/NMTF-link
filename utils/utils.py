import sys
from enum import Enum

def bold(s):
    return f'\033[1m {s} \033[0m'

class EvaluationMetric(Enum):
    APS = 'APS'
    AUROC = 'AUROC'
    PEARSON = 'PEARSON'
    RMSE = 'RMSE'

    @classmethod
    def _missing_(cls, value):
        print(f'Invalid metric option <{value}>, switched to default RMSE',
              file=sys.stderr)
        return EvaluationMetric.RMSE


class StopCriterion(Enum):
    MAXIMUM_METRIC = 'MAXIMUM_METRIC'
    MAXIMUM_ITERATIONS = 'MAXIMUM_ITERATIONS'
    RELATIVE_ERROR = 'RELATIVE_ERROR'

    @classmethod
    def _missing_(cls, value):
        print(f'Invalid metric option <{value}>, switched to default MAXIMUM_METRIC',
              file=sys.stderr)
        return StopCriterion.MAXIMUM_METRIC

