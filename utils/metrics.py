from typing import Any, List, Tuple, Dict
from abc import ABC, abstractmethod
import sklearn.metrics as metrics
import numpy as np

class Metric(ABC):
    def __init__(self):
        pass
    @classmethod
    def get_metrics_by_names(cls, names: List[str]) -> List["Metric"]:
        available_metrics = cls.__subclasses__()
        
        available_names = [metric._NAME for metric in available_metrics]
        print(available_names)
        metrics = []
        for name in names:
            assert (name in available_names
            ), f"{name} is not available, choose in {available_names}"
            idx = available_names.index(name)
            metric = available_metrics[idx]()
            metrics.append(metric)
        return metrics
    
class ACC(Metric):
    _NAME = "acc"

    def __init__(
        self
    ):
        super(ACC, self).__init__()
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray
    ) -> float:
        return metrics.accuracy_score(y_true, y_score)

class F1_macro(Metric):
    """F1_score.
    """
    _NAME = "f1_macro"

    def __init__(
        self
    ):
        super(F1_macro, self).__init__()
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray,
        **kwargs
    ) -> float:
        return metrics.f1_score(y_true, y_score, average="macro")

class AUCROC(Metric):
    """F1_score.
    """
    _NAME = "aucroc"

    def __init__(
        self
    ):
        super(AUCROC, self).__init__()
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray,
        **kwargs
    ) -> float:
        return metrics.roc_auc_score(y_true,y_score, multi_class='ovo')




        