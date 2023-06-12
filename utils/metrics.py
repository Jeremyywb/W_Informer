from typing import Any, List, Tuple, Dict
from abc import ABC, abstractmethod
import sklearn.metrics as metrics
from torchmetrics.classification import MultilabelF1Score
from torchmetrics import AUROC
import numpy as np
import torch


    
    
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

    
class MULTIF1(Metric):
    """F1_score.
    """
    _NAME = "multifl"

    def __init__(
        self
    ):
        super(MULTIF1, self).__init__()
    def metric_fn(
        self, 
        y_true: torch.Tensor, 
        y_score: torch.Tensor,
        # num_lables,
        mask=None
    ) -> float:
        metric = MultilabelF1Score(num_labels=y_true.shape[1])
        f1 = metric(y_score, y_true)
        return f1
    
    
    
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
        # y_true: np.ndarray, 
        # y_pred: np.ndarray,
        y_true: torch.Tensor, 
        y_pred: torch.Tensor,
        **kwargs
    ) -> float:
        # return np.mean(metrics.roc_auc_score(y_true,y_score, average=None))

        metric = AUROC(task="multilabel", num_labels=y_true.shape[1])
        _ = metric(y_pred, y_true.int())
        auroc = metric.compute().item()
        metric.reset()

        return auroc



def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div


class MULSMAPE(Metric):
    """F1_score.
    """
    _NAME = "mulsmape"

    def __init__(
        self
    ):
        super(MULSMAPE, self).__init__()
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray,
        mask=None
    ) -> float:
        if mask is None:
            mask = np.ones(y_score.shape)
        delta_y = np.abs((y_true - y_score))
        scale = np.abs(y_true) + np.abs(y_score)
        smape = divide_no_nan(delta_y, scale)
        smape = smape * mask
        smape = 2 * np.mean(smape)
        return 100 * np.mean(smape)


class MAE(Metric):
    """F1_score.
    """
    _NAME = "mae"

    def __init__(
        self
    ):
        super(MAE, self).__init__()
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray,
        mask=None
    ) -> float:
        mae = np.abs(y_true - y_score)
        return np.mean(mae)
    
    