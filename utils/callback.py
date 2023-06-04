from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
import logging
import sys
import torch
import time
import datetime
import copy


class EarlyStopping:
    def __init__(self, early_stopping_metric,patience=7, verbose=False,max_minze=True, delta=0):
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._best_score = None
        self._early_stop = False
        self._MAXIMIZE = max_minze
        self._delta = delta
        self.early_stopping_metric = early_stopping_metric

    def __call__(self, on_stop_sc, model):
        if self._MAXIMIZE:
            self.score = on_stop_sc
        else:
            self.score = -on_stop_sc

        if self._best_score is None:
            self._best_score = self.score
        elif self.score < self._best_score + self._delta:
            self._counter += 1
            print(f'EarlyStopping counter: {self._counter} out of {self._patience}')
            if self._counter >= self._patience:
                self._early_stop = True
        else:
            self._best_score = self.score
            self._best_model = model.state_dict()
            self._counter = 0




class History(object):
    """Callback that records events into a `History` object.

    Args:
        verbose(int): Print results every verbose iteration.

    Attributes:
        _verbose(int): Print results every verbose iteration.
        _history(Dict[str, Any]): Record all information of metrics of each epoch.
        _start_time(float): Start time of training.
        _epoch_loss(float): Average loss per epoch.
        _epoch_metrics(Dict[str, Any]): Record all information of metrics of each epoch.
        _samples_seen(int): Traversed samples.
    """
    def __init__(
        self,
        logPath:str='log',
        logfile:str='log',
        modelPath:str='path/to/model/',
        epochs:int=1,
        num_checkpoints:int=10,

        verbose: int = 1
    ):
        super(History, self).__init__()
        self._verbose = verbose
        self.modelPath = modelPath
        
        self._epochs = epochs
        self._per_epoch = max(1,int(self._epochs/ num_checkpoints))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        # stdout_handler = logging.StreamHandler(sys.stdout)
        # stdout_handler.setLevel(logging.DEBUG)
        # stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(f'{logPath}/{logfile}.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        # self.logger.addHandler(stdout_handler)

    def _save_checkpoint(
        self, 
        State, #from EarlyStopping
        score,
        best_score, #from EarlyStopping
         path,  #from trainner
         prefix #from trainner
    ):
        saveName = f'{path}/_{prefix}checkpoint.pt'
        self.logger.info( f'save checkpoint:{saveName}' )
        if self._verbose:
            print(f'Validation score decreased ({best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(State,saveName)

    def on_train_begin(
        self,
        logs: Optional[Dict[str, Any]] = None
    ):
        """Called at the start of training.

        Args:
            logs(Dict[str, Any]|None): The logs is a dict or None.
        """
        self._history = {"loss": [], "lr": []}
        self._start_time = logs["start_time"]
        self._epoch_loss = 0. # nqa

    def on_epoch_begin(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """Called at the beginning of each epoch.

        Args:
            epoch(int): The index of epoch.
            logs(Dict[str, Any]|None): The logs is a dict or None.
        """
        self._epoch_metrics = {"Train Loss": 0.} # nqa
        self._samples_seen = 0.

    def on_epoch_end(
        self,
        state_dict,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """Called at the end of each epoch.

        Args:
            epoch(int): The index of epoch.
            logs(Dict[str, Any]|None): The logs is a dict or None.
                contains `loss` and `metrics`.
        """
        # self._epoch_metrics["loss"] = self._epoch_loss
        for metric_name, metric_value in self._epoch_metrics.items():
            if metric_name not in self._history:
                self._history.update({metric_name: []})
            self._history[metric_name].append(metric_value)
        if epoch%self._per_epoch==0:
            torch.save(state_dict, f'{self.modelPath}/modelhistory_epoch{epoch}_checkpoint.pt')

        if self._verbose == 0 or epoch % self._verbose != 0:
            return

        msg = f"epoch {epoch:0>3}/{self._epochs}==============\n"
        for metric_name, metric_value in self._epoch_metrics.items():
            if metric_name != "lr":
                msg += f"| {metric_name:<3}: {metric_value:.6f}"
        total_time = int(time.time() - self._start_time)
        msg += f"| {str(datetime.timedelta(seconds=total_time)) + 's':<6}" 
        # print(msg)
        self.logger.info(msg)

    def on_batch_end(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """Called at the end of each batch in training.

        Args:
            batch(int): The index of batch.
            logs(Dict[str, Any]|None): The logs is a dict or None.
                contains `loss` and `batch_size`.
        """
        batch_size = logs["batch_size"]
        self._epoch_loss = (
            self._samples_seen * self._epoch_loss + batch_size * logs["loss"]
        ) / (self._samples_seen + batch_size)
        self._samples_seen += batch_size
