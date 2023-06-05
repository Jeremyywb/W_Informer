from typing import List, Dict, Any, Callable, Optional, Tuple
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from utils.callback import EarlyStopping
from collections import OrderedDict
from utils.callback import History
from utils.metrics import Metric
from copy import deepcopy
import numpy as np
import logging
import time
import torch
import gc
import os

class torchModel(object):
    """docstring for torchModel"""
    def __init__(
      self,
      model=None,
      cfg=None,
      CheckpointPath:str='None',
      logPath:str='None',
      logfile:str="None",
      num_checkpoints:int=10
      
      ):

        super(torchModel, self).__init__()
        self._model = model
        self._cfg = cfg
        self._CheckpointPath = CheckpointPath
        self._logPath = logPath
        self._logfile = logfile
        self._num_checkpoints = num_checkpoints
        
        if not os.path.exists(CheckpointPath):
            os.mkdir(CheckpointPath)        
        if not os.path.exists(logPath):
            os.mkdir(logPath)
    def compile(
      self,
      loss=None,
      optimizer=None,
      lr:float=1e-3,
      eval_metrics:List[str]=[],
      early_stopping=None,
      verbose:int=1
      ):
        self._lr = lr
        self._verbose = verbose
        self._loss_fun = loss
        self._eval_metrics = eval_metrics
        self.optimizer = optimizer
        self._metrics = Metric.get_metrics_by_names(eval_metrics)
        self._early_stopping = early_stopping

    

    # if early_stopping is not None:
    #   self.early_stopping = early_stopping
    #   assert early_stopping.early_stopping_metric in eval_metrics
    #   idx = eval_metrics.index(early_stopping.early_stopping_metric)
    #   self.early_stopping_metric = self.metrics[idx]


    def fit(
    self,
    train:DataLoader,
    valid:DataLoader,
    epochs:int
    ):

    # optim.Adam
        self._optimizer = self.optimizer(
                    self._model.parameters(),
                    lr=self._lr,
                    weight_decay=self._cfg.WEIGHT_DECAY)
        self._epochs = epochs
        self._scheduler = OneCycleLR(self._optimizer,max_lr = 1e-3, # Upper learning rate boundaries in the cycle for each parameter group
                                steps_per_epoch = len(train), # The number of steps per epoch to train for.
                                epochs = epochs, # The number of epochs to train for.
                                anneal_strategy = 'cos') # 
        self._history = History(
                    logPath         = self._logPath,
                    logfile         = self._logfile,
                    modelPath       = self._CheckpointPath,
                    epochs          = epochs,
                    num_checkpoints = self._num_checkpoints,
                    verbose         = self._verbose
               )

        self._history.on_train_begin( {"start_time": time.time()} )
        for epoch in range(epochs):
            iter_count = 0
            
            self._model.train()
            self._on_epoch_begin(epoch)
            epoch_logs = self._train_epoch(train)

            self._model.eval()
            eval_epoch_logs = self._eval_epoch(valid)
            epoch_logs.update( eval_epoch_logs )

            self._on_epoch_end(epoch,epoch_logs)
            if self._early_stopping._early_stop:
                self._history._save_checkpoint( 
                    self._early_stopping._best_model,
                    self._early_stopping.score,
                    self._early_stopping._best_score,
                    self._CheckpointPath,
                    'OneFOldBest'

                    ) 
                break
    def _on_epoch_begin(self,epoch):
        self._history.on_epoch_begin(epoch)
    def _on_epoch_end(self,epoch,logs):
        state_dict = {
          'modelstate_dict': self._model.state_dict(),
          'optimizer_state_dict': self._optimizer.state_dict(),
          'scheduler_state_dict': self._scheduler.state_dict()
            }
        on_stop_sc =logs[ f'Valid {self._early_stopping.early_stopping_metric}']

        self._early_stopping( on_stop_sc, self._model ) 
        self._history._epoch_metrics.update(logs)
        self._history.on_epoch_end(state_dict,epoch,logs)
    def _train_epoch(self,train_loader):
        train_loss = []
        for batch_idx, batch in enumerate(train_loader):
            self._optimizer.zero_grad()
            output,y_true = self._process_one_batch(batch)
            self._loss = self._loss_fun(output, y_true)
            train_loss.append(self._loss.item())
  
            self._loss.backward()
            self._optimizer.step()
            self._scheduler.step()
        train_loss = np.average(train_loss)
        epoch_logs = {"lr": self._scheduler.get_lr(),'Train Loss': train_loss}
        return epoch_logs
    @torch.no_grad()
    def _eval_epoch(self,eval_loader,thres=0.63):
        eval_loss_total = 0
        for i, batch_data in enumerate(eval_loader):
            output,y = self._process_one_batch(batch_data)
            loss = self._loss_fun(output, y)
            eval_loss_total += loss.item()
            if i == 0:
                y_true = y.detach().cpu()
                y_pred = output.detach().cpu()
            else:
                # Avoid situ ation that batch_size is just equal to 1
                y_true = torch.cat((y_true, y.detach().cpu()))
                y_pred = torch.cat((y_pred, output.detach().cpu()))

                del y, output
                _ = gc.collect()

        eval_loss_avg = eval_loss_total / len(eval_loader)
        eval_epoch_logs = {'Valid Loss':eval_loss_avg}
        for evname,metric in zip(self._eval_metrics,self._metrics):
            if evname == 'f1_macro':
                
                y_true1 = y_true.numpy().reshape(-1, )
                for o in self._cfg.CKPT_METRIC:
                    # float("f1@0.49".split('@')[1])
                    thres = float(o.split('@')[1])
                    y_pred1 = (y_pred.numpy().reshape(-1, ) > thres).astype("int")
                    _score  = metric.metric_fn(y_true1, y_pred1)
                    eval_epoch_logs[f'Valid {o}'] = _score
            else:
                y_pred1 = y_pred.numpy()
                y_true1 = y_true.numpy()
                _score  = metric.metric_fn(y_true1, y_pred1)
                eval_epoch_logs[f'Valid {evname}'] = _score
        del y_pred1,y_true1,y_pred,y_true
        _ = gc.collect()
        return eval_epoch_logs


    def _process_one_batch(self,batch):
        raise NotImplementedError()
    def predict(self,batch):
        raise NotImplementedError()

