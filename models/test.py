import os
from torch import optim, nn, utils, Tensor
# import lightning.pytorch as pl
import torch.nn.functional as F
from models.Tainer import torchModel
from utils.callback import EarlyStopping
from utils.customloss import SMAPELoss

class trainModel(torchModel):
    """docstring for trainModel"""
    def __init__(self, model,cfg,CheckpointPath, logPath, logfile, num_checkpoints):
        super(trainModel, self).__init__(model,cfg, CheckpointPath, logPath, logfile, num_checkpoints)
        self._cfg = cfg

    def _process_one_batch(self, batch):
        '''
        Parameters
        ----------
        ESx (sequence(float)): targets timeserise for encoder inputs 
        ETx (sequence(int))  : time feats for encoder inputs 
        ECx (sequence(int))  : station name as cate feas for encoder inputs
        DSx (sequence(float)): targets timeserise for encoder inputs
        DTx (sequence(int))  : time feats for encoder inputs
        DCx (sequence(int))  : station name as cate feas for encoder inputs      
        
        Variables
        ---------
        Y (real targets):
        O (predict targets):

        '''
        ESx = batch['seq_x'].float().to(self._cfg.DEVICE)
        ETx = batch['timef_x'].to(self._cfg.DEVICE)
        ECx = batch['stations_x'].to(self._cfg.DEVICE)
        DSx = batch['seq_y'].float().to(self._cfg.DEVICE)
        DTx = batch['timef_y'].to(self._cfg.DEVICE)
        DCx = batch['stations_y'].to(self._cfg.DEVICE)
        Y = DSx[:,-self._cfg.prd_len:,: ].to(self._cfg.DEVICE)
        O = self._model(ESx,ETx,ECx,DSx,DTx,DCx)
        return O, Y
    @torch.no_grad()
    def predict(self,test_loader):
        # test_loader batch size = 1
        self._model.eval()
        preds = []
        for i, batch_data in enumerate(test_loader):
            pred , _ = self._process_one_batch(batch_data)
            preds.append(pred.detach().cpu().numpy())
        preds = np.array(preds)
        return preds

from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

class Datset(Dataset):
    def __init__(
        self,
        data: List[Dict],
    ) -> None:
        self.samples = data
        self.n_samples = len(data)
        
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.samples[ idx ]


def get_model(cfg):
    model = InformerForecast(
        toke_dim=2, 
        seq_len=1024,
        label_len=128, 
        out_len=896, 
        factor=5, 
        d_model=64, 
        n_heads=6, 
        e_layers=3, 
        d_layers=2, 
        d_ff=128, 
        dropout=0.2 
        )
    return model 


cfg.BATCH_SIZE = 128

IFMRegressor = get_model(cfg).to( cfg.DEVICE )
dataset  = Datset(data= train_rnd_sam )
train_loader = DataLoader(dataset,batch_size= cfg.BATCH_SIZE,shuffle=True,collate_fn=None)
dataset2  = Datset(valid_rnd_sam )
valid_loader = DataLoader(dataset2,batch_size= cfg.BATCH_SIZE,shuffle=True,collate_fn=None)


args = {'early_stopping_metric' : 'mulsmape',
'patience' : 8,
'verbose' : True,
'max_minze' : True,
'delta' : 0.0005}


google_root = '/content/sample_data/'
cache_root = '/kaggle/working/'
ckg_p = cache_root + 'checkPoint'
log_p = cache_root + 'log'


earlystopping = EarlyStopping(**args)
TRAINER = trainModel(IFMRegressor,cfg,ckg_p,log_p,'IFMRegressorTrain',20)
TRAINER.compile(
      loss=torch.nn.BCELoss(),
      optimizer=optim.Adam,
      lr=1e-3,
      eval_metrics=['mae','mulsmape'],
      early_stopping=earlystopping,
      verbose=1)
TRAINER.fit(train_loader,valid_loader,epochs=50)