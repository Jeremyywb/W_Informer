from models.attn import FullAttention, ProbAttention, AttentionLayer
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from models.embed import CatesEmbedding,TokenEmbedding,ModalembProj,TokenEmbAndNorm
from models.encoder import ConvLayer,ConvPoolLayer,Encoder,EncoderLayer
import torch.nn.functional as F
from itertools import islice
import torch.nn as nn
import torch

class CrossLayer(nn.Module):
    def __init__(
        self, 
        d_model, 
        d_ff=None,
        n_heads=6,
        factor=35,#top 245|from 512-1024
        dropout=0.1, 
        attType = 'full',
        activation="relu"
        ):

        super(CrossLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        ATT = FullAttention if attType=='full' else ProbAttention
        self._crossLayer = AttentionLayer(
                ATT(False,factor, 
                    attention_dropout=dropout, 
                    output_attention=False), 
                d_model, 
                n_heads, 
                mix=False
            )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # namask = torch.isnan(x)
        x = x + self.dropout(
                self._crossLayer(
                    x, cross, cross,
                    attn_mask=cross_mask )[0]
            )
                    
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))
        x = self.norm2( (x+y))
        if x.dtype == torch.float16 and (
            torch.isinf(x).any() or torch.isnan(x).any()
        ):
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x
            
class EncoderBlock(nn.Module):
    """docstring for SelfAttBlock"""
    def __init__(
        self, 
        crosslayers=None,
        conv1layers=None,
        numcross:int=3,
        numconv1:int=2,
        ):
        super(EncoderBlock, self).__init__()
        if numcross<numconv1+1:
            raise ValueError(f'''invalid parameter numconv1:{numconv1};
                        ->numcross:{numcross} should greater than numconv1 + 1''')
        self._numcross = numcross
        self._numconv1 = numconv1
        self._crossList = nn.ModuleList(crosslayers)
        if conv1layers is None:
            self._conv1List = [None]*len( self._crossList )
        else:
            self._conv1List = nn.ModuleList(conv1layers)
            for i in range(numcross-numconv1):
                self._conv1List.append(None)

    def forward(self,x,cross):
        for encoder,conv1D in zip(self._crossList,self._conv1List ):
            x = encoder(x,cross)
            if conv1D is not None:
                x = conv1D(x)
        return x

class AttEncoders(nn.Module):
    def __init__(
        self,
        iscross,
        d_model,
        numselfs,
        encoderCrosses=None,
        encoderConv1Ds=None,
        numcross:int=3,
        numconv1:int=2,
    ):
    super(AttEncoders, self).__init__()
    self._encoders = nn.ModuleList([
        EncoderBlock(
            crosslayers=crosslayers,
            conv1layers=conv1layers,
            numcross=numcross,
            numconv1=numconv1,
            )
        for crosslayers,conv1layers in zip(encoderCrosses,encoderConv1Ds)
    ])
    self._layerNorms = nn.ModuleList([
        nn.LayerNorm(d_model) for i in range(len( encoderCrosses ))
    ])

    def forward(self,xlist):
        xencodes = [ ]
        if iscross:
            iter_encoders   = iter(self._encoders)
            iter_layernorms = iter(self._encoders)
            for idx,x in enumerate( xlist):
                indices = [_id for _id in range(len(xlist)) if _id != idx]
                for num,restid in enumerate( indices):
                    block = next( iter_encoders )
                    norm  = next( iter_layernorms )
                    if num ==0:
                        y = block(x, xlist[restid] )
                    else:
                        y = norm( y + block( xlist[restid] ))
                xencodes.append(y)
        else:
            for x,block,norm in zip(xlist,self._encoders,self._layerNorms):
                x = block(x,x)
                x = norm(x)
                xencodes.append(x)
        return xencodes

class ModalConbin(nn.Module):
    def __init__(
        self,
        numcomb
    ):
    super(ModalConbin, self).__init__()
    self._project = nn.Sequential(
            nn.Linear(numcomb,numcomb*6),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(numcomb*6,numcomb*6),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(numcomb*6,1),
            nn.ReLU()
    )
    def forward(self,combs:List[torch.Tensor]):
        o = torch.cat([
            x.reshape( ( x.shape[0],-1) ).unsqueeze(-1)
            for x in combs
            ],
            dim=-1
        )

        return self._project(o).reshape( combs[0].shape )



class MultiMInformerClf(nn.Module):
    '''
    Parameters:
        numcross:denotes the number of cross attention layers
           numcross * 3 ,3 is the number of modal parts
        numconv1:denotes the number of layers of conv1D with maxpooling etc.
        cfg: global configuration of parameters

    '''
    def __init__(
        self,
        numcross,
        numconv1,
        numclass,
        cfg
        ):
        super(MultiMInformerClf, self).__init__()
        # nn.Embedding(num_embeddings, embedding_dim, padding_idx
        # nn.Embedding(**arg, padding_idx=0)
        self._cfg = cfg
        Embedding = {'NNEemb':nn.Embedding,'Token':TokenEmbedding}
        self._embeddings = nn.ModuleList([
            ModalembProj(
                [ Embedding[EmbType](**EmbArgs) ],
                **Others ) 
            for EmbType,EmbArgs,Others in cfg.embconfig
        ])


        self_encoderCrosses = [[[
                    CrossLayer(**cfg.SelfAttParameters) for i in range( numcross) ]
                    for i in range(cfg.numfeats)
            ]]*2
        self_encoderConv1Ds = [[[
                    ConvPoolLayer(**cfg.SelfConvParameters) for i in range( numconv1) ]
                    for i in range(cfg.numfeats)
                ]]*2
        cross_encoderCrosses = [[
                    CrossLayer(**cfg.CrosAttParameters) for i in range( numcross) ]
                    for i in range(cfg.totalCross)
                ]
        cross_encoderConv1Ds = [[
                    ConvPoolLayer(**cfg.CrosConvParameters) for i in range( numconv1) ]
                    for i in range(cfg.totalCross)
                ]
        self.PackSelfAttentions = AttEncoders(
                iscross=False,
                d_model=cfg.d_model,
                numselfs=cfg.numfeats,
                encoderCrosses=self_encoderCrosses[0],
                encoderConv1Ds=self_encoderConv1Ds[0],
                numcross=numcross,
                numconv1=numconv1
        )    
        self.PackCrosAttentions = AttEncoders(
                iscross=True,
                d_model=cfg.d_model,
                numselfs=cfg.numfeats,
                encoderCrosses=self_encoderCrosses,
                encoderConv1Ds=self_encoderConv1Ds,
                numcross=numcross,
                numconv1=numconv1
        )
        self.PackSelfAttentions_2 = AttEncoders(
                iscross=False,
                d_model=cfg.d_model,
                numselfs=cfg.numfeats,
                encoderCrosses=self_encoderCrosses[1],
                encoderConv1Ds=self_encoderConv1Ds[1],
                numcross=numcross,
                numconv1=numconv1
        ) 

        self.PackCrosCombs = nn.ModuleList([
            ModalConbin(len(comb)) for comb in cfg.combconfig
        ])

        self.FinalComb = ModalConbin(cfg.finalComb)
        self.FinalNorm = nn.BatchNorm1d(cfg.d_model * 3)
        self.OutProj = nn.Sequential(
                    nn.Linear( (cfg.d_model*3),256 ),
                    nn.ReLU(),
                    nn.Dropout(cfg.global_dropout),
                    nn.Linear( 256,numclass )
                    )
    def nanstd(self,o,dim):
        return torch.sqrt(
                torch.nanmean(
                    torch.pow( torch.abs(o-torch.nanmean(o,dim=1).unsqueeze(1)),2),
                    dim=dim)
                )
    def _pooling(self,x):
        x_std = torch.std(x,dim=1)
        x_mean = torch.nanmean(x, dim=1)
        x_max = torch.max(x,dim=1).values
        score  = torch.cat([x_std, x_mean,x_max], dim=1)
        return score

    def forward(self,x_nums,x_cates):
        numshape,cateshape = x_nums.shape[-1],x_cates.shape[-1]
        if len(self._embeddings)!= numshape+cateshape:
            raise ValueError(f'''Number Embedding Layer not equal to feature dims
                for numfeats:{numshape+cateshape}''')

        data = [ ]
        for n,Embedding in enumerate(self._embeddings):
            if n<numshape:
                e = Embedding(x_nums[:,:,n])
            else:
                e = Embedding(x_cates[:,:,n-numshape])
            data.append(e)
        data = self.PackSelfAttentions(data)
        Combs = []
        for CrosCombLayer,comb in zip(self.PackCrosCombs,self._cfg.combconfig)
            Combs.append( CrosCombLayer( [ data[idx] for idx in comb ] ) )
        Combs = self.PackCrosAttentions( Combs + [ data[idx] for idx in self._cfg.NoneCombidx  ] )
        data  = self.PackSelfAttentions_2(data)
        o = self.FinalComb( data+Combs )
        o = self.FinalNorm(self._pooling(o))
        o = self.OutProj(o)
        return  F.sigmoid(o)

