from models.attn import FullAttention, ProbAttention, AttentionLayer
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from models.embed import CatesEmbedding,TokenEmbLinear,ModalembProj
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
        # self.fc1 = nn.Linear(d_model, d_ff)
        # self.fc2 = nn.Linear(d_ff, d_model)
        self.fc1 = nn.Conv1d(d_model, d_ff, 1)
        self.fc2 = nn.Conv1d(d_ff, d_model, 1)
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
        y = self.dropout(self.activation(self.fc1(y.permute(0, 2, 1))))
        y = self.dropout(self.fc2(y).transpose(1, 2))
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
        numconv1:int=2
    ):
        super(AttEncoders, self).__init__()
        self.iscross = iscross
        self._encoders = nn.ModuleList([
            EncoderBlock(
                crosslayers=crosslayers,
                conv1layers=conv1layers,
                numcross=numcross,
                numconv1=numconv1,
                )
            for crosslayers,conv1layers in zip(encoderCrosses,encoderConv1Ds)
        ])
        self._layerNorms1 = nn.ModuleList([
            nn.LayerNorm(d_model) for i in range(len( encoderCrosses ))
        ])
        # self._layerNorms2 = nn.ModuleList([
        #     nn.LayerNorm(d_model) for i in range(len( encoderCrosses ))
        # ])
        self._ffns = nn.ModuleList([
                nn.Sequential(
                    nn.(8, 32, 1)
                    nn.Conv1d(d_model*3,4*d_model, 1),
                    # nn.Linear(d_model*3,4*d_model),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(4*d_model,d_model, 1),
                    # nn.Linear(4*d_model,d_model),
                    nn.Dropout(0.1)
                ) for i in range(len( encoderCrosses ))
            ])


    def forward(self,xlist):
        xencodes = [ ]
        if self.iscross:
            iter_encoders   = iter(self._encoders)
            iter_layernorms1 = iter(self._layerNorms1)
            # iter_layernorms2 = iter(self._layerNorms2)
            iter_ffns = iter(self._ffns)
            mainX = xlist[0]
            for idx, x in enumerate( xlist[1:] ):
                block = next(iter_encoders )
                ffn  = next( iter_ffns )
                norm1  = next( iter_layernorms1 )
                if idx==0:
                    y = block(mainX, x)
                else:
                    y = torch.cat([y, block(mainX, x)], dim=-1)
            y = norm1(ffn(y.permute(0,2,1)).transpose(1,2) )
            return y
            # for idx,x in enumerate( xlist):
            #     indices = [_id for _id in range(len(xlist)) if _id != idx]
            #     for num,restid in enumerate( indices):
            #         block = next( iter_encoders )
            #         norm1  = next( iter_layernorms1 )
            #         norm2  = next( iter_layernorms2 )
            #         ffn  = next( iter_ffns )
            #         if num ==0:
            #             y = block(x, xlist[restid] )
            #         else:
            #             y = norm1( y + block( x, xlist[restid] ))
            #             y = norm2( y + ffn(y) )
            #     xencodes.append(y)
        else:
            for x, block, norm in zip(xlist, self._encoders, self._layerNorms1):
                x = block(x, x)
                x = norm(x)
                xencodes.append(x)
            return xencodes


class ModalConbin(nn.Module):
    def __init__(
        self,
        numcomb,
        d_model
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
        self._ffn = nn.Sequential(
                # nn.Linear(d_model,4*d_model),
                nn.Conv1d(d_model, 4*d_model, 1),
                nn.ELU(),
                nn.Dropout(0.1),
                # nn.Linear(4*d_model,d_model),
                nn.Conv1d(4*d_model, d_model, 1),
                nn.Dropout(0.1)
            )
        
        self._layerNorm = nn.LayerNorm(d_model)
        
    def forward(self,combs:List[torch.Tensor]):
        o = torch.cat([
            x.reshape( ( x.shape[0],-1) ).unsqueeze(-1)
            for x in combs
            ],
            dim=-1
        )
        o = self._project(o).reshape(combs[0].shape)
        o = self._layerNorm(o + self._ffn(o.permute(0,2,1)).transpose(1,2) )
        return  o

class DiaConv1D(nn.Module):
    def __init__(
        self,
        # dilation,
        kernel_size,
        d_model
    ):
        super(DiaConv1D, self).__init__()
        parameters = {
            'in_channels':d_model ,
            'out_channels': d_model,
            'kernel_size': kernel_size,
            'padding': 1,
            'dilation':1,
            'padding_mode':'circular'
        }
        self.net = nn.Sequential(
            nn.Conv1d(**parameters ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self,x):
        x = self.net(x.permute(0,2,1)).transpose(1,2)
        return x

class PackDiaConv1D(nn.Module):
    def __init__(
        self,
        dilations,
        d_model
    ):
        super(PackDiaConv1D, self).__init__()
        self.nets = nn.ModuleList(
            [DiaConv1D(dia,d_model) for dia in dilations]
        )
    def forward(self,x):
        for diaConv1D in self.nets:
            x = diaConv1D(x)
        return x

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
        Embedding = {'NNEemb':nn.Embedding,'Token':TokenEmbLinear}
        self._embeddings = nn.ModuleList([
            ModalembProj(
                [ Embedding[EmbType](**EmbArgs) ],
                **Others ) 
            for EmbType,EmbArgs,Others in cfg.embconfig
        ])
        self.EmbStepComb = nn.Sequential(
                # nn.Linear(cfg.d_model*2, 4*cfg.d_model),
                nn.Conv1d(cfg.d_model*2, 4*cfg.d_model,1),
                nn.ELU(),
                nn.Dropout(0.1),
                # nn.Linear(4*cfg.d_model, cfg.d_model),
                nn.Conv1d(4*cfg.d_model, cfg.d_model, 1),
                nn.Dropout(0.1)
                
            )
        self.EmbCombNorm = nn.LayerNorm(cfg.d_model)

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
        self.PackCrosAttentions = nn.ModuleList([
            AttEncoders(
                iscross=True,
                d_model=cfg.d_model,
                numselfs=cfg.numfeats,
                encoderCrosses=cross_encoderCrosses,
                encoderConv1Ds=cross_encoderConv1Ds,
                numcross=numcross,
                numconv1=numconv1
                ) for i in range(len(cfg.CrossPacks))
        ])

        self._PackDiaConv1D = nn.ModuleList(
          [PackDiaConv1D(cfg.dias, cfg.d_model) for i in range( 4)]
        ) 


        # self.PackSelfAttentions_2 = AttEncoders(
        #         iscross=False,
        #         d_model=cfg.d_model,
        #         numselfs=cfg.numfeats,
        #         encoderCrosses=self_encoderCrosses[1],#on how many used final
        #         encoderConv1Ds=self_encoderConv1Ds[1],
        #         numcross=numcross,
        #         numconv1=numconv1
        # ) 

        # self.PackCrosCombs = nn.ModuleList([
        #     ModalConbin(len(comb)) for comb in cfg.combconfig
        # ])

        # self.FinalComb = ModalConbin(cfg.finalComb)
        self.FinalNorm = nn.BatchNorm1d(cfg.d_model * 12)
        self.OutProj = nn.Sequential(
                    # nn.Linear((cfg.d_model*12), 256),
                    nn.Conv1d((cfg.d_model*12), 256, 1),
                    nn.ReLU(),
                    nn.Dropout(cfg.global_dropout),
                    # nn.Linear(256, numclass)
                    nn.Conv1d(256, numclass, 1)
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
        for n, Embedding in enumerate(self._embeddings):
            if n < numshape:
                e = Embedding(x_nums[:, :, n]) # 256|512/2
            else:
                e = Embedding(x_cates[:, :, n-numshape])
            data.append(e)

        if self._cfg.CombMode == 'embedding_step':
            embstep_comb = []
            for comb in self._cfg.combconfig:
                o = torch.cat([data[idx] for idx in comb], dim=-1)
                o = self.EmbCombNorm(
                        self.EmbStepComb(
                            o.permute(0,2,1)
                            ).transpose(1,2)
                        )
                embstep_comb.append(o)
            data = [data[c] for c in self._cfg.NoneCombidx]
            data.extend(embstep_comb)
        data = self.PackSelfAttentions(data)# 64|256/4
        Combs = []
        for comb, encoder in zip(self._cfg.CrossPacks, self.PackCrosAttentions):
            Combs.append(encoder([data[c] for c in comb]))
        usedCrosId = [c[0] for c in self._cfg.CrossPacks]
        for i, o in zip(usedCrosId, Combs):
            data[i] = o
        data = [diaConv1D(o) for diaConv1D,o in zip(self._PackDiaConv1D, data)]
        # data = [conv(o) for conv, o in zip(self.PackConvLayer, data)] #5 for pack|4
        data = torch.cat([self._pooling(o) for o in data], dim=1)#3*5 |3*4
        data = self.FinalNorm(data)
        data = self.OutProj(data.unsqueeze(-1)).squeeze(-1)#conv1d proj
        return F.sigmoid(data)
        


