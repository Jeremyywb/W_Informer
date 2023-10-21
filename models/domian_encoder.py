from models.attn import FullAttention, ProbAttention, AttentionLayer
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from models.embed import CatesEmbedding,TokenEmbLinear,ModalembProj
from models.embed import PositionalEmbedding,TokenEmbedding,PosAndProject
from models.encoder import ConvLayer,ConvPoolLayer,Encoder,EncoderLayer
import torch.nn.functional as F
from itertools import islice
import torch.nn as nn
import torch


class CrossLayer(nn.Module):
    def __init__(
        self,
        mask_flag,
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
                ATT(mask_flag,factor, 
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
        mask_flag,
        crosslayers=None,
        conv1layers=None,
        d_model=512,
        numcross:int=3,
        numconv1:int=2,
        ):
        super(EncoderBlock, self).__init__()
        if numcross<numconv1+1:
            raise ValueError(f'''invalid parameter numconv1:{numconv1};
                        ->numcross:{numcross} should greater than numconv1 + 1''')
        self._numcross = numcross
        self._numconv1 = numconv1
        self.mask_flag = mask_flag
        self._crossList = nn.ModuleList(crosslayers)
        self.last_extro_norm = nn.LayerNorm(d_model)
        if conv1layers is None:
            self._conv1List = [None]*len( self._crossList )
        else:
            self._conv1List = nn.ModuleList(conv1layers)
            for i in range(numcross-numconv1):
                self._conv1List.append(None)

    def forward(self,x,cross, cross_mask=None):
        for encoder,conv1D in zip(self._crossList, self._conv1List ):
            if self.mask_flag:
                x = encoder(x, cross, cross_mask = cross_mask)
            else:
                x = encoder(x, cross)
            if conv1D is not None:
                x = conv1D(x)
        return self.last_extro_norm(x)




class PromptAwareEncoder(nn.Module):
    '''
    Parameters:
        numcross:denotes the number of cross attention layers
           numcross * 3 ,3 is the number of modal parts
        numconv1:denotes the number of layers of conv1D with maxpooling etc.
        cfg: global configuration of parameters
    Inputs:
        x: texts embeddings
        o: domian paragraph embeddings

    '''
    def __init__(
        self,
        mask_flag,
        numcross,
        numconv1,
        d_model,
        attParameter,
        downConvPara,
        ):
        super(PromptAwareEncoder, self).__init__()
        self.mask_flag = mask_flag
        _attlayers = [CrossLayer(**attParameter) for i in range( numcross) ]
        _downconvs = [ConvPoolLayer(**downConvPara) for i in range( numconv1) ]
        self.encoder = EncoderBlock(
                    mask_flag = mask_flag,
                    crosslayers = _attlayers,
                    conv1layers = _downconvs,
                    d_model = d_model,
                    numcross = numcross,
                    numconv1 = numconv1
            )
        
    def forward(
        self,
        x,
        o,
        cross_mask = None
    ):

        return self.encoder(x,o,cross_mask)

