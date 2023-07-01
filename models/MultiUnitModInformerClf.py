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

class CateAtte(nn.Module):
    def __init__(
        self, 
        d_model,
        norm_dim,
        n_heads=6,
        factor=35,#top 245|from 512-1024
        dropout=0.1, 
        attType = 'full',
        ):

        super(CateAtte, self).__init__()
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
        self.norm1 = nn.LayerNorm(norm_dim)
    def forward(self, x, x_mask=None, cross_mask=None):
        # namask = torch.isnan(x)
        x = x.permute(0,2,1)
        x = x * self.dropout(
                self._crossLayer(
                    x, x, x,
                    attn_mask=cross_mask )[0]
            )
        return self.norm1(x.transpose(1,2))


class EncoderBlock(nn.Module):
    """docstring for SelfAttBlock"""
    def __init__(
        self, 
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
        self._crossList = nn.ModuleList(crosslayers)
        self.last_extro_norm = nn.LayerNorm(d_model)
        if conv1layers is None:
            self._conv1List = [None]*len( self._crossList )
        else:
            self._conv1List = nn.ModuleList(conv1layers)
            for i in range(numcross-numconv1):
                self._conv1List.append(None)

    def forward(self,x,cross):
        for encoder,conv1D in zip(self._crossList, self._conv1List ):
            x = encoder(x,cross)
            if conv1D is not None:
                x = conv1D(x)
        return self.last_extro_norm(x)

class AttEncoders(nn.Module):
    def __init__(
        self,
        iscross,
        d_model,
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
                d_model = d_model,
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
                    # nn.(8, 32, 1)
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

class CateAwareClf(nn.Module):
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
        super(CateAwareClf, self).__init__()

        self._cfg = cfg

        # Embedding = {'NNEemb':nn.Embedding,'Token':TokenEmbLinear}
        #################CATE NET########################
        self._c1_embeddings = nn.ModuleList([
            nn.Embedding(**EmbArgs) for EmbArgs in cfg.cate_emb_cfgs
        ])
        
        self._c2_embeddings = nn.ModuleList([
            nn.Embedding(**EmbArgs) for EmbArgs in cfg.cate_emb_cfgs
        ])

        # embedding interaction
        self.categorical_att =  nn.ModuleList([
            CateAtte(
                d_model = seq_len,
                norm_dim = cfg.emb_dim_all,
                n_heads = 8,
                factor  = 60,
                dropout = 0.2,
                attType = 'prob'
            ) for seq_len in [cfg.long_term,cfg.short_term]
        ])


        self.cates_proj =  nn.ModuleList([
            PosAndProject(
                embedim = cfg.emb_dim_all ,#seperate proj
                d_model = cfg.emb_dim_all ,
                max_len = seq_len,
                kernel_size = cfg.glb_kernel_size ,
                DEBUG = False
            ) for seq_len in [cfg.long_term,cfg.long_term,cfg.short_term,cfg.short_term]
            #long term num,long term cate,short term num,short term cate,
        ])
      
        # self.cates_proj = PosAndProject(
        #         embedim = cfg.emb_dim_all,
        #         d_model = cfg.emb_dim_all ,
        #         max_len = cfg.seq_len ,
        #         kernel_size = cfg.glb_kernel_size ,
        #         DEBUG = False
        #     )

        L_crosslayers = [CrossLayer(**cfg.CatesAttPara) for i in range( numcross) ]
        L_conv1layers = [ConvPoolLayer(**cfg.CatesConvPara) for i in range( numconv1) ]
        S_crosslayers = [CrossLayer(**cfg.CatesAttPara) for i in range( numcross) ]
        S_conv1layers = [ConvPoolLayer(**cfg.CatesConvPara) for i in range( numconv1) ]
        self.L_self_att = EncoderBlock(
                    crosslayers = L_crosslayers,
                    conv1layers = L_conv1layers,
                    d_model = cfg.CatesAttPara['d_model'],
                    numcross = numcross,
                    numconv1 = numconv1
            )
        
        self.S_self_att = EncoderBlock(
                    crosslayers = S_crosslayers,
                    conv1layers = S_conv1layers,
                    d_model = cfg.CatesAttPara['d_model'],
                    numcross = numcross,
                    numconv1 = numconv1
            )

        # selfcate_enc_att = [
        #             [CrossLayer(**cfg.CatesAttPara) for i in range( numcross) ]
        #     ]
        # selfcate_enc_conv1D = [[
        #             ConvPoolLayer(**cfg.CatesConvPara) for i in range( numconv1) ]
        #         ]
        # self.CateSelfAttentions = AttEncoders(
        #         iscross = False,
        #         d_model = cfg.CatesAttPara['d_model'],#cate_cnt*d_model
        #         encoderCrosses = selfcate_enc_att,
        #         encoderConv1Ds = selfcate_enc_conv1D,
        #         numcross = numcross,
        #         numconv1 = numconv1
        #     )

        # #################NUMER NET#######################
        self.L_n_embeddings = nn.ModuleList([
            TokenEmbLinear(**EmbArgs)
            for EmbArgs,Others in cfg.num_emb_cfgs
        ])
        self.S_n_embeddings = nn.ModuleList([
            TokenEmbLinear(**EmbArgs)
            for EmbArgs,Others in cfg.num_emb_cfgs
        ])

        # self_enc_att = [
        #             [CrossLayer(**cfg.NumerAttPara) for i in range( numcross) ]
        #             for i in range(cfg.numer_feat_cnt)
        #     ]
        # self_enc_conv1D = [[
        #             ConvPoolLayer(**cfg.NumerConvPara) for i in range( numconv1) ]
        #             for i in range(cfg.numer_feat_cnt)
        #         ]

        # self.NumerSelfAttentions = AttEncoders(
        #         iscross = False,
        #         d_model = cfg.d_model,
        #         encoderCrosses = self_enc_att,
        #         encoderConv1Ds = self_enc_conv1D,
        #         numcross = numcross,
        #         numconv1 = numconv1
        #     )

        # finalCross = [[CrossLayer(**cfg.SelfAttParameters) for i in range( numcross)]]
        # finalCov1D = [[ConvPoolLayer(**cfg.SelfConvParameters) for i in range( numconv1)]]

        # self.FianlAtt = AttEncoders(
        #         iscross = False,
        #         d_model = cfg.d_model,
        #         encoderCrosses = finalCross,
        #         encoderConv1Ds = finalCov1D,
        #         numcross = numcross,
        #         numconv1 = numconv1
        #      )

        # finalDimBase = cfg.numer_feat_cnt + cfg.cate_feat_cnt
        finalDimBase = cfg.emb_dim_all*2*2
        # finalDimBase = cfg.emb_dim_all
        print(f'[Norm Shape]\n[=================]FinalNorm:{ 3*finalDimBase}')
        
        self.FinalNorm = nn.BatchNorm1d(finalDimBase*3)
        self.OutProj = nn.Sequential(
                    # nn.Linear((cfg.d_model*12), 256),
                    nn.Conv1d(finalDimBase*3, 256, 1),
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
#X,  [X_CAT_0, X_CAT_1, X_CAT_2], POS, COR, POS2,ROOM_FQ
    def forward(
        self,
        x_nums,
        x_cates,
        # x_pos,
        # x_cor,
        # x_cor_pos,
        # room_od
    ):###

        L_data = [ ]
        S_data = [ ]
        for L_emb,S_emb,L_o,S_o in zip(
                        self._c1_embeddings,
                        self._c2_embeddings,
                        x_cates[0],
                        x_cates[1]
                        ):
            # print(f'''DEBUG CATE [Input shape]
            # [======================]CATE EMBEDDINGS:{o.shape}''')
            L_o = L_emb(L_o)
            S_o = S_emb(S_o)
            # print(f'''DEBUG CATE [OutPut shape]
            # [======================]CATE EMBEDDINGS:{o.shape}''')
            L_data.append(L_o)
            S_data.append(L_o)
        for i in range(len(self.L_n_embeddings)):
            L_NE = self.L_n_embeddings[i]( x_nums[0][:,:,i] )
            S_NE = self.S_n_embeddings[i]( x_nums[1][:,:,i] )
            L_data.append(L_NE)
            S_data.append(S_NE)

            # print(f'''DEBUG Numer [Input shape]
            # [======================]Numer EMBEDDINGS:{x_nums[:,:,i].shape}''')
            # print(f'''DEBUG Numer [Input shape]
            # [======================]Numer EMBEDDINGS:{ebed.shape}''')
            # data.append(ebed)
        L_data = torch.cat(L_data,dim=-1)
        S_data = torch.cat(S_data,dim=-1)
        # print(f'''DEBUG CATE [Input shape]
        #     [======================]CAT CATE EMBEDDINGS:{data.shape}''')
        #compare no categorical
        L_o_att = self.categorical_att[0](L_data)
        S_o_att = self.categorical_att[1](S_data)
 
        # print(f'''DEBUG CATE [OutPut shape]
        #     [======================]ATT CATE EMBEDDINGS:{data.shape}''')
        L_data = self.cates_proj[0](L_data)
        L_o_att = self.cates_proj[1](L_o_att)
        S_data = self.cates_proj[2](S_data)
        S_o_att = self.cates_proj[3](S_o_att)
        # print(f'''DEBUG CATE [OutPut shape]
        #     [======================]PROJ CATE-ATT:{data.shape}''')
        L_data = torch.cat([L_data,L_o_att],dim=-1)#attention together
        S_data = torch.cat([S_data,S_o_att],dim=-1)#attention together
        L_data = self.L_self_att(L_data, L_data)
        S_data = self.S_self_att(S_data, S_data)

        L_data = torch.cat([self._pooling(L_data),self._pooling(S_data)],dim=1)

        L_data = self.FinalNorm(L_data)
        L_data = self.OutProj(L_data.unsqueeze(-1)).squeeze(-1)
        return F.sigmoid(L_data)