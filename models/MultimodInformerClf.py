from models.attn import FullAttention, ProbAttention, AttentionLayer
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
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # namask = torch.isnan(x)
        x = x + self.dropout(
                    self._crossLayer(
                        x, cross, cross,
                        attn_mask=cross_mask
                    )[0])
                    
        y = x = self.norm1(x)
            # .masked_fill(namask,0)).masked_fill(namask,torch.nan)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm2( (x+y))
            # .masked_fill(namask,0) ).masked_fill(namask,torch.nan)

class CrossBlock(nn.Module):
    """docstring for CrossBlock"""
    def __init__(
        self, 
        crosslayers=None,
        conv1layers=None,
        numcross:int=3,
        numconv1:int=2,
        ):
        super(CrossBlock, self).__init__()
        if numcross<numconv1+1:
            raise ValueError(f'''invalid parameter numconv1:{numconv1};
                        ->numcross:{numcross} should greater than numconv1 + 1''')
        self._numcross = numcross
        self._numconv1 = numconv1
        self._crossList = nn.ModuleList(crosslayers)
        NumOfCross  = len( crosslayers )
        NumOfConv1D = len( conv1layers )
        if conv1layers is None:
            self._conv1List = [None]*len( self._crossList )
        else:
            self._conv1List = nn.ModuleList(conv1layers)
            for i in range(NumOfCross-NumOfConv1D):
                self._conv1List.append(None)

    def forward(self,y,z):

        iter_cross = iter(self._crossList)
        iter_conv1d = iter(self._conv1List)

        for _ in range(self._numcross):
            y1 = next(iter_cross)(y,z)
            z1 = next(iter_cross)(z,y)
            conv1dA = next(iter_conv1d)
            conv1dB = next(iter_conv1d)
            if (
              conv1dA is not None
              and conv1dB is not None
            ):
                y1 = conv1dA(y1)
                z1 = conv1dB(z1)
            y = y1
            z = z1
        return y1,z1
            
class SelfAttBlock(nn.Module):
    """docstring for SelfAttBlock"""
    def __init__(
        self, 
        crosslayers=None,
        conv1layers=None,
        numcross:int=3,
        numconv1:int=2,
        ):
        super(SelfAttBlock, self).__init__()
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

    def forward(self,x):
        for encoder,conv1D in zip(self._crossList,self._conv1List ):
            x = encoder(x,x)
            if conv1D is not None:
                x = conv1D(x)
        return x



class MultiMInformerClf(nn.Module):
    '''
    Parameters:
        numcross:denotes the number of cross attention layers
           numcross * 3 ,3 is the number of modal parts
        numconv1:denotes the number of layers of conv1D with maxpooling etc.
        cfg: global configuration of parameters
    config example:
        crossargs[dict]:attr of CFG class for CrossLayer para. examples
            {'d_model':256, 
            'd_ff':512,
            'n_heads':4,
            'factor':5,
            'dropout':0.1, 
            'activation':"relu"}
        convargs[dict]:attr of CFG class for ConvLayer para. examples
            {"d_model":256,"kernel_siz":5}

        probAtt = {
         "mask_flag":False, 
         "factor":5, 
         "attention_dropout":0.2, 
         "output_attention":False
        }
        attLayer ={
            "d_model":d_model, 
            "n_heads":n_heads, 
            "mix":True
        }
        encoderLayer = {
        'd_model':d_model, 
        'd_ff':d_model*2, 
        'dropout':dropout,
        'activation':activation
        }
        numSelfAttLayer
        num_class
        modalX.token = {
            c_in=token_dim, d_model=embedim
        }
        modalX.Others = {
                embedim,
                d_model,
                max_len,
                kernel_size=5
        }
        modalY.cates = {
            list_vocab_sizes = list_vocab_sizes,
           list_embed_dims = list_embed_dims,
           tot_cat_emb_dim  = embedim,
        }
        modalY.Others = {
                embedim,
                d_model,
                max_len,
                kernel_size=5
        }
        modalZ.cates = {
            list_vocab_sizes = list_vocab_sizes,
           list_embed_dims = list_embed_dims,
           tot_cat_emb_dim  = embedim,
        }
        modalZ.Others = {
                embedim,
                d_model,
                max_len,
                kernel_size=5
        }
        global_dropout

    '''
    def __init__(
        self,
        numcross,
        numconv1,
        numclass,
        cfg
        ):
        super(MultiMInformerClf, self).__init__()
        

        self._embeddingX = ModalembProj(
            [TokenEmbedding( **cfg.modalX.token)],
            **cfg.modalX.Others
        ) 
        self._embeddingY = ModalembProj(
            [CatesEmbedding(**cfg.modalY.cates)],
            **cfg.modalY.Others
        ) 
        self._embeddingZ = ModalembProj(
            [CatesEmbedding(**cfg.modalZ.cates),
             TokenEmbedding(**cfg.modalZ.token1),
             TokenEmbedding(**cfg.modalZ.token2)],
            **cfg.modalZ.Others
        ) #V1
        # self._embeddingX = ModalembProj(
        #     [TokenEmbedding( **cfg.modalX.token)],
        #         **cfg.modalX.Others
        #     ) 
        # self._embeddingY = ModalembProj(
        #     [nn.Embedding(**args) for args in cfg.modalY.cates],
        #         **cfg.modalY.Others
        #     ) 
        # self._embeddingZ = ModalembProj(
        #     [
        #     nn.Embedding(**args) for args in cfg.modalZ.cates
        #     ] + [
        #     TokenEmbedding(**args) for args in cfg.modalZ.tokens
        #     ],
        #         **cfg.modalZ.Others
        #     ) #v2

        # self._crossModalBlock = CrossBlock(
        #         # crossargs:check out examples in paras
        #         # convargs:check out examples in paras
        #         [CrossLayer(**cfg.crossargs) for i in range( numcross*6 ) ],
        #         [ConvPoolLayer(**cfg.convargs) for i in range( numconv1*3 )],
        #         numcross,
        #         numconv1
        #     ) #v2
        # self._crossModalBlock = CrossBlock(
        #         # crossargs:check out examples in paras
        #         # convargs:check out examples in paras
        #         [CrossLayer(**cfg.crossargs) for i in range( numcross*2) ],
        #         [ConvPoolLayer(**cfg.convargs) for i in range( numconv1*2 )],
        #         numcross,
        #         numconv1
        #     ) #prev
        self._crossBlocks = nn.ModuleList([
            CrossBlock(
                [CrossLayer(**cfg.crossargs) for i in range( numcross*2) ],
                [ConvPoolLayer(**cfg.convargs) for i in range( numconv1*2 )],
                numcross,
                numconv1) for i in range(3)
            ])
        SelfAttParameters = cfg.crossargs
        SelfAttParameters['attType'] = 'prob'
        self._selfAttentions = nn.ModuleList([
            SelfAttBlock(
                [CrossLayer(**SelfAttParameters) for i in range( numcross) ],
                [ConvPoolLayer(**cfg.convargs) for i in range( numconv1 )],
                numcross,
                numconv1)   for i in range(6)
            ])
        

        # encoders = [Encoder(
        #                 [EncoderLayer( 
        #                     AttentionLayer(
        #                         ProbAttention(**cfg.SelfATT.probAtt),
        #                             **ARGattLayer
        #                     ),**ARGencoderLayer
        #                     )
        #                  for o in range(cfg.SelfATT.numSelfAttLayer)  ],
        #                 [ConvLayer(dm,cfg.SelfATT.kernel_size) 
        #                  for l in range(cfg.SelfATT.numSelfAttLayer-1) ] if cfg.SelfATT.distil else None,
        #              norm_layer=torch.nn.LayerNorm(dm) 
        #              ) for ARGattLayer,ARGencoderLayer,dm in zip(cfg.SelfATT.attLyrArgs,
        #                                                          cfg.SelfATT.enclyrArgs,
        #                                                         [cfg.d_model]*6
        #                                                          )
        # # for i in range(3)
        # ]
        self._normAndAct0 =  nn.Sequential(
            nn.BatchNorm1d(cfg.d_model * 6),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        self._normAndAct1 =  nn.Sequential(
            nn.BatchNorm1d(cfg.d_model *6),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        # self.dropout = nn.Dropout(0.2)
        # self.conv1 = nn.Conv1d(in_channels=cfg.d_model*6, out_channels=cfg.d_model*24, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=cfg.d_model*24, out_channels=cfg.d_model*6, kernel_size=1)
        # self.norm1 = nn.LayerNorm(cfg.d_model*6)
        # self.activation = F.relu 
        
        
        self._crossConv1Ds = nn.ModuleList( ConvPoolLayer(**cfg.convargs) for i in range(3))
        self._modalWeithProj = nn.Sequential(
                nn.Linear(6,32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32,32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32,6),
                nn.Linear(6,1),
                nn.ReLU()
            )

        self._projection = nn.Sequential(
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
        # print(f'DEBUG value [x]\n==========={x}')
        # print(f'DEBUG shape [x]\n==========={x.shape}')
        
        # x_std = self.nanstd(x,1)
        x_std = torch.std(x,dim=1)
        x_mean = torch.nanmean(x, dim=1)
        # x_max = torch.max(o.masked_fill( torch.isnan(x),-torch.inf ),dim=1).values
        x_max = torch.max(x,dim=1).values
        score  = torch.cat([x_std, x_mean,x_max], dim=1)
        return score

    def forward(self,x,y,z):
        x = self._embeddingX(x) #x
        y = self._embeddingY(y) #x_cat
        z = self._embeddingZ(z) #x_extro
        x = self._selfAttentions[0]( x )#[0]
        y = self._selfAttentions[1]( y )#[0]
        z = self._selfAttentions[2]( z )#[0]
        # y1,z1 = self._crossModalBlock(y,z)
        y1,z1 = self._crossBlocks[0](y,z)
        x1,y1 = self._crossBlocks[1](x,y)
        x1,z1 = self._crossBlocks[2](x,z)
        
        x1 = self._crossConv1Ds[0](x1)
        y1 = self._crossConv1Ds[1](y1)
        # print('DEBUG value [conv1d]')
        # print(y1)
        z1 = self._crossConv1Ds[2](z1)
        # self._crossBlocks
        # y1,z1 = self._crossModalBlock(y,z)
        x = self._selfAttentions[3]( x )#[0]
        y = self._selfAttentions[4]( y )#[0]
        z = self._selfAttentions[5]( z )#[0]

        o = torch.cat([
            x.reshape( ( x.shape[0],-1) ).unsqueeze(-1),
            y.reshape( ( x.shape[0],-1) ).unsqueeze(-1),
            z.reshape( ( x.shape[0],-1) ).unsqueeze(-1),
            x1.reshape( ( x.shape[0],-1) ).unsqueeze(-1),
            y1.reshape( ( x.shape[0],-1) ).unsqueeze(-1),
            z1.reshape( ( x.shape[0],-1) ).unsqueeze(-1)
            ],dim=-1)#6

        o = self._modalWeithProj(o).reshape( x.shape )
        o = self._pooling(o)
        o = self._projection(o)
        return  F.sigmoid(o)

        
        # y = o
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        # y = self.dropout(self.conv2(y).transpose(-1,1))
        # o = self.norm1(o+y)
        # o = self._pooling(o)
        # o = self._projection(o)
        # return  F.sigmoid(o)






        # o = torch.cat([
        #     # self._pooling(x),
        #     self._pooling(y),
        #     self._pooling(z)], 
        #     dim=1) #dim*6
        # y,z = self._crossModalBlock(y,z)#
        # o = self._normAndAct0(o)
        # o = o + torch.cat([
        #     # self._pooling(x),
        #     self._pooling(y),
        #     self._pooling(z)], 
        #     dim=1)
        # o = self._normAndAct1(o)

        # # o =  #bs seq dim
        # # c = #bs seq dim*3
        # # x = self._selfAttentions[0](x)[0]
        # # y = self._selfAttentions[1](y)[0]
        # # z = self._selfAttentions[2](z)[0]
        # o = torch.cat([o,
        #     self._pooling(self._selfAttentions[0](y+z )[0]),
        #     self._pooling(self._selfAttentions[1](torch.cat([ y,z],dim=-1 ))[0])], 
        #     dim=1) #3*2+3+3*2=15 dim
        # # _out = torch.cat(
        # #     [self._pooling(x),self._pooling(y), self._pooling(z)], 
        # #     dim=1)
        # o = self._projection(o)
        # return  F.sigmoid(o)

