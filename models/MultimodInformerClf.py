from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import CatesEmbedding,TokenEmbedding,ModalembProj
from models.encoder import ConvLayer,ConvPoolLayer,Encoder,EncoderLayer
import torch.nn.functional as F
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
                                ATT(
                                    False, 
                                    factor, 
                                    attention_dropout=dropout, 
                                    output_attention=False
                                    ), 
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
        x = x + self.dropout(
                    self._crossLayer(
                        x, cross, cross,
                        attn_mask=cross_mask
                    )[0])
                    
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm2(x+y)

class CrossBlock(nn.Module):
    """docstring for CrossBlock"""
    def __init__(
        self, 
        crosslayers,
        conv1layers,
        numcross,
        numconv1,
        ):
        super(CrossBlock, self).__init__()
        self._numcross = numcross
        self._numconv1 = numconv1
        self._crossList = nn.ModuleList(crosslayers)
        self._conv1List = nn.ModuleList(conv1layers)

    # def _crossAtt(self,conv1s,crosses,x,y,z,hasConv):
    def _crossAtt(self,conv1s,crosses,y,z,hasConv):
        if hasConv:
            # x1 = conv1s[0](
            #             crosses[0](x,y) + crosses[1](x,z)
            #         )
            # y1 = conv1s[1](
            #             crosses[2](y,x) + crosses[3](y,z)
            #         )
            # z1 = conv1s[2](
            #             crosses[4](z,x) + crosses[5](z,y)
            #         )
            y1 = conv1s[0](crosses[0](y,z))
            z1 = conv1s[1](crosses[1](z,y))
        else:
            # x1 = crosses[0](x,y) + crosses[1](x,z)
            # y1 = crosses[2](y,x) + crosses[3](y,z)
            # z1 = crosses[4](z,x) + crosses[5](z,y)         
            x1 = crosses[0](y,z)
            z1 = crosses[1](z,y)
        return y1,z1

        # return x1,y1,z1

    def forward(
        self,
        # x, 
        y,
        z
        ):
        for layer in range(self._numconv1):
            # x,y,z = self._crossAtt( 
            #             self._conv1List[layer*3:layer*3+3],
            #             self._crossList[layer*3:layer*3+6],
            #             x,y,z,
            #             True
            #             )
            x,y,z = self._crossAtt( 
                        self._conv1List[layer*2:layer*2+2],
                        self._crossList[layer*2:layer*2+2],
                        y,z,
                        True
                        )
        for layer in range(self._numconv1,self._numcross):
            # x,y,z = self._crossAtt( 
            #             None,
            #             self._crossList[layer*3:layer*3+6],
            #             x,y,z,
            #             False
            #             )
            y,z = self._crossAtt( 
                        None,
                        self._crossList[layer*2:layer*2+2],
                        y,z,
                        False
                        )
        return y,z
        # return x,y,z
            



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
        

        # self._embeddingX = ModalembProj(
        #     [TokenEmbedding( **cfg.modalX.token)],
        #         **cfg.modalX.Others
        #     ) 
        self._embeddingY = ModalembProj(
            [CatesEmbedding(**cfg.modalY.cates)],
                **cfg.modalY.Others
            ) 
        self._embeddingZ = ModalembProj(
            [CatesEmbedding(**cfg.modalZ.cates),
            TokenEmbedding(**cfg.modalZ.token1),
            TokenEmbedding(**cfg.modalZ.token2)
            ],
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
        self._crossModalBlock = CrossBlock(
                # crossargs:check out examples in paras
                # convargs:check out examples in paras
                [CrossLayer(**cfg.crossargs) for i in range( numcross*2) ],
                [ConvPoolLayer(**cfg.convargs) for i in range( numconv1*2 )],
                numcross,
                numconv1
            )
        

        encoders = [Encoder(
                        [EncoderLayer( 
                            AttentionLayer(
                                ProbAttention(**cfg.SelfATT.probAtt),
                                    **ARGattLayer
                            ),**ARGencoderLayer
                            )
                         for o in range(cfg.SelfATT.numSelfAttLayer)  ],
                        [ConvLayer(dm) 
                         for l in range(cfg.SelfATT.numSelfAttLayer-1) ] if cfg.SelfATT.distil else None,
                     norm_layer=torch.nn.LayerNorm(dm) 
                     ) for ARGattLayer,ARGencoderLayer,dm in zip(cfg.SelfATT.attLyrArgs,
                                                                 cfg.SelfATT.enclyrArgs,
                                                                [cfg.d_model,cfg.d_model*2]
                                                                 )
        # for i in range(3)
        ]
        self._selfAttentions = nn.ModuleList( encoders )
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
        # self._pooling
        self._projection = nn.Sequential(
                    # nn.Linear( cfg.d_model*9,(cfg.d_model*9)//2 ),#（256*9->256）
                    # nn.ReLU(),
                    # nn.Dropout(cfg.global_dropout),
                    # nn.Linear( (cfg.d_model*9)//2 ,(cfg.d_model*9)//4 ),
                    # nn.ReLU(),
                    # nn.Dropout(cfg.global_dropout),
                    # nn.Linear( (cfg.d_model*9)//4,128 ),
                    # nn.ReLU(),
                    # nn.Dropout(cfg.global_dropout),
                    # nn.Linear( 128,numclass )
                    nn.Linear( cfg.d_model*15,(cfg.d_model*15)//4 ),#（256*9->256）
                    nn.ReLU(),
                    nn.Dropout(cfg.global_dropout),
                    # nn.Linear( (cfg.d_model*9)//2 ,(cfg.d_model*9)//4 ),
                    # nn.ReLU(),
                    # nn.Dropout(cfg.global_dropout),
                    nn.Linear( (cfg.d_model*15)//4,256 ),
                    nn.ReLU(),
                    nn.Dropout(cfg.global_dropout),
                    nn.Linear( 256,numclass )
                    )
    def _pooling(self,x):
        # print(f'DEBUG value [x]\n==========={x}')
        # print(f'DEBUG shape [x]\n==========={x.shape}')
        
        x_std = torch.std(x, dim=1)
        x_mean = torch.mean(x, dim=1)
        x_max = torch.max(x, dim=1).values
        score  = torch.cat([x_std, x_mean,x_max], dim=1)
        return score

    def forward(self,x,y,z):
        # x = self._embeddingX(x) #x
        y = self._embeddingY(y,x) #x_cat
        z = self._embeddingZ(z,x) #x_extro


        o = torch.cat([
            # self._pooling(x),
            self._pooling(y),
            self._pooling(z)], 
            dim=1) #dim*6
        y,z = self._crossModalBlock(y,z)#
        o = self._normAndAct0(o)
        o = o + torch.cat([
            # self._pooling(x),
            self._pooling(y),
            self._pooling(z)], 
            dim=1)
        o = self._normAndAct1(o)

        # o =  #bs seq dim
        # c = #bs seq dim*3
        # x = self._selfAttentions[0](x)[0]
        # y = self._selfAttentions[1](y)[0]
        # z = self._selfAttentions[2](z)[0]
        o = torch.cat([o,
            self._pooling(self._selfAttentions[0](y+z )[0]),
            self._pooling(self._selfAttentions[1](torch.cat([ y,z],dim=-1 ))[0])], 
            dim=1) #3*2+3+3*2=15 dim
        # _out = torch.cat(
        #     [self._pooling(x),self._pooling(y), self._pooling(z)], 
        #     dim=1)
        o = self._projection(o)
        return  F.sigmoid(o)

