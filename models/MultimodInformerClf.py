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

    def _crossAtt(self,conv1s,crosses,x,y,z,hasConv):
        if hasConv:
            x1 = conv1s[0](
                        crosses[0](x,y) + crosses[1](x,z)
                    )
            y1 = conv1s[1](
                        crosses[2](y,x) + crosses[3](y,z)
                    )
            z1 = conv1s[2](
                        crosses[4](z,x) + crosses[5](z,y)
                    )
        else:
            x1 = crosses[0](x,y) + crosses[1](x,z)
            y1 = crosses[2](y,x) + crosses[3](y,z)
            z1 = crosses[4](z,x) + crosses[5](z,y)         
        return x1,y1,z1
    def forward(
        self,
        x, 
        y,
        z
        ):
        for layer in range(self._numconv1):
            x,y,z = self._crossAtt( 
                        self._conv1List[layer*3:layer*3+3],
                        self._crossList[layer*3:layer*3+6],
                        x,y,z,
                        True
                        )
        for layer in range(self._numconv1,self._numcross):
            x,y,z = self._crossAtt( 
                        None,
                        self._crossList[layer*3:layer*3+6],
                        x,y,z,
                        False
                        )
        return x,y,z
            



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
            TokenEmbedding( **cfg.modalX.token),
                **cfg.modalX.Others
            ) 
        self._embeddingY = ModalembProj(
            CatesEmbedding(**cfg.modalY.cates),
                **cfg.modalY.Others
            ) 
        self._embeddingZ = ModalembProj(
            CatesEmbedding(**cfg.modalZ.cates),
                **cfg.modalZ.Others
            ) 

        self._crossModalBlock = CrossBlock(
                # crossargs:check out examples in paras
                # convargs:check out examples in paras
                [CrossLayer(**cfg.crossargs) for i in range( numcross*6 ) ],
                [ConvPoolLayer(**cfg.convargs) for i in range( numconv1*3 )],
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
                                                                [cfg.d_model,cfg.d_model*3]
                                                                 )
        # for i in range(3)
        ]
        self._selfAttentions = nn.ModuleList( encoders )
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
                    nn.Linear( cfg.d_model*12,(cfg.d_model*12)//4 ),#（256*9->256）
                    nn.ReLU(),
                    nn.Dropout(cfg.global_dropout),
                    # nn.Linear( (cfg.d_model*9)//2 ,(cfg.d_model*9)//4 ),
                    # nn.ReLU(),
                    # nn.Dropout(cfg.global_dropout),
                    nn.Linear( (cfg.d_model*12)//4,256 ),
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
        x = self._embeddingX(x) #x
        y = self._embeddingY(y) #x_cat
        z = self._embeddingZ(z) #x_extro
        x,y,z = self._crossModalBlock(x,y,z)
        # o =  #bs seq dim
        # c = #bs seq dim*3
        # x = self._selfAttentions[0](x)[0]
        # y = self._selfAttentions[1](y)[0]
        # z = self._selfAttentions[2](z)[0]
        x = torch.cat([
            self._pooling(self._selfAttentions[0](x+y+z )[0]),
            self._pooling(self._selfAttentions[1](torch.cat([ x,y,z],dim=-1 ))[0])], 
            dim=1) #3*4 dim
        # _out = torch.cat(
        #     [self._pooling(x),self._pooling(y), self._pooling(z)], 
        #     dim=1)
        x = self._projection(x)
        return  F.sigmoid(x)









# class CFG:
#     '''========================================='''
#     #****************process*********
#     '''========================================='''
#     train = False
#     DEBUG = False
#     PREV_SEQ_CUT = 512
#     BackPre = True
#     prev_set = False
#     root     = '/kaggle/input/backpre/'
# #     root = '/content/gdrive/MyDrive/data/'
    
#     Dpath    = root + 'TRAIN/'
#     CFG_path = root + 'CFG/'
#     '''========================DATA=Modal========================='''
#     # ==Data===modalx
#     NUM_FEATS = ["elapsed_time_diff"]
#     # ==Data===modaly--list_vocab_sizes--list_embed_dims
#     CAT_FEATS = ['event_name', #vocab_size 11
#                  'name', #vocab_size 6
# #                  'text',  #vocab_size 597
#                  'fqid',  #vocab_size 128
# #                  'room_fqid', #vocab_size 19
# #                  'text_fqid' #vocab_size 126
#                 ]
#     modalYVocab = None #wait to be set
#     modalYembed_dims = [32,32, 64]
#     modalYtotdim = 128#32+32+64
#     # ==Data===modalz--list_vocab_sizes--list_embed_dims
#     EXTRO_CAT = ['room_coor_x',
#                  'room_coor_y',
# #                  'screen_coor_x',
# #                  'screen_coor_y'
#                 ]

    
#     modalZVocab = None
#     modalZembed_dims = [64,64]
#     modalZtotdim = 128 #64+64
#     coord_div=100
#     # (-20.0, 12.0)
#     # (-10.0, 5.0)
#     # (0.0, 19.0)
#     # (0.0, 14.0)
# #     EXTRO_CAT_CLIP = {'room_coor_x':[-100,62.0,62+100+1],
# #                      'room_coor_y':[-46, 27.0,62+100+1],
# #                      'screen_coor_x':[0,88,89],
# #                      'screen_coor_y':[0,65,66]}
    
#     EXTRO_CAT_CLIP = {'room_coor_x': [-20,   12,   20+12+1],
#                      'room_coor_y':  [-10.0, 5.0,  10+5+1],
# #                      'screen_coor_x':[0.0,   19.0, 20],
# #                      'screen_coor_y':[0.0,   14.0, 15]
#                      }
#     LIST_EMBED_DIMS = [ 32,32,
# #                        256,
#                        64,64,64,
# #                        128,128,128,128
#                        64,64
# #                        ,64,64
#                       ]
#     '''========================DATA=Modal=======↑=================='''
# #     COLS_TO_USE = ["session_id", "level", "level_group", "elapsed_time",
# #                    "event_name", "name", "room_fqid"]
#     CAT_FEATS_ALL = CAT_FEATS + EXTRO_CAT
#     COLS_TO_USE = ['session_id',
# #                  'index',
#                  'elapsed_time',
#                  'event_name', #%missing        0.000000
#                  'name', #%missing      0.000000
# #                  'level', 
#                  'room_coor_x',#%missing        0.078841
#                  'room_coor_y',#%missing        0.078841
# #                  'screen_coor_x',#%missing        0.078841
# #                  'screen_coor_y',#%missing        0.078841
# #                  'text',#%missing     0.634287
#                  'fqid',    #%missing       0.314653
#                  'room_fqid',#%missing      0.000000
#                  'text_fqid',#%missing      0.634283
#                  'level_group']
#     '''============================================='''
#     #****************models*********
#     '''============================================='''
#     d_model = 256
#     max_len = 512 #seq len
#     glb_kernel_size = 5
#     global_dropout = 0.2
#     num_class = 3
# #   CrossBlock  crossargs[dict]:attr of CFG class for CrossLayer para. examples
#     crossargs={'d_model':d_model, 
#             'd_ff':512,
#             'n_heads':4,
#             'factor':5,
#             'dropout':0.1, 
#             'activation':"relu"}
# #       CrossBlock  convargs[dict]:attr of CFG class for ConvLayer para. examples
#     convargs = {"d_model":256,"kernel_size":5}
#     class modalX:
#         token_dim = 1
#         token = {
#             "c_in": token_dim,
#             "d_model": CFG.d_model
#         }
#         Others = {
#                 'embedim':128,
#                  "d_model": CFG.d_model,
#                 'max_len': CFG.max_len,
#                 'kernel_size':CFG.glb_kernel_size
#         }
        
#     class modalY:
#         def __init__(self,modalYVocab,modalYtotdim):
# #         embedim = CFG.modalYtotdim
#             self.cates = {
#                "list_vocab_sizes" :modalYVocab,
#                'list_embed_dims' : CFG.modalYembed_dims,
#                'tot_cat_emb_dim'  : modalYtotdim,
#             }
#             self.Others = {
#                     'embedim':modalYtotdim,
#                      "d_model": CFG.d_model,
#                     'max_len': CFG.max_len,
#                     'kernel_size':CFG.glb_kernel_size
#         }

#     class modalZ:
#         def __init__(self,modalZVocab,modalZtotdim):
# #         embedim = CFG.modalZtotdim
#             self.cates = {
#                 'list_vocab_sizes' : modalZVocab,
#                'list_embed_dims' : CFG.modalZembed_dims,
#                'tot_cat_emb_dim'  : modalZtotdim,
#             }
#             self.Others = {
#                     'embedim':modalZtotdim,
#                      "d_model": CFG.d_model,
#                     'max_len': CFG.max_len,
#                     'kernel_size':CFG.glb_kernel_size
#             }
        
        
#     class SelfATT:
#         probAtt = {
#          "mask_flag":False, 
#          "factor":5, 
#          "attention_dropout":0.2, 
#          "output_attention":False
#         }
#         attLayer ={
#             "d_model":CFG.d_model, 
#             "n_heads":6, 
#             "mix":True
#         }
#         encoderLayer = {
#         'd_model':CFG.d_model, 
#         'd_ff':CFG.d_model*2, 
#         'dropout':0.1,
#         'activation':'gelu'
#         }
#         numSelfAttLayer = 2
#         distil=True
#     '''============================================='''
#     #****************training*********
#     '''============================================='''   
#     SEED = 42
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#     # "cuda:0"
#     EPOCH = 50
#     CKPT_METRIC = [
# #                     "f1@0.49","f1@0.53","f1@0.57",
#                    "f1@0.6","f1@0.63","f1@0.66",
#                    "f1@0.69","f1@0.73","f1@0.76",
#                    "f1@0.79","f1@0.83","f1@0.86"]

#     # ==DataLoader==
#     BATCH_SIZE = 128
#     NUM_WORKERS: 4

#     # ==Solver==
#     LR = 1e-3
#     WEIGHT_DECAY = 1e-4

#     # ==Early Stopping==
#     ES_PATIENCE = 5

#     # ==Evaluator==
#     EVAL_METRICS = ["auroc", "f1"]
#     def set_targets_seqlen(self,level_group):
#         PREV_DICT = {
#             "5-12":[["0-4"],512],
#             "13-22":[["0-4","5-12"], 512]#1024
#         }
#         self.level_curr = level_group
#         if level_group!='0-4':
#             self.prev_set=True
#             self.level_prev = PREV_DICT[level_group][0]
#             self.prev_seq_len = PREV_DICT[level_group][1]
#         f = lambda x:f'q{x}'
#         QNS_PER_LV_GP = {"0-4": list(map(f,list(range(1, 4)))),
#                  "5-12": list(map(f,list(range(4, 14)))), 
#                  "13-22": list(map(f,list(range(14, 19))))}
#         SEQ_LEN_DICT = {"0-4":512 ,
#                  "5-12": 800 ,
#                  "13-22": 1500}#1000
#         self.targets = QNS_PER_LV_GP[level_group]
#         self.seq_len = SEQ_LEN_DICT[level_group]
#         self.num_class = len(self.targets)
#     def set_(self,vocabs):
#         self.modalYVocab = vocabs[:len(self.CAT_FEATS)]
#         self.modalZVocab = vocabs[len(self.CAT_FEATS):]
#         modalYtotdim = sum(self.modalYembed_dims)
#         modalZtotdim = sum(self.modalZembed_dims)
#         self.modalY = CFG.modalY(self.modalYVocab, modalYtotdim)
#         self.modalZ = CFG.modalZ(self.modalZVocab, modalZtotdim)
        

# cfg = CFG()
# cfg.set_([ 1,2,3,4,5])
# cfg.CAT_FEATS
# cfg
