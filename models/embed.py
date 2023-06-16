import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from models.encoder import ConvPoolLayer




class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # max_len?预定句子长度？
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 传入位置
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):

        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        # c_in=target_dim = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        # (N,Cin,Lin) or (N,seqdim,seqlen)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
                #参数可以自动初始化，但是这里自定义用kaiming
    def forward(self, x):
        # x->(batch,seq,dim)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2) #(bs,seq,embdim )
        # batch first
        #seq_len=seq_len-kernel_size+1? False结果应该和stride有关 
        # seq_len=seq_len-stride+1；True
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        # c_in len of timeseries?
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 18
        weekday_size = 7; day_size = 32; month_size = 13
        # num tokens?

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        # minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x 
        # + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)#max_len?
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        # self.position_embedding(x) (bs, seq_len,d_model)
        
        return self.dropout(x)



class CollEmbedding(nn.Module):
    def __init__(self, token_dim, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(CollEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=token_dim, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)#max_len?
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self._station_embedding = nn.Embedding( 7, d_model )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark,x_station):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        # self.position_embedding(x) (bs, seq_len,d_model)
        x =  x + self._station_embedding(x_station)
        return self.dropout(x)



class CatesEmbedding(nn.Module):
    def __init__(
        self, 
        list_vocab_sizes ,
        list_embed_dims  ,
        tot_cat_emb_dim  ,
        ):

        super(CatesEmbedding, self).__init__()
        self._cat_emb_list = nn.ModuleList(
          [nn.Embedding( vocab_sizes, embed_dim ,padding_idx=0) 
                for vocab_sizes,embed_dim in zip(list_vocab_sizes,list_embed_dims)
                ]
            )
        # combin_dims = sum(list_embed_dims)
        # self._emb_enc = nn.Sequential(
        #     nn.Linear(combin_dims, combin_dims ),
        #     nn.ReLU(),
        #     nn.Linear(combin_dims, int(combin_dims/2) ),
        #     nn.ReLU(),
        #     nn.Linear(int(combin_dims/2), tot_cat_emb_dim),
        #     nn.ReLU()
        # )
        
    def forward(self, x_cat):
        embeddings = []
        for i,emb_layer in enumerate(self._cat_emb_list):
            o = emb_layer( x_cat[:,:,i] )
            # print(o.shape)
            embeddings.append( o )
        embeddings = torch.cat(embeddings,axis=2 )
        # embeddings = torch.stack(embeddings, dim=2 )
        
        # print('*********before cat embs *********')
        # print(embeddings.shape)
        # embeddings = self._emb_enc(embeddings) #commented


        # print('*********after cat embs *********')
        # print(embeddings.shape)
        return embeddings



class CollectEmbedding(nn.Module):
    """Colect embedings before put into ecoder.

    Args:
        token_dim           : input  dim of numeric timeseries
        token_emb_dim(int)  : output dim of numeric timeseries
        list_vocab_sizes    : input  vocab size list of each category timeseries
        tot_cat_emb_dim(int): sum of all category embedding dims
        final_emb_dim       : output of projecttion dim from all embeddings
                            : in->token_emb_dim + tot_cat_emb_dim 
                            : out-> final_emb_dim
        dropout(float)      : dropout rate of CollectEmbedding net. 
    Attributes:
        _pos_emb(ModuleList(nn.Layer)): embedding of position of timeseries.
        _emb_list(nn.Layer)           : embedding of numberic and category timeseries values.
        _emb_dim_proj(int)            : The output embedding projection layer.
        _dropout(paddle.nn.Layer)     : The output dropout layer.
    """
    def __init__(
        self, 
        token_dim,
        token_emb_dim,
        list_vocab_sizes,
        list_embed_dims,
        tot_cat_emb_dim, 
        final_emb_dim, 
        dropout=0.1
        ):
        super(CollectEmbedding, self).__init__()

        all_embed_dim = tot_cat_emb_dim + token_emb_dim
        self._pos_emb = PositionalEmbedding(d_model= token_emb_dim)#max_len?
        # may be add is not a goog choice
        # separate cats embedding with position embedding
        self._tokenEmb = TokenEmbedding(c_in=token_dim, d_model=token_emb_dim)
        self._catesEmb = CatesEmbedding(list_vocab_sizes = list_vocab_sizes,
                           list_embed_dims = list_embed_dims,
                           tot_cat_emb_dim  = tot_cat_emb_dim,
                                       # args... 
                                       )
        
        # 
        # self._emb_list = nn.ModuleList( [ 
        #     TokenEmbedding(c_in=token_dim, d_model=token_emb_dim),
        #     CatesEmbedding(list_vocab_sizes = list_vocab_sizes,
        #                    list_embed_dims = list_embed_dims,
        #                    tot_cat_emb_dim  = tot_cat_emb_dim,
        #                                # args... 
        #                                ) ] )
        
        # self._emb_dim_proj = nn.Sequential(
        #             nn.Linear( all_embed_dim,all_embed_dim*2 ),
        #             nn.ReLU(),
        #             # nn.Dropout(p=dropout),
        #             # nn.Linear( all_embed_dim,final_emb_dim ),
        #             nn.Linear( all_embed_dim*2,all_embed_dim ),
        #             # nn.Dropout(p=dropout)
        #     )
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Prapares linearly projected queries, keys and values for usage of subsequent
        multiple parallel attention. then splits heads (reshape and transpose) to get keys
        and values from different representation subspaces. The results are used as key-values pairs
        for subsequent multiple parallel attention.

        Args:
            x(list[torch.Tensor]): [x,x_cat].
        Returns:
            d(list): list for input data type from x(dict).
            p(torch.Tensor): position embedding.
            e(list[torch.Tensor]): [numric_embeddings,category_embddings].
            o(torch.Tensor): embedings output reciver->linear project of e + position embedding .
        """
        p = self._pos_emb(x[0] )
        # e = []
        # embstype = ['num','cat']
        to = self._tokenEmb( x[0] )
        to = self.dropout( to + p )
        ca = self._catesEmb( x[1] )
        o = torch.cat( [ to,ca ] , axis=2 )
        return o 

        # for n in range(2):
        #     e_i = self._emb_list[n]( x[n] ) #(bs,seq,dim)
        #     # print(f'*********emb.shape {embstype[n]} *********')
        #     e.append( e_i )
        #     # print(e_i.shape)
            
        # o = torch.cat(e, axis=2 )
        # print('*********concat.shape*********')
        # print(o.shape)
        # o = self._emb_dim_proj( o )+o
        # o = self.dropout(o)
        # return o + p#(bs,seq,all_embed_dim)
     
class ModalembProj(nn.Module):
    def __init__(
        self,
        embedding,
        embedim,
        d_model,
        max_len,
        kernel_size,
        DEBUG=False
        # kernel_size=5
        ):
        super(ModalembProj, self).__init__()
        self.DEBUG = DEBUG
        self._embedding = nn.ModuleList(embedding)

        if self.DEBUG:
            print('DEBUG Parameter [embedding]\n[==============]')
            print(' - ',embedding.parameters)
        # self._encode  = nn.Sequential(
        #         nn.Linear( embedim,   embedim*2 ),
        #         nn.ReLU(),
        #         nn.Linear( embedim*2, embedim  ),
        #         nn.ReLU(),
        #     )

        self._con1D = ConvPoolLayer(embedim ,d_model,kernel_size )
        max_len = int((max_len+2- kernel_size+1-1)/2+1)

        # self._con1D = nn.Conv1d(
        #     # (in_channels=d_model, out_channels=d_ff, kernel_size=1
        #     # projection part play role as linear kernel should be 1
        #                     in_channels=embedim, 
        #                     out_channels=d_model, 
        #                     kernel_size=1, 
        #                     # padding=1, 
        #                     # padding_mode='circular'
        #                     )
        self._pos_embed = PositionalEmbedding(d_model=d_model,max_len=max_len)
        # self._dropout = nn.Dropout(0.1)
        self._dropout2 = nn.Dropout(0.1)
    def forward(self,x):
        if self.DEBUG:
            print('DEBUG shape [Input]')
            print('[==============] - ',x.shape)
        if len(self._embedding)>1:
            # embs = []
            # t = 0
            # for o,_emb in enumerate( self._embedding ):
            #     embs.append( _emb( x[:,:,:o] ) )
            #     t +=1

            # _o = sum( embs )
            # for o,e in enumerate( embs[:-1] ):
            #     for e1 in  embs[o+1:]:
            #         _o += e*e1
            #         t += 1
            # _o = _o/t #V2
            #     self._embedding[0](x[:,:,:-2]),
            #     self._embedding[1](x[:,:,-2]),
            #     self._embedding[2](x[:,:,-1])],

            # x = torch.cat([
            #     self._embedding[0](x[:,:,:-2]),
            #     self._embedding[1](x[:,:,-2]),
            #     self._embedding[2](x[:,:,-1])],
            #     dim = -1
            #     ) #V1
            x = torch.cat([
                self._embedding[0](x[0]),
                self._embedding[1](x[1][:,:,0].unsqueeze(-1)  ),
                self._embedding[2](x[1][:,:,1].unsqueeze(-1)) ],
                dim = -1
                ) #V3
            # x = x_time * x
            
        else:
            x = self._embedding[0](x)#(bs,seq,embdim)
            # x = x_time * x
        if self.DEBUG:
            print('DEBUG shape [Embed]')
            print('[==============] - ',x.shape)
            print('DEBUG Parameter [con1D]')
            print('[==============] - ',self._con1D.parameters)
        # x = self._dropout(self._encode(x))#(bs,seq,embdim)
        x = self._con1D(x)
        x = x + self._pos_embed(x)#(bs,seq,d_model)
        return self._dropout2(x)