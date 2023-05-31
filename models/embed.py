import torch.nn.functional as F
import torch.nn as nn
import torch
import math




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

        minute_size = 4; hour_size = 24
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
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

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

class CatesEmbedding(nn.Module):
    def __init__(
        self, 
        list_vocab_sizes = list_vocab_sizes, 
        tot_cat_emb_dim  = tot_cat_emb_dim,
        ):

        super(CatesEmbedding, self).__init__()
        assert tot_cat_emb_dim % len(list_vocab_sizes) == 0
        each_cat_out_dim = int( tot_cat_emb_dim/len(list_vocab_sizes) )
        self._cat_emb_list = nn.ModuleList(
          [nn.Embedding( vocab_sizes, each_cat_out_dim ) for vocab_sizes in list_vocab_sizes]
            )
        self.emb_enc = nn.Sequential(
            nn.Linear(tot_cat_emb_dim, tot_cat_emb_dim*2 ),
            nn.ReLU(),
            nn.Linear(tot_cat_emb_dim*2, tot_cat_emb_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        embeddings = []
        for i,emb_layer in enumerate(self._cat_emb_list):
            o = emb_layer( x[:,:,i] )
            embeddings.append( o )
        embeddings = torch.cat(embeddings, -1 )
        embeddings = self.emb_enc(embeddings)
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
        tot_cat_emb_dim, 
        final_emb_dim, 
        dropout=0.1
        ):
        super(DataEmbedding, self).__init__()

        all_embed_dim = tot_cat_emb_dim + token_emb_dim
        self._pos_emb = PositionalEmbedding(d_model= final_emb_dim)#max_len?
        self._emb_list = nn.ModuleList( [ 
            TokenEmbedding(c_in=token_dim, d_model=token_emb_dim),
            CatesEmbedding(list_vocab_sizes = list_vocab_sizes, 
                           tot_cat_emb_dim  = tot_cat_emb_dim,
                                       # args... 
                                       ) ] )
        
        self._emb_dim_proj = nn.Sequential(
                    nn.Linear( all_embed_dim,all_embed_dim*2 ),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear( all_embed_dim*2,final_emb_dim )
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
            )
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Prapares linearly projected queries, keys and values for usage of subsequent
        multiple parallel attention. then splits heads (reshape and transpose) to get keys
        and values from different representation subspaces. The results are used as key-values pairs
        for subsequent multiple parallel attention.

        Args:
            x(dict[str:torch.Tensor]): dict of input tensors.
        Returns:
            d(list): list for input data type from x(dict).
            p(torch.Tensor): position embedding.
            e(list[torch.Tensor]): [numric_embeddings,category_embddings].
            o(torch.Tensor): embedings output reciver->linear project of e + position embedding .
        """
        d = ['num_list','cat_list' ]
        p = self._pos_emb(x['num_list'])
        e = []
        for typ,emb_layer in zip(d,self._emb_list):
            e_i = emb_layer( x[typ] ) #(bs,seq,dim)
            e.append( e_i )
        o = torch.cat(e, -1 )
        o = self._emb_dim_proj( o )
        o = o + p

        return self.dropout( o )#(bs,seq,final_emb_dim)

