import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import CollectEmbedding

class Informer(nn.Module):
    """Colect embedings before put into ecoder.

    Args:
        token_dim(int)  : input numric feature dim
        n_e_l           : num of encoder_layers
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
        input_seq_len,
        final_emb_dim=256,
        dropout=0.1,
        factor=5,
        output_attention = False,
        n_heads=8,
        d_ff=256,
        activation='gelu',
        distil=True,
        num_class=3,
        n_e_l=3,
        attn='prob',
        mix=True,
        device=torch.device('cuda')):
        super(Informer, self).__init__()

        # Encoding

        self._enc_embedding = CollectEmbedding(
                        token_dim,
                        token_emb_dim,
                        list_vocab_sizes,
                        list_embed_dims,
                        tot_cat_emb_dim, 
                        final_emb_dim, 
                        dropout)#(bs,seq,final_emb_dim)256
        # Attention
        self._ontop_down_conv1D = ConvLayer(final_emb_dim) 
        Attn = ProbAttention if attn=='prob' else FullAttention
        att_ = {
         "mask_flag":False, 
         "factor":factor, 
         "attention_dropout":dropout, 
         "output_attention":False
        }
        attl_ ={
            "d_model":final_emb_dim, 
            "n_heads":n_heads, 
            "mix":mix
        }
        ecl_ = {
        'd_model':final_emb_dim, 
        'd_ff':d_ff, 
        'dropout':dropout,
        'activation':activation
        }
        self._encoder = Encoder(
                [EncoderLayer( AttentionLayer(Attn(**att_),**attl_),**ecl_)
                 for o in range(n_e_l)  ],
                [ConvLayer(final_emb_dim) 
                 for l in range(n_e_l-1) ] if distil else None,
                 norm_layer=torch.nn.LayerNorm(final_emb_dim) 
                 )
        self._att_previous = AttentionLayer(Attn(**att_),**attl_)
        self._dropout_prev = nn.Dropout(p=dropout)

        # conv1d_seq_len = self.pooling_out_len(Lin=input_seq_len ,num_pool=1)
        # self.out_len_seq = self.pooling_out_len(Lin=conv1d_seq_len ,num_pool= n_e_l-1)
        # V2 del flatten /add mean/std polling
        self.mlps = nn.Sequential(
                    # nn.Flatten(),
                    # nn.Linear( self.out_len_seq*final_emb_dim,final_emb_dim ),
                    nn.Linear( final_emb_dim*2,final_emb_dim ),#（256*2->256）
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear( final_emb_dim,128 ),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear( 128,num_class )
                    )
        

    def pooling_out_len(
        self,
        Lin,
        num_pool
        ):
        f = lambda x:int((x-1)/2+1)
        for i in range( num_pool):
            Lin = f(Lin)
        return Lin
        
    def forward(
        self, 
        x, 
        x_cat, 
        enc_self_mask=None,
        previous_state=None,
        output_seq_states=False
        ):
        x = self._enc_embedding([x,x_cat])
        # x = self._ontop_down_conv1D(x)

        if previous_state is not None:
            new_x, attn = self._att_previous(
            x, previous_state, previous_state,
            attn_mask = None
             )
            x = x + self._dropout_prev(new_x)
        x, attns = self._encoder(x, attn_mask=enc_self_mask)


        x_std = torch.std(x, dim=1)  # x(B,L,D)->(B,D) Std pooling
        x_mean = torch.mean(x, dim=1)  # Mean pooling
        score = torch.cat([x_std, x_mean], dim=1)
        score = self.mlps( score )
        score = F.sigmoid(score)

        if output_seq_states:
            return x
        return score


class InformerStackClf(nn.Module):
    """Colect embedings before put into ecoder.

    Args:
        token_dim(int)  : input numric feature dim
        e_layers            : num of encoder_layers in each encoder
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
        input_seq_len,
        final_emb_dim=256,
        dropout=0.1,
        factor=5,
        output_attention = False,
        n_heads=8,
        d_ff=256,
        activation='gelu',
        distil=True,
        num_class=3,
        # n_e_l=3,
        e_layers=[3,2,1],
        attn='prob',
        mix=True,DEBUG=False,
        prev_len_cut_to=128,
        add_prev=True,
        device=torch.device('cuda')):
        super(InformerStackClf, self).__init__()

        # Encoding
        self.DEBUG = DEBUG
        self.prev_len_cut_to = prev_len_cut_to
        self.DEBUGINFO = ''
        self._enc_embedding = CollectEmbedding(
                        token_dim,
                        token_emb_dim,
                        list_vocab_sizes,
                        list_embed_dims,
                        tot_cat_emb_dim, 
                        final_emb_dim, 
                        dropout)#(bs,seq,final_emb_dim)256
        # Attention
        self._ontop_down_conv1D = ConvLayer(final_emb_dim) 
        Attn = ProbAttention if attn=='prob' else FullAttention
        att_ = {
         "mask_flag":False, 
         "factor":factor, 
         "attention_dropout":dropout, 
         "output_attention":False
        }
        attl_ ={
            "d_model":final_emb_dim, 
            "n_heads":n_heads, 
            "mix":mix,
            'DEBUG':DEBUG
        }
        ecl_ = {
        'd_model':final_emb_dim, 
        'd_ff':d_ff, 
        'dropout':dropout,
        'activation':activation
        }

        inp_len_divs = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [ EncoderLayer( AttentionLayer(Attn(**att_),**attl_),**ecl_) 
                    for l in range(el)],#Encoder[EncoderLayers]s
                [ConvLayer(final_emb_dim) for l in range(el-1) ] if distil else None,#ConvLayers
                norm_layer=torch.nn.LayerNorm(final_emb_dim)) 
            for el in e_layers]
        self._encoder = EncoderStack(encoders, inp_len_divs)


        # self._encoder = Encoder(
        #         [EncoderLayer( AttentionLayer(Attn(**att_),**attl_),**ecl_)
        #          for o in range(n_e_l)  ],
        #         [ConvLayer(final_emb_dim) 
        #          for l in range(n_e_l-1) ] if distil else None,
        #          norm_layer=torch.nn.LayerNorm(final_emb_dim) 
        #          )
        self._att_previous = AttentionLayer(Attn(**att_),**attl_)
        self._dropout_prev = nn.Dropout(p=dropout)

        # conv1d_seq_len = self.pooling_out_len(Lin=input_seq_len ,num_pool=1)
        # self.out_len_seq = self.pooling_out_len(Lin=conv1d_seq_len ,num_pool= n_e_l-1)
        # V2 del flatten /add mean/std polling
        if add_prev:
            self.mlps = nn.Sequential(
                    # nn.Flatten(),
                    # nn.Linear( self.out_len_seq*final_emb_dim,final_emb_dim ),
                    nn.Linear( final_emb_dim*6,final_emb_dim ),#（256*2->256）
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear( final_emb_dim,128 ),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear( 128,num_class )
                    )
        else:
            self.mlps = nn.Sequential(
                    # nn.Flatten(),
                    # nn.Linear( self.out_len_seq*final_emb_dim,final_emb_dim ),
                    nn.Linear( final_emb_dim*3,final_emb_dim ),#（256*2->256）
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear( final_emb_dim,128 ),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear( 128,num_class )
                    )
        

    def pooling_out_len(
        self,
        Lin,
        num_pool
        ):
        f = lambda x:int((x-1)/2+1)
        for i in range( num_pool):
            Lin = f(Lin)
        return Lin
        
    def forward(
        self, 
        x, 
        x_cat,
        x_prev=None,
        x_cat_prev=None,
        enc_self_mask=None,
        output_seq_states=False
        ):
        x = self._enc_embedding([x,x_cat])
        x = self._ontop_down_conv1D(x)
        
        if x_prev is not None:
            if self.DEBUG:
                print(f'''===============|DEBUG STEP[PREV ENCODE]|===============''')
            stp = x_prev.shape[1]//self.prev_len_cut_to
            for i in range(stp):         
                xp = self._enc_embedding(
                        [
                        x_prev[:,i*self.prev_len_cut_to:(i+1)*self.prev_len_cut_to,:],
                        x_cat_prev[:,i*self.prev_len_cut_to:(i+1)*self.prev_len_cut_to,:]
                        ]
                    )
                oxptmp = None
                xp = self._ontop_down_conv1D(xp)
                if i==0:
                    oxp, _ = self._encoder(xp, attn_mask=enc_self_mask)
                else:
                    oxptmp ,_= self._encoder(xp, attn_mask=enc_self_mask)
                    oxp = oxp+oxptmp  #B L++,D?
        #     del _
        #     if self.DEBUG:
        #         print( f'''===============DEBUG STEP[PREV ATT]|===============''')
        #     new_x, attn = self._att_previous(
        #     x, oxp, oxp,
        #     attn_mask = None
        #      )
        #     x = x + self._dropout_prev(new_x)
        # if self.DEBUG:
        #     print( f'''===============DEBUG STEP[CURR ENCODE]|===============''')
        # if x.shape[1]>256:
        #     if self.DEBUG:
        #         print( f'''DEBUG STEP[CURR MemContol]|===============''')
        #         print( f'''DEBUG [INFO]|lenght seq input from {x.shape[1]} to {int((x.shape[1]-1)/2+1) }''')
        #     x = self._ontop_down_conv1D(x)
        x, attns = self._encoder(x, attn_mask=enc_self_mask)
        del attns
        x_std = torch.std(x, dim=1)  # x(B,L,D)->(B,D) Std pooling
        x_mean = torch.mean(x, dim=1)  # Mean pooling
        x_max = torch.max(x, dim=1).values
        score = torch.cat([x_std, x_mean,x_max], dim=1)
        if x_prev is not None:
            x_std = torch.std(oxp, dim=1)
            x_mean = torch.mean(oxp, dim=1)
            x_max = torch.max(oxp, dim=1).values
            score  = torch.cat([score,x_std, x_mean,x_max], dim=1)

        score = self.mlps( score )
        score = F.sigmoid(score)

        if output_seq_states:
            return x
        return score
