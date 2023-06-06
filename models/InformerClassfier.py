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
        mix=True,
        device=torch.device('cuda')):
        super(InformerStackClf, self).__init__()

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
        x = self._ontop_down_conv1D(x)
        
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




class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding  enc_in = input dim of targets
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder([EncoderLayer(AttentionLayer(
                        Attn( False, 
                              factor, 
                              attention_dropout=dropout, 
                              output_attention=output_attention), 
                        d_model, 
                        n_heads, 
                        mix=False),#AttentionLayer
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation ) for l in range(el)],#Encoder[EncoderLayers]s
                [ConvLayer(d_model) for l in range(el-1) ] if distil else None,#ConvLayers
                norm_layer=torch.nn.LayerNorm(d_model)) 
            for el in e_layers]

        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]




            

        #         x = x.view(-1, 64*5*5) # Flatten layer
        # x = self.dropout(self.fc1(x))
        # x = self.dropout(self.fc2(x))
        # x = F.log_softmax(self.fc3(x),dim = 1)
        #         self.hidden = nn.Linear(4, 8)
        # self.act = nn.ReLU()
        # self.output = nn.Linear(8, 3)
        #         self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        # self.layer_2 = nn.Linear(in_features=5, out_features=1) #

        # ['acc', 'f1_macro', 'aucroc']
