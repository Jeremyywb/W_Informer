import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import CollectEmbedding

class Informer(nn.Module):
        """Colect embedings before put into ecoder.

    Args:
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
        tot_cat_emb_dim, 
        input_seq_len,
        final_emb_dim=512,
        dropout=0.1,
        factor=5,
        output_attention = False,
        n_heads=8,
        d_ff=512,
        activation='gelu',
        distil=True,
        num_class=num_class,

        label_len, out_len, 
                 d_model=512,  n_e_l=3, d_layers=2,  
                dropout=0.0, attn='prob', embed='fixed', freq='h',  
                  mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention


        # Encoding
        self.enc_embedding = CollectEmbedding(
                        token_dim,
                        token_emb_dim,
                        list_vocab_sizes,
                        tot_cat_emb_dim, 
                        final_emb_dim, 
                        dropout)#(bs,seq,final_emb_dim)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self._encoder = Encoder(
                [EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, 
                                attention_dropout=dropout,
                                output_attention=output_attention), 
                            final_emb_dim, 
                            n_heads, 
                            mix=False ),
                        final_emb_dim,d_ff,
                        dropout=dropout,
                        activation=activation ) for o in range(n_e_l)  ],
                [ConvLayer(final_emb_dim) for l in range(n_e_l-1) ] if distil else None,
                norm_layer=torch.nn.LayerNorm(final_emb_dim) 
                )
        

        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=final_emb_dim, out_channels=c_out, kernel_size=1, bias=True)
        out_len_seq = self.pooling_out_len(Lin=input_seq_len ,num_pool= n_e_l-1)
        self.mlps = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear( out_len_seq*final_emb_dim,final_emb_dim ),
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
        x_mark_enc, 
        enc_self_mask=None, 
        ):
        x = self.enc_embedding(x)
        x, attns = self._encoder(x, attn_mask=enc_self_mask)
        x = self.mlps(x)
        x = F.sigmoid(x)
        return x


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
            Encoder( 
                [EncoderLayer(
                    AttentionLayer(
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

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
