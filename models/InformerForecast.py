import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.embed import CollEmbedding

class InformerForecast(nn.Module):
    def __init__(
        self, 
        toke_dim=2, 
        seq_len=1024,
        label_len=128, 
        out_len=896, 
        factor=5, 
        d_model=64, 
        n_heads=6, 
        e_layers=3, 
        d_layers=2, 
        d_ff=128, 
        dropout=0.2, 
        attn='prob',
        embed_type='fixed', freq='h', activation='gelu', 
        output_attention = False, distil=True, mix=True,
        device=torch.device('cuda')):
        super(InformerForecast, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding

        self.enc_embedding = CollEmbedding(toke_dim, d_model, embed_type, freq, dropout)
        self.dec_embedding = CollEmbedding(toke_dim, d_model, embed_type, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
                [EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, 
                                attention_dropout=dropout,
                                output_attention=output_attention), 
                            d_model, 
                            n_heads, 
                            mix=False ),
                        d_model,d_ff,
                        dropout=dropout,
                        activation=activation ) for o in range(e_layers)  ],
                [ConvLayer(d_model) for l in range(e_layers-1) ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model) 
                )
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
        self.projection = nn.Linear(d_model, toke_dim, bias=True)
        
    def forward(
        self, 
        ESx, 
        ETx,
        ECx, 
        DSx, 
        DTx, 
        DCx, 
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        '''
        Parameters
        ----------
        ESx (sequence(float)): targets timeserise for encoder inputs 
        ETx (sequence(int))  : time feats for encoder inputs 
        ECx (sequence(int))  : station name as cate feas for encoder inputs
        DSx (sequence(float)): targets timeserise for encoder inputs
        DTx (sequence(int))  : time feats for encoder inputs
        DCx (sequence(int))  : station name as cate feas for encoder inputs  

        Variabels
        ----------
        OE:encoder output recier -->emb-->encode
        OA:encoder output attention
        OD:encoder output recier -->emb-->decode
        '''
        OE = self.enc_embedding( ESx, ETx,ECx, )#seq_len->seq_len/2?
        OE, OA = self.encoder(OE, attn_mask=enc_self_mask)

        OD = self.dec_embedding(DSx,DTx,DCx)
        OD = self.decoder(OD, OE, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        OD = self.projection(OD)
        
        if self.output_attention:
            return OD[:,-self.pred_len:,:], attns
        else:
            return OD[:,-self.pred_len:,:] # [B, L, D]