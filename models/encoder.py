import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvPoolLayer(nn.Module):
    def __init__(self, c_in,d_model,kernel_size=3):
        super(ConvPoolLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=d_model,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding=1)
        # self.norm2 = nn.LayerNorm(d_model)
        # dilation ->default 1
        # (L_in+2*padding-dilation*(kernel_size-1)-1)/2+1
        # (L_in+2*1-1*(3-1)-1 )/2+1=(L_in-1)/2+1

    def forward(self, x):

        x= self.downConv(x.permute(0, 2, 1))
        # namask =torch.isnan(x)
        x = self.activation(self.norm(x))
        x = self.maxPool(x)
            # .masked_fill(namask,torch.nan) )
        x = x.transpose(1,2)
        # namask = torch.isnan(x)

        # x = self.downConv(x.permute(0, 2, 1))
        # x = self.norm(x)#不同的example 不同的feature 同一个dim做norm
        # x = self.activation(x)
        # x = self.maxPool(x)
        # x = x.transpose(1,2)
        # x = self.norm2(x.masked_fill(namask,0)).masked_fill(namask,torch.nan)
        # del namask

        return x

class ConvLayer(nn.Module):
    def __init__(self, c_in,kernel_size,withMask=True):
        super(ConvLayer, self).__init__()
        self._withMask = withMask
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding=1)
        # dilation ->default 1
        # (L_in+2*padding-dilation*(kernel_size-1)-1)/2+1
        # (L_in+2*1-1*(3-1)-1 )/2+1=(L_in-1)/2+1

    def forward(self, x):
        if self._withMask:
            mask = x.permute(0,2,1)==0
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x) #permute后:不同的example 不同的dim 同一个fe
        x = self.activation(x) 
        if self._withMask:
            x[mask] = - torch.inf
        x = self.maxPool(x)
        if self._withMask:
            x[mask] = 0
        x = x.transpose(1,2)
        return x




class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # seq_len->seq_len feed forward layer(ffn) use conv1D kernel==1
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # seq_len->seq_len
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm2(x+y), attn
class Encoder(nn.Module):
    def __init__(self, encoder_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self._encoder_layers = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for enc_layer, conv_layer in zip(self._encoder_layers, self.conv_layers):
                x, attn = enc_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self._encoder_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for enc_layer in self._encoder_layers:
                x, attn = enc_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)#??????
        return x, attns
class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens
    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)#choose shape of input len
            # inp_len = (x.shape[1]-1)//(2**i_len) 512/256/128
            # input from original from couputed length -input_len
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        # concatnate all outputs
        
        return x_stack, attns