
import torch.nn as nn
from transformer.sublayers import MultiHeadAttention
from transformer.sublayers import MultiBranchAttention
from transformer.sublayers import PoswiseFeedForwardNet
from transformer.modules import LayerNormalization

class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        # (b , len, hidden)  (b, len_q, len_k)
        return enc_outputs, attn


class WeightedEncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedEncoderLayer, self).__init__()
        self.enc_self_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        return self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask=self_attn_mask)


class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.dec_enc_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask, is_initial):
        # 因为要做两次attention，所以分了选择
        if is_initial:
            dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                            dec_inputs, attn_mask=self_attn_mask)
            dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                          enc_outputs, attn_mask=enc_attn_mask)
        else:
            dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_inputs, enc_outputs,
                                                          enc_outputs, attn_mask=enc_attn_mask)
            dec_self_attn = None
        # dec_outputs_ = self.pos_ffn(dec_outputs)
        #
        # # haiyou an add & self.layer_norm(dec_outputs_ + dec_outputs)
        # dec_outputs_ = self.layer_norm(dec_outputs_ + dec_outputs)

        # # add pre
        dec_outputs_ = self.pos_ffn(self.layer_norm(dec_outputs))
        dec_outputs_ = (dec_outputs_ + dec_outputs)

        return dec_outputs_, dec_self_attn, dec_enc_attn


class WeightedDecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        self.dec_enc_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, attn_mask=enc_attn_mask)
        
        return dec_outputs, dec_self_attn, dec_enc_attn
