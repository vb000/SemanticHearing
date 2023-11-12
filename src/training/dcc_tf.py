import math
from collections import OrderedDict
from typing import Optional
import logging

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding

def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod

class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        """
        x = x.permute(0, 2, 1) # [B, T, C]
        x = super().forward(x)
        x = x.permute(0, 2, 1) # [B, C, T]
        return x

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride,
                      padding, groups=in_channels, dilation=dilation),
            LayerNormPermuted(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1,
                      padding=0),
            LayerNormPermuted(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class DilatedCausalConvEncoder(nn.Module):
    """
    A dilated causal convolution based encoder for encoding
    time domain audio input into latent space.
    """
    def __init__(self, channels, num_layers, kernel_size=3):
        super(DilatedCausalConvEncoder, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # Compute buffer lengths for each layer
        # buf_length[i] = (kernel_size - 1) * dilation[i]
        self.buf_lengths = [(kernel_size - 1) * 2**i
                            for i in range(num_layers)]

        # Compute buffer start indices for each layer
        self.buf_indices = [0]
        for i in range(num_layers - 1):
            self.buf_indices.append(
                self.buf_indices[-1] + self.buf_lengths[i])

        # Dilated causal conv layers aggregate previous context to obtain
        # contexful encoded input.
        _dcc_layers = OrderedDict()
        for i in range(num_layers):
            dcc_layer = DepthwiseSeparableConv(
                channels, channels, kernel_size=3, stride=1,
                padding=0, dilation=2**i)
            _dcc_layers.update({'dcc_%d' % i: dcc_layer})
        self.dcc_layers = nn.Sequential(_dcc_layers)

    def init_ctx_buf(self, batch_size, device):
        """
        Returns an initialized context buffer for a given batch size.
        """
        return torch.zeros(
            (batch_size, self.channels,
                 (self.kernel_size - 1) * (2**self.num_layers - 1)),
            device=device)

    def forward(self, x, ctx_buf):
        """
        Encodes input audio `x` into latent space, and aggregates
        contextual information in `ctx_buf`. Also generates new context
        buffer with updated context.
        Args:
            x: [B, in_channels, T]
                Input multi-channel audio.
            ctx_buf: {[B, channels, self.buf_length[0]], ...}
                A list of tensors holding context for each dilation
                causal conv layer. (len(ctx_buf) == self.num_layers)
        Returns:
            ctx_buf: {[B, channels, self.buf_length[0]], ...}
                Updated context buffer with output as the
                last element.
        """
        T = x.shape[-1] # Sequence length

        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            # DCC input: concatenation of current output and context
            dcc_in = torch.cat(
                (ctx_buf[..., buf_start_idx:buf_end_idx], x), dim=-1)

            # Push current output to the context buffer
            ctx_buf[..., buf_start_idx:buf_end_idx] = \
                dcc_in[..., -self.buf_lengths[i]:]

            # Residual connection
            x = x + self.dcc_layers[i](dcc_in)

        return x, ctx_buf

class CausalTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    """
    Adapted from:
    "https://github.com/alexmt-scale/causal-transformer-decoder/blob/"
    "0caf6ad71c46488f76d89845b0123d2550ef792f/"
    "causal_transformer_decoder/model.py#L77"
    """
    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        chunk_size: int = 1
    ) -> Tensor:
        tgt_last_tok = tgt[:, -chunk_size:, :]

        # self attention part
        tmp_tgt, sa_map = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=None,
        )
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        ca_map = None
        if memory is not None:
            tmp_tgt, ca_map = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=None, # Attend to the entire chunk
                key_padding_mask=None,
            )
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # final feed-forward network
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok, sa_map, ca_map

class CausalTransformerDecoder(nn.Module):
    """
    A casual transformer decoder which decodes input vectors using
    precisely `ctx_len` past vectors in the sequence, and using no future
    vectors at all.
    """
    def __init__(self, model_dim, ctx_len, chunk_size, num_layers,
                 nhead, use_pos_enc, ff_dim, conditioning='conv'):
        super(CausalTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.unfold = nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)
        self.pos_enc_tgt = PositionalEncoding(model_dim, max_len=1000)
        self.pos_enc_mem = PositionalEncoding(model_dim, max_len=100)
        self.tf_dec_layers = nn.ModuleList([CausalTransformerDecoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim,
            batch_first=True) for _ in range(num_layers)])
        self.conditioning = conditioning

        if conditioning == 'film':
            self.film = nn.Sequential(
                nn.Linear(model_dim, 2 * model_dim),
                nn.ReLU())

    def init_ctx_buf(self, batch_size, device):
        return torch.zeros(
            (batch_size, self.num_layers + 1, self.ctx_len, self.model_dim),
            device=device)

    def _causal_unfold(self, x):
        """
        Unfolds the sequence into a batch of sequences
        prepended with `ctx_len` previous values.

        Args:
            x: [B, ctx_len + L, C]
            ctx_len: int
        Returns:
            [B * L, ctx_len + 1, C]
        """
        B, T, C = x.shape
        x = x.permute(0, 2, 1) # [B, C, ctx_len + L]
        x = self.unfold(x.unsqueeze(-1)) # [B, C * (ctx_len + chunk_size), -1]
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, C, self.ctx_len + self.chunk_size)
        x = x.reshape(-1, C, self.ctx_len + self.chunk_size)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, input, embedding, ctx_buf, K=4000):
        """
        Args:
            input: [B, model_dim, T]
            embedding: [B, NE, model_dim, embed_len]
            ctx_buf: [B, num_layers, ctx_len, model_dim]
            K: int
                Number of batches to process at once to avoid OOM.
        Returns:
            output: [B, model_dim, T]
            ctx_buf: [B, num_layers, ctx_len, model_dim]
        """

        # Mod pad the input so the sequence length is a multiple
        # of chunk_size.
        input, mod = mod_pad(input, self.chunk_size, (0, 0))

        # Init
        B, C, T = input.shape
        output = input.permute(0, 2, 1).contiguous()
        mem = None

        if self.conditioning == 'conv':
            # Convolutional/mutltiplicative conditioning
            input = input.view(1, B * C, T)
            input = F.pad(
                input, (embedding.shape[-1] - 1, 0)) # [1, B * C, T + embed_len - 1]
            emb_filter = torch.mean(embedding, dim=1).reshape(B * C, 1, -1)
            output = F.conv1d(input, emb_filter, groups=B * C)
            output = output.view(B, C, T)
            output = output.permute(0, 2, 1)
        elif self.conditioning == 'attn':
            # Use cross attn for conditioning
            mem = embedding.permute(0, 1, 3, 2) # [B, NE, embed_len, C]
            if self.use_pos_enc:
                mem = mem.view(-1, mem.shape[-2], mem.shape[-1])
                mem = mem + self.pos_enc_mem(mem)
                mem = mem.view(B, -1, mem.shape[-2], mem.shape[-1])
            mem = mem.reshape(B, -1, mem.shape[-1]) # [B, NE * embed_len, C]
            mem = mem.unsqueeze(1).repeat(
                1, (T // self.chunk_size), 1, 1
            ) # [B, T // chunk_size, NE * embed_len, C]
            mem = mem.reshape(
                -1, mem.shape[-2], mem.shape[-1]
            ) # [B * (T // chunk_size), NE * embed_len, C]
        elif self.conditioning == 'film':
            # Use FILM for conditioning
            emb_filter = torch.mean(embedding, dim=(1, 3)) # [B, C]
            emb_filter = self.film(emb_filter) # [B, 2 * C]
            gamma, beta = emb_filter.chunk(2, dim=-1)
            output = output * gamma.unsqueeze(1) + beta.unsqueeze(1)
        else:
            emb_filter = torch.mean(embedding, dim=(1, 3)) # [B, C]
            output = output * emb_filter.unsqueeze(1) # [B, T, C]

        for i, layer in enumerate(self.tf_dec_layers):
            # Prepend the context to the input and update the context
            # [B, ctx_len + T, C]
            tgt = torch.cat([ctx_buf[:, i, :, :], output], dim=1)
            ctx_buf[:, i, :, :] = tgt[:, -self.ctx_len:, :]

            # Unfold the sequence into a batch of sequences prepended
            # with `ctx_len` previous values.
            # [B * (T // chunk_size), ctx_len + chunk_size, C]
            tgt = self._causal_unfold(tgt)

            # Positional encoding
            if i == 0 and self.use_pos_enc:
                tgt = tgt + self.pos_enc_tgt(tgt)

            _tgt = torch.zeros_like(tgt)[:, :self.chunk_size, :]
            for k in range(int(math.ceil(tgt.shape[0] / K))):
                s, e = k * K, (k + 1) * K
                _mem = None if mem is None else mem[s:e]
                _tgt[s:e], _, _ = layer(tgt[s:e], _mem, self.chunk_size)

            output = _tgt.reshape(B, T, C)

        # Remove the mod padding
        output = output.permute(0, 2, 1)
        if mod != 0:
            output = output[:, :, :-mod]

        return output, ctx_buf

class MaskNet(nn.Module):
    def __init__(self, model_dim, num_enc_layers, dec_buf_len,
                 dec_chunk_size, num_dec_layers, use_pos_enc, conditioning):
        super(MaskNet, self).__init__()

        # Encoder based on dilated causal convolutions.
        self.encoder = DilatedCausalConvEncoder(channels=model_dim,
                                                num_layers=num_enc_layers)

        # Transformer decoder that operates on chunks of size
        # buffer size.
        self.decoder = CausalTransformerDecoder(
            model_dim=model_dim, ctx_len=dec_buf_len, chunk_size=dec_chunk_size,
            num_layers=num_dec_layers, nhead=8, use_pos_enc=use_pos_enc,
            ff_dim=2 * model_dim, conditioning=conditioning)

    def forward(self, x, l, enc_buf, dec_buf):
        """
        Generates a mask based on encoded input `e` and the one-hot
        label `label`.

        Args:
            x: [B, C, T]
                Input audio sequence
            l: [B, C]
                Label embedding
            ctx_buf: {[B, C, <receptive field of the layer>], ...}
                List of context buffers maintained by DCC encoder
        """
        # Enocder the label integrated input
        e, enc_buf = self.encoder(x, enc_buf)

        # Decoder conditioned on embedding
        m, dec_buf = self.decoder(input=e, embedding=l, ctx_buf=dec_buf)

        return m, enc_buf, dec_buf

class Net(nn.Module):
    def __init__(self, label_len, L=8,
                 model_dim=512, num_enc_layers=10,
                 dec_buf_len=100, num_dec_layers=2,
                 dec_chunk_size=72, out_buf_len=2,
                 use_pos_enc=True, conditioning="mult", lookahead=True):
        super(Net, self).__init__()
        self.L = L
        self.out_buf_len = out_buf_len
        self.model_dim = model_dim
        self.lookahead = lookahead

        # Input conv to convert input audio to a latent representation
        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=model_dim, kernel_size=kernel_size, stride=L,
                      padding=0, bias=False),
            nn.ReLU())

        # Label embedding layer
        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU())

        # Mask generator
        self.mask_gen = MaskNet(
            model_dim=model_dim, num_enc_layers=num_enc_layers,
            dec_buf_len=dec_buf_len,
            dec_chunk_size=dec_chunk_size, num_dec_layers=num_dec_layers,
            use_pos_enc=use_pos_enc, conditioning=conditioning)

        # Output conv layer
        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=model_dim, out_channels=1,
                kernel_size=(out_buf_len + 1) * L,
                stride=L,
                padding=out_buf_len * L, bias=False),
            nn.Tanh())

    def init_buffers(self, batch_size, device):
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size, device)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size, device)
        out_buf = torch.zeros(batch_size, self.model_dim, self.out_buf_len,
                              device=device)
        return enc_buf, dec_buf, out_buf
    
    def predict(self, x, label, enc_buf, dec_buf, out_buf, pad=True):
        mod = 0
        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)

        # Generate latent space representation of the input
        x = self.in_conv(x)

        # Generate label embedding
        l = self.label_embedding(label) # [B, label_len] --> [B, channels]
        l = l.unsqueeze(1).unsqueeze(-1) # [B, 1, channels, 1]

        # Generate mask corresponding to the label
        m, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)

        # Apply mask and decode
        x = x * m
        x = torch.cat((out_buf, x), dim=-1)
        out_buf = x[..., -self.out_buf_len:]
        x = self.out_conv(x)

        # Remove mod padding, if present.
        if mod != 0:
            x = x[:, :, :-mod]
        
        return x, enc_buf, dec_buf, out_buf

    def forward(self, inputs, init_enc_buf=None, init_dec_buf=None,
                init_out_buf=None, pad=True):
        """
        Extracts the audio corresponding to the `label` in the given
        `mixture`. Generates `chunk_size` samples per iteration.
        Args:
            mixed: [B, n_mics, T]
                input audio mixture
            label: [B, num_labels]
                one hot label
        Returns:
            out: [B, n_spk, T]
                extracted audio with sounds corresponding to the `label`
        """
        x, label = inputs['mixture'], inputs['label_vector']

        if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:
            assert init_enc_buf is None and \
                   init_dec_buf is None and \
                   init_out_buf is None, \
                "Both buffers have to initialized, or " \
                "both of them have to be None."
            enc_buf, dec_buf, out_buf = self.init_buffers(
                x.shape[0], x.device)
        else:
            enc_buf, dec_buf, out_buf = \
                init_enc_buf, init_dec_buf, init_out_buf

        x, enc_buf, dec_buf, out_buf = self.predict(
            x, label, enc_buf, dec_buf, out_buf)

        if init_enc_buf is None:
            return x
        else:
            return x, enc_buf, dec_buf, out_buf

# Define optimizer, loss and metrics

def optimizer(model, data_parallel=False, **kwargs):
    # Trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(params, **kwargs)

def loss(pred, tgt):
    return -si_snr(pred, tgt).mean()

def metrics(mixed, output, gt):
    """ Function to compute metrics """
    metrics = {}

    def metric_i(metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).cpu().item())
        return _vals

    for m_fn in [snr, si_snr]:
        metrics[m_fn.__name__] = metric_i(m_fn,
                                          mixed[:, :gt.shape[1], :],
                                          output,
                                          gt)

    return metrics

if __name__ == '__main__':
    net = CausalTransformerDecoder(
        model_dim=8, ctx_len=4, chunk_size=4, num_layers=2, nhead=4, conditioning='attn',
        use_pos_enc=True, ff_dim=16
    )
    x = torch.randn(2, 8, 16)
    e = torch.randn(2, 2, 8, 2)
    buf = torch.rand(2, 2, 4, 8)
    out = net(x, e, buf)
    print(out[0].shape, out[1].shape)
