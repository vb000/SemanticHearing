import os
import math
from collections import OrderedDict
from typing import Optional
import logging
from copy import deepcopy

import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

from src.training.dcc_tf import mod_pad, MaskNet
from src.helpers.eval_utils import itd_diff, ild_diff

class Net(nn.Module):
    def __init__(self, label_len, L=8,
                 model_dim=512, num_enc_layers=10,
                 dec_buf_len=100, num_dec_layers=2,
                 dec_chunk_size=72, out_buf_len=2,
                 use_pos_enc=True, conditioning="mult", lookahead=True,
                 pretrained_path=None):
        super(Net, self).__init__()
        self.L = L
        self.out_buf_len = out_buf_len
        self.model_dim = model_dim
        self.lookahead = lookahead

        # Input conv to convert input audio to a latent representation
        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels=2,
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
                in_channels=model_dim, out_channels=2,
                kernel_size=(out_buf_len + 1) * L,
                stride=L,
                padding=out_buf_len * L, bias=False),
            nn.Tanh())

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path)['model_state_dict']

            # Load all the layers except label_embedding and freeze them
            for name, param in self.named_parameters():
                if 'label_embedding' not in name:
                    param.data = state_dict[name]
                    param.requires_grad = False

    def init_buffers(self, batch_size, device):
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size, device)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size, device)
        out_buf = torch.zeros(batch_size, self.model_dim, self.out_buf_len,
                              device=device)
        return enc_buf, dec_buf, out_buf
    
    def predict(self, x, label, enc_buf, dec_buf, out_buf):
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

        return x, enc_buf, dec_buf, out_buf

    def forward(self, inputs, init_enc_buf=None, init_dec_buf=None,
                init_out_buf=None, pad=True, writer=None, step=None, idx=None):
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

        mod = 0
        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)

        x, enc_buf, dec_buf, out_buf = self.predict(
            x, label, enc_buf, dec_buf, out_buf)

        # Remove mod padding, if present.
        if mod != 0:
            x = x[:, :, :-mod]
        
        out = {'x': x}

        if init_enc_buf is None:
            return out
        else:
            return out, enc_buf, dec_buf, out_buf

# Define optimizer, loss and metrics

def optimizer(model, data_parallel=False, **kwargs):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(params, **kwargs)

def loss(_output, tgt):
    pred = _output['x']
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()

def metrics(inputs, _output, gt):
    """ Function to compute metrics """
    mixed = inputs['mixture']
    output = _output['x']
    metrics = {}

    def metric_i(metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append(torch.mean((metric(p, t) - metric(s, t))).cpu().item())
        return _vals

    for m_fn in [snr, si_snr]:
        metrics[m_fn.__name__] = metric_i(m_fn,
                                          mixed[:, :gt.shape[1], :],
                                          output,
                                          gt)

    return metrics

def test_metrics(inputs, _output, gt):
    test_metrics = metrics(inputs, _output, gt)
    output = _output['x']
    delta_itds, delta_ilds, snrs = [], [], []
    for o, g in zip(output, gt):
        delta_itds.append(itd_diff(o.cpu(), g.cpu(), sr=44100))
        delta_ilds.append(ild_diff(o.cpu().numpy(), g.cpu().numpy()))
        snrs.append(torch.mean(si_snr(o, g)).cpu().item())
    test_metrics['delta_ITD'] = delta_itds
    test_metrics['delta_ILD'] = delta_ilds
    test_metrics['si_snr'] = snrs
    return test_metrics

def format_results(idx, inputs, output, gt, metrics, output_dir=None):
    results = metrics
    results['metadata'] = inputs['metadata']
    results = deepcopy(results)

    # Save audio
    if output_dir is not None:
        output = output['x']
        for i in range(output.shape[0]):
            out_dir = os.path.join(output_dir, f'{idx + i:03d}')
            os.makedirs(out_dir)
            torchaudio.save(
                os.path.join(out_dir, 'mixture.wav'), inputs['mixture'][i], 44100)
            torchaudio.save(
                os.path.join(out_dir, 'gt.wav'), gt[i], 44100)
            torchaudio.save(
                os.path.join(out_dir, 'output.wav'), output[i], 44100)

    return results

if __name__ == "__main__":
    torch.random.manual_seed(0)

    model = Net(41)
    model.eval()

    with torch.no_grad():
        x = torch.randn(1, 2, 417)
        emb = torch.randn(1, 41)

        y = model({'mixture': x, 'label_vector': emb})

        print(f'{y.shape=}')
        print(f"First channel data:\n{y[0, 0]}")
