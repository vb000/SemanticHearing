"""
Torch dataset object for synthetically rendered spatial data.
"""

import os
import json
import random
from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scaper
import torch
import torchaudio
import torchaudio.transforms as AT
import sofa
from random import randrange

# Ignore scaper normalization warnings
warnings.filterwarnings(
    "ignore", message="Soundscape audio is clipping!")
warnings.filterwarnings(
    "ignore", message="Peak normalization applied")

class SemAudioBinauralBaseDataset(torch.utils.data.Dataset):  # type: ignore
    """
    Base class for FSD Sound Scapes dataset
    """
    def __init__(self, fg_dir, bg_dir, jams_dir, hrtf_dir, dset, sr=None,
                 resample_rate=None, max_num_targets=1):
        assert dset in ['train', 'val', 'test'], \
            "`dset` must be one of ['train', 'val', 'test']"

        self.labels = None
        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.hrtf_dir = hrtf_dir
        self.jams_dir = jams_dir
        self.dset = dset
        logging.info("Loading dataset: jams=%s fg_dir=%s bg_dir=%s" %
                     (self.jams_dir, self.fg_dir, self.bg_dir))

        self.samples = sorted(list(Path(self.jams_dir).glob('[0-9]*')))
        self.hrtf_dir = hrtf_dir

        self.max_num_targets = max_num_targets

        jamsfile = os.path.join(self.samples[0], 'mixture.jams')
        _, jams, _, _ = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)
        _sr = jams['annotations'][0]['sandbox']['scaper']['sr']
        assert _sr == sr, "Sampling rate provided does not match the data"

        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = resample_rate
        else:
            self.resampler = lambda a: a
            self.sr = sr

    def get_label_vector(self, labels):
        """
        Generates a multi-hot vector corresponding to `labels`.
        """
        vector = torch.zeros(len(self.labels))

        for label in labels:
            idx = self.labels.index(label)
            assert vector[idx] == 0, "Repeated labels"
            vector[idx] = 1

        return vector

    def load_sample(self, sample_dir, hrtf_dir, fg_dir, bg_dir, num_targets,
                    sample_targets=False, resampler=None):
        """
        Loads a single sample:
        sample_dir: Path to sample directory (jams file + metadata JSON)
        hrtf_dir: Path to hrtf dataset (sofa files)
        fg_dir: Path to foreground dataset ()
        bg_dir: Path to background dataset (TAU)
        num_targets: Number of gt targets to choose.
        sample_targets: Whether or not targets should be randomly chosen from the list
        channel: Channel index to select. If None, returns all channels
        """

        sample_path = sample_dir

        # Load HRIR
        metadata_path = os.path.join(sample_path, 'metadata.json')
        with open(metadata_path, 'rb') as f:
            metadata = json.load(f)

        HRTF_path = os.path.join(hrtf_dir, os.path.basename(metadata['sofa']))
        HRTF = sofa.Database.open(HRTF_path)

        # Load Audio
        jamsfile = os.path.join(sample_path, 'mixture.jams')
        mixture, jams, ann_list, event_audio_list = scaper.generate_from_jams(
            jamsfile, fg_path=fg_dir, bg_path=bg_dir)

        # Load source information
        sources = []
        source_list = metadata['sources']
        for i in range(len(source_list)):
            order = source_list[i]['order']
            pos = source_list[i]['position']
            rir_id = source_list[i]['hrtf_index']
            label = source_list[i]['label']
            if i == 0:
                assert label == 'background', "Background not first source"

            rir = HRTF.Data.IR.get_values(indices={"M":rir_id})
            sources.append((order, i, pos, rir, label))

        # Sort sources by order
        sources = sorted(sources, key=lambda x: x[0])

        gt_events = [x[4] for x in sources]
        gt_events = gt_events[:-1] # Remove background from gt_events

        if sample_targets:
            labels = random.sample(gt_events, randrange(1,num_targets+1))
        else:
            labels = gt_events[:num_targets]

        label_vector = self.get_label_vector(labels)

        mixture = np.zeros((2, mixture.shape[0]))

        gt = np.zeros_like(mixture)

        # Go over each source and convolve with HRTF
        metadata['chosen_sources'] = []
        for source in sources:
            _, i, _, rir, label = source

            # Get audio event as single-channel
            a = event_audio_list[i]
            a = a[..., 0]

            # Convolve single-channel with HRTF to get binaural
            tmp = np.zeros_like(mixture)
            tmp[0] = np.convolve(a, rir[0], mode='same')
            tmp[1] = np.convolve(a, rir[1], mode='same')

            if label in labels:
                gt += tmp
                metadata['chosen_sources'].append(i)

            mixture += tmp

        mixture = torch.from_numpy(mixture)
        gt = torch.from_numpy(gt)

        maxval = (torch.max(torch.abs(mixture)) + 1e-6)
        mixture = mixture / maxval
        gt = gt / maxval

        if resampler is not None:
            mixture = resampler(mixture.to(torch.float))
            gt = resampler(gt.to(torch.float))

        # Add microphone positions to metadata
        mic_positions = HRTF.Receiver.Position.get_values(system="cartesian")[..., 0]
        metadata['mic_positions'] = mic_positions.tolist()

        return mixture, gt, label_vector, metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        sample_targets=False
        if self.dset == 'train':
            num_targets = self.max_num_targets
            sample_targets=True
        elif self.dset == 'val':
            num_targets = idx%self.max_num_targets + 1
        elif self.dset == 'test':
            num_targets = self.max_num_targets

        mixture, gt, label_vector, metadata = \
            self.load_sample(
                sample_dir=sample_dir, hrtf_dir=self.hrtf_dir, fg_dir=self.fg_dir,
                bg_dir=self.bg_dir, num_targets=num_targets, sample_targets=sample_targets,
                resampler=self.resampler)

        mixture = self.resampler(mixture.to(torch.float))
        gt = self.resampler(gt.to(torch.float))

        # Ground-truth shifts using cross-correlation between gt channels
        _gt = gt.numpy()
        _gt = _gt / np.max(np.abs(_gt), axis=1, keepdims=True)

        shift = np.argmax(np.correlate(_gt[0][32:-32], _gt[1])) - 32
        shift = torch.tensor(shift)

        inputs = {
            'mixture': mixture,
            'label_vector': label_vector,
            'shift': shift,
            'metadata': metadata,
        }

        return inputs, gt

    def to(self, inputs, gt, device):
        inputs['mixture'] = inputs['mixture'].to(device)
        inputs['label_vector'] = inputs['label_vector'].to(device)
        inputs['shift'] = inputs['shift'].to(device)
        gt = gt.to(device)
        return inputs, gt

    def output_to(self, output, device):
        for k, v in output.items():
            output[k] = v.to(device)
        return output

    def output_detach(self, output):
        for k, v in output.items():
            output[k] = v.detach()
        return output

    def collate_fn(self, batch):
        inputs, gt = zip(*batch)
        inputs = {
            'mixture': torch.stack([i['mixture'] for i in inputs]),
            'label_vector': torch.stack([i['label_vector'] for i in inputs]),
            'shift': torch.stack([i['shift'] for i in inputs]),
            'metadata': [i['metadata'] for i in inputs]
        }
        gt = torch.stack(gt)
        return inputs, gt

    def tensorboard_add_sample(self, writer, tag, sample, step):
        """
        Adds a sample of FSDSynthDataset to tensorboard.
        """
        resample_rate = 8000 if self.sr > 8000 else self.sr

        inputs, output, gt = sample
        m = inputs['mixture']
        o = output['x']
        labels = []
        for lv in inputs['label_vector']:
            label = ''
            for i, l in enumerate(lv):
                if l == 1:
                    label += self.labels[i] + ';'
            labels.append(label)

        # Resample to 8kHz and normalize
        m, gt, o = (
            torchaudio.functional.resample(_, self.sr, resample_rate).cpu()
            for _ in (m, gt, o))
        m, gt, o = (_ / _.abs().max() for _ in (m, gt, o))

        def _add_audio(a, audio_tag, axis, plt_title):
            for i, ch in enumerate(a[:1]):
                axis.plot(ch, label='mic %d' % i)
                writer.add_audio(
                    '%s/mic %d' % (audio_tag, i), ch.unsqueeze(0), step, resample_rate)
            axis.set_title(plt_title)
            axis.legend()

        n_samples = min(8, m.shape[0])
        for b in range(n_samples):
            # Add waveforms
            rows = 3 # input, output, gt
            fig = plt.figure(figsize=(10, 2 * rows))
            axes = fig.subplots(rows, 1, sharex=True)
            l = labels[b]
            _add_audio(m[b], '%s/sample_%d/0_input' % (tag, b), axes[0], "Mixed")
            _add_audio(o[b], '%s/sample_%d/1_out_%s' % (tag, b, l), axes[1], f"Out ({l})")
            _add_audio(gt[b], '%s/sample_%d/2_gt_%s' % (tag, b, l), axes[2], f"GT ({l})")
            writer.add_figure('%s/sample_%d/waveform' % (tag, b), fig, step)

    def tensorboard_add_metrics(self, writer, tag, metrics, step):
        """
        Add metrics to tensorboard.
        """
        vals = np.asarray(metrics['scale_invariant_signal_noise_ratio'])

        writer.add_histogram('%s/%s' % (tag, 'SI-SNRi'), vals, step)

        return
