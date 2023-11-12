from src.training.datasets.curated_binaural import CuratedBinauralDataset
import os
import sofa
import json
import scaper
import random
import torch
import numpy as np
from random import randrange

from data.multi_ch_simulator import Multi_Ch_Simulator

import hashlib


class CuratedBinauralAugRIRDataset(CuratedBinauralDataset):
    """
    Torch dataset object for synthetically rendered spatial data.
    """
    def __init__(self, *args, **kwargs):
        self.scaper_bg_dir = kwargs['bg_scaper_dir']
        kwargs.pop('bg_scaper_dir', None)
        
        if 'reverb' in kwargs:
            self.reverb = kwargs['reverb']
            kwargs.pop('reverb', None)
        else:
            self.reverb = True

        super().__init__(*args, **kwargs)
        
        # Simulate
        self.simulator = Multi_Ch_Simulator(self.hrtf_dir, self.dset, self.sr, self.reverb)
    
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

        # Load background audio
        bg_jamsfile = os.path.join(sample_path, 'background.jams')
        _, _, _, bg_event_audio_list = scaper.generate_from_jams(
            bg_jamsfile, fg_path=self.scaper_bg_dir, bg_path=bg_dir)

        # Load foreground audio
        fg_jamsfile = os.path.join(sample_path, 'mixture.jams')
        mixture, _, _, fg_event_audio_list = scaper.generate_from_jams(
            fg_jamsfile, fg_path=fg_dir, bg_path='.')

        # Read number of background sources
        num_background = metadata['num_background']

        source_labels = []
        source_list = metadata['sources']
        for i in range(len(source_list)):
            label = source_list[i]['label']
            
            # Sanity check
            if i < num_background:
                assert label not in self.labels, "Background sources are not in the right order"
            else:
                assert label in self.labels, "Foreground sources are not in the right order"

            source_labels.append(label)

        # Concatenate event audio lists
        event_audio_list = np.array(bg_event_audio_list + fg_event_audio_list, dtype=np.float32)[..., 0]

        # Generate random simulator
        simulator = self.simulator.get_random_simulator()

        if self.dset == 'test':
            seed = int.from_bytes(hashlib.sha256(str(sample_dir).encode()).digest()[:4], 'little') # 32-bit int
            simulator.seed(seed)
        
        total_samples = mixture.shape[0]
        gt_audio = simulator.initialize_room_with_random_params(len(source_list), 0, source_labels, num_background)\
                            .simulate(event_audio_list)[..., :total_samples]
        metadata = simulator.get_metadata()

        # Load source information
        sources = []
        source_list = metadata['sources']
        for i in range(len(source_list)):
            order = source_list[i]['order']
            pos = source_list[i]['position']
            label = source_list[i]['label']

            sources.append((order, i, pos, gt_audio[i], label))

        # Sort sources by order
        sources = sorted(sources, key=lambda x: x[0])

        gt_events = [x[4] for x in sources]
        
        # Remove background from gt_events
        gt_events = gt_events[:-num_background]

        if sample_targets:
            labels = random.sample(gt_events, randrange(1,num_targets+1))
        else:
            labels = gt_events[:num_targets]

        label_vector = self.get_label_vector(labels)

        # Generate mixture and gt audio
        mixture = np.sum(gt_audio, axis=0)
        gt = np.zeros_like(mixture)

        # Go over each source and convolve with HRTF
        metadata['chosen_sources'] = []
        for source in sources:
            _, i, _, audio, label = source

            if label in labels:
                gt += audio
                metadata['chosen_sources'].append(i)
                
        mixture = torch.from_numpy(mixture)
        gt = torch.from_numpy(gt)
        
        maxval = float(torch.max(torch.abs(mixture)))
        # If maxval > 1, normalize so that input is between [-1, 1]
        if maxval > 1:
            mixture = mixture / maxval
            gt = gt / maxval
            
            maxval = 1
            
        # # Augment scale
        # if self.dset != 'test':
        #     random_amplitude = np.random.uniform(0.2, 1)
        #     random_scale = random_amplitude / maxval
        #     mixture *= random_scale
        #     gt *= random_scale
        #     metadata['random_amplitude'] = random_amplitude

        if resampler is not None:
            mixture = resampler(mixture.to(torch.float))
            gt = resampler(gt.to(torch.float))

        return mixture, gt, label_vector, metadata
