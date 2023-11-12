import os, glob
import argparse
import sys

import pandas as pd
import numpy as np
import random
import ffmpegio
import tqdm

import torchaudio, librosa
from scipy.io.wavfile import write as wavwrite

from data.compile.ontology import Ontology


def read_audio_file_torch(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform

def read_audio_file(file_path, sr):
    """
    Reads audio file to system memory.
    """
    return librosa.core.load(file_path, mono=False, sr=sr)[0]


def write_audio_file(file_path, data, sr):
    """
    Writes audio file to system memory.
    @param file_path: Path of the file to write to
    @param data: Audio signal to write (n_channels x n_samples)
    @param sr: Sampling rate
    """
    wavwrite(file_path, sr, data)

def convert_videos(video_paths, audio_dir, segment_duration_s):
    os.makedirs(audio_dir, exist_ok=True)

    instrumental_dir = os.path.join(audio_dir, 'instrumental')
    os.makedirs(instrumental_dir, exist_ok=True)
    
    vocals_dir = os.path.join(audio_dir, 'vocals')
    os.makedirs(vocals_dir, exist_ok=True)
    
    for i in tqdm.tqdm(range(len(video_paths))):
        path = video_paths[i]

        print(path)

        song_name = os.path.basename(path)
        audio_streams = ffmpegio.probe.audio_streams_basic(path)
        duration_samples = audio_streams[0]['duration'].numerator
        sr = audio_streams[0]['sample_rate']
        segment_duration_samples = int(round(sr * segment_duration_s))
        
        # Remaining audio must be at least 1/2 chunk size
        num_chunks = 1 + (duration_samples - segment_duration_samples // 2 - 1) // segment_duration_samples
        
        for chunk_id in tqdm.tqdm(range(num_chunks)):
            start_time = chunk_id * segment_duration_s
            _, mixture = ffmpegio.audio.read(path, ss=start_time, t=segment_duration_s, ac=1)
            _, vocals = ffmpegio.audio.read(path, ss=start_time, t=segment_duration_s, map=[['0','4']], ac=1)

            instrumental = mixture - vocals
            
            # Save audio files only if they are not completely silent (i.e. no vocals this chunk)
            if (np.abs(vocals) > 5e-3).any():
                vocals_path = os.path.join(vocals_dir, f'{song_name}_v_{chunk_id}.wav')
                write_audio_file(vocals_path, vocals, sr)

            if (np.abs(instrumental) > 5e-3).any():
                instrumental_path = os.path.join(instrumental_dir, f'{song_name}_i_{chunk_id}.wav')
                write_audio_file(instrumental_path, instrumental, sr)

class MUSDB18LabelCollector():
    def __init__(self, ontology_path) -> None:
        self.ontology = Ontology(ontology_path)


    def write_csv(self, dataset_dir, dataset_type):
        samples = []

        preproc_dir = os.path.join(dataset_dir, 'audio', dataset_type)

        instrumental_dir = os.path.join(preproc_dir, 'instrumental')
        vocals_dir = os.path.join(preproc_dir, 'vocals')

        for sample_path in glob.glob(os.path.join(vocals_dir, '*.wav')):
            rel_path = os.path.relpath(sample_path, dataset_dir)
            label = 'Singing'
            sample = dict(label=label,
                          fname=rel_path,
                          id=self.ontology.get_id_from_name(label))
            samples.append(sample)

        for sample_path in glob.glob(os.path.join(instrumental_dir, '*.wav')):
            rel_path = os.path.relpath(sample_path, dataset_dir)
            label = 'Melody'
            sample = dict(label=label,
                          fname=rel_path,
                          id=self.ontology.get_id_from_name(label))
            samples.append(sample)
        
        df = pd.DataFrame.from_records(samples)
        output_csv = os.path.join(dataset_dir, dataset_type + '.csv')
        df.to_csv(output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/BinauralCuratedDataset/musdb18')
    parser.add_argument('--segment_duration_s', type=str, default=15)
    args = parser.parse_args()
    
    random.seed(0)

    assert os.path.exists(args.dataset_dir), f"Path {args.dataset_dir} to dataset is invalid (not found)"
    
    audio_dir = os.path.join(args.dataset_dir, 'audio')

    if not os.path.exists(audio_dir):
        print("[INFO] DATASET HAS NOT BEEN PREPROCESSED - PREPROCESSING... (THIS MAY TAKE SOME TIME)")
        os.makedirs(audio_dir, exist_ok=True)

        test_video_list = sorted(list(glob.glob(os.path.join(args.dataset_dir, 'test', '*'))))
        
        # Split train into train & val sets
        train_video_list = sorted(list(glob.glob(os.path.join(args.dataset_dir, 'train', '*'))))
        
        random.shuffle(train_video_list)
        val_split = int(round(0.1 * len(train_video_list)))
        val_video_list = train_video_list[:val_split]
        train_video_list = train_video_list[val_split:]

        convert_videos(train_video_list, os.path.join(audio_dir, 'train'), args.segment_duration_s)
        convert_videos(test_video_list, os.path.join(audio_dir, 'test'), args.segment_duration_s)
        convert_videos(val_video_list, os.path.join(audio_dir, 'val'), args.segment_duration_s)

    collector = MUSDB18LabelCollector('data/ontology.json')
    collector.write_csv(args.dataset_dir, 'train')
    collector.write_csv(args.dataset_dir, 'test')
    collector.write_csv(args.dataset_dir, 'val')
