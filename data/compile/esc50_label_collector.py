import os, sys
import argparse

import pandas as pd
import numpy as np
from data.compile.ontology import Ontology


dictionary = {
    "dog":"Bark",
    "rooster":"Crowing, cock-a-doodle-doo",
    "pig":"Pig",
    "cow":"Cattle, bovinae",
    "frog":"Frog",
    "cat":"Meow",
    "hen":"Chicken, rooster",
    "insects":"Insect",
    "sheep":"Sheep",
    "crow":"Crow",

    "rain":"Rain",
    "sea_waves":"Waves, surf",
    "crackling_fire":"Crackle",
    "crickets":"Cricket",
    "chirping_birds":"Chirp, tweet",
    "water_drops":"Drip",
    "wind":"Wind",
    "pouring_water":"Pour",
    "toilet_flush":"Toilet flush",
    "thunderstorm":"Thunderstorm",

    "crying_baby":"Baby cry, infant cry",
    "sneezing":"Sneeze",
    "clapping":"Clapping",
    "breathing":"Breathing",
    "coughing":"Cough",
    "footsteps":"Walk, footsteps",
    "laughing":"Laughter",
    "brushing_teeth":"Toothbrush",
    "snoring":"Snoring",
    "drinking_sipping":None,

    "door_wood_knock":"Knock",
    "mouse_click":None,
    "keyboard_typing":"Computer keyboard",
    "door_wood_creaks":"Creak",
    "can_opening":None,
    "washing_machine":None,
    "vacuum_cleaner":"Vacuum cleaner",
    "clock_alarm":"Alarm clock",
    "clock_tick":"Tick-tock",
    "glass_breaking":"Shatter",

    "helicopter":"Helicopter",
    "chainsaw":"Chainsaw",
    "siren":"Siren",
    "car_horn":"Vehicle horn, car horn, honking",
    "engine":"Engine",
    "train":"Train",
    "church_bells":"Church bell",
    "airplane":"Fixed-wing aircraft, airplane",
    "fireworks":"Fireworks",
    "hand_saw":"Sawing"
}

def write_csv(dataset, csv_name):
    dataset = dataset[dataset.apply(lambda x: dictionary[x['category']] is not None)]


class ESC50LabelCollector():
    def __init__(self, dataset_dir, ontology_path) -> None:
        self.ontology = Ontology(ontology_path)

        # Load metadata
        meta = pd.read_csv(os.path.join(dataset_dir, 'meta/esc50.csv'))

        # # Create a audio path column
        # meta['audio_path'] = meta['filename'].apply(
        #     lambda x: os.path.join('..', '..', '..', 'ESC-50-master', 'audio', x))

        # Use first 3 folds for training, 4th for validation, 5th for testing
        self.train_meta = meta[meta['fold'] <= 3]
        self.val_meta = meta[meta['fold'] == 4]
        self.test_meta = meta[meta['fold'] == 5]
        
        self.dataset_dir = dataset_dir

    def filter_samples(self, dataset: pd.DataFrame):
        dataset['label'] = dataset['category'].apply(lambda x: dictionary[x])
        dataset = dataset.dropna().copy()
    
        dataset['fname'] = dataset['filename'].apply(lambda x: os.path.join('audio', x))
        dataset['id'] = dataset['label'].apply(lambda x: self.ontology.get_id_from_name(x))

        return dataset

    def write_samples(self):
        columns = ['fname', 'label', 'id']

        train = self.filter_samples(self.train_meta)
        train = train[columns]

        val = self.filter_samples(self.val_meta)
        val = val[columns]
        
        test = self.filter_samples(self.test_meta)
        test = test[columns]
        
        train.to_csv(os.path.join(self.dataset_dir, 'train.csv'), index=False)
        val.to_csv(os.path.join(self.dataset_dir, 'val.csv'), index=False)
        test.to_csv(os.path.join(self.dataset_dir, 'test.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/BinauralCuratedDataset/ESC-50')

    args = parser.parse_args()

    label_collector = ESC50LabelCollector(args.dataset_dir, 'data/ontology.json')
    label_collector.write_samples()
