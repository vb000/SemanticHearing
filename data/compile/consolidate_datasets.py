import os, glob
import subprocess
import argparse
from tqdm import tqdm
import yaml, json
import typing
import pandas as pd
import random
from ontology import Ontology

my_ontology = Ontology('data/ontology.json')

# Merges several datasets into a single Scaper Format
# Replaces Soundscape Gen
# Given datasets distributed as:
# - base_dir/
# -- Dataset1/
# --- train.csv
# --- test.csv
# --- val.csv
# --- audio/*.wav
# -- Dataset2/
# --- ...
# -- Dataset3/
# --- ...

# Generates:
# - base_dir/
# -- ScaperFormat/
# --- train/
# ---- class1/*.wav
# ---- class2/*.wav
# --- test/
# ---- ...
# --- val/
# ---- ...

# [1] Go over each dataset
# [2] Go over dataset type
# [3] Go over label in dataset
# [4] Check which classname this item label belongs to
# [5] Create symlink between wavefile under the relevant classname directory

# `classname` refers to the label name in *our* subset, `label` is reserved for dataset name

all_samples = []

def meta_csv_to_dict(meta):
    """
    Convert a ['fname', 'labels', 'mids'] headed dataframe to
    a dict with labels as keys, and list of file names as values.
    """
    samples_dict = {}
    samples = pd.read_csv(meta)
    ids = list(samples['id'].unique())
    for _id in ids:
        samples_dict[_id] = list(samples.loc[samples['id'] == _id]['fname'])

    return samples_dict

def is_valid_background(label_id, foreground_ids):
    """
    A label is valid background label iff it is not in an exclude subtree,
    and neither an ancestor nor a child of any foreground label.
    Since AudioSet hierarchies is more of a DAG than a tree, the concept of
    ancestor is a bit unclear. Instead, we just check if either node is reachable
    from the other.
    (we can precompute this at the beginning of each node but it's not much of a bottleneck)
    """

    excluded_subtrees = [my_ontology.MUSIC,
                         my_ontology.get_id_from_name('Human voice')]
    for subtree in excluded_subtrees:
        if my_ontology.is_reachable(subtree, label_id):
            return False

    for fg_id in foreground_ids:
        if my_ontology.is_reachable(label_id, fg_id) or\
           my_ontology.is_reachable(fg_id, label_id):
            return False
        
    return True


from scipy.io.wavfile import read
import librosa
import numpy as np


def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

from scipy.ndimage import uniform_filter1d
def trim_silence(s):
    # data, sr = librosa.load(s)
    sr, data = read(s)
    if len(data.shape) > 1:
        data = np.sum(data, axis=1)
    
    if data.dtype != np.float32:
        data = pcm2float(data)
    start, end = librosa.effects.trim(data, top_db=40)[1]
    
    data = data[start:end]
    
    window_size = int(round(1 * 44100))
    avg_power = uniform_filter1d(data**2, size=window_size, mode='constant')
    threshold = 0.1 * avg_power.max()
    
    mask = avg_power < threshold
    if mask.any():
        first_silence = np.argmax(mask)
    else:
        first_silence = end

    return start, first_silence, end

def write_scaper_source(dataset_name: str,
                        dataset_type: str,
                        base_dir: str,
                        fg_dest_dir: str,
                        bg_dest_dir: str,
                        id2classname: typing.Dict,
                        dry_run: bool) -> None:
    dataset_path = os.path.join(base_dir, dataset_name)
    fg_out_dir = os.path.join(fg_dest_dir, dataset_type)
    bg_out_dir = os.path.join(bg_dest_dir, dataset_type)
    
    file_list_csv = os.path.join(dataset_path, f"{dataset_type}.csv")
    dataset = pd.read_csv(file_list_csv)

    print(f"Consolidating dataset {dataset_name}/{dataset_type}...")

    for index, sample_data in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        # Check if we want to include this class
        if sample_data["id"] in id2classname:
            out_dir = fg_out_dir
            classname = id2classname[sample_data["id"]]
        elif is_valid_background(sample_data["id"], list(id2classname.keys())):
            out_dir = bg_out_dir
            classname = my_ontology.get_label(sample_data["id"])
        else:
            continue
        
        out_path = os.path.join(out_dir, classname)
        os.makedirs(out_path, exist_ok=True)

        s = os.path.join('..', '..', '..', dataset_name, sample_data['fname'])

        fname = os.path.join(dataset_name.lower() + '_' + os.path.basename(sample_data['fname']))
        d = os.path.join(out_path, fname)

        if dry_run:
            print("Would symlink %s to %s" % (s, d))
            continue
        else:
            start_sample, first_silence, end_sample = trim_silence(os.path.join(dataset_path, sample_data['fname']))
            assert start_sample < end_sample
            all_samples.append({'fname':d, 'start_sample':int(start_sample), 'end_sample':int(end_sample), 'first_silence':first_silence})
            os.symlink(s, d)

def read_yaml(yaml_path):
    with open(yaml_path, "r") as stream:
        yaml_data = yaml.safe_load(stream)

    return yaml_data

def read_json(json_path):
    with open(json_path, "r") as stream:
        json_data = json.load(stream)

    return json_data

def preprocess_ontology(ontology):
    res = {}
    for sound in ontology:
        res[sound['id']] = sound
        res[sound['name']] = sound
    return res

def get_subtree(_id, ontology):
    subtree = [_id]
    for child_id in ontology[_id]['child_ids']:
        subtree.extend(get_subtree(child_id, ontology))

    return subtree

if __name__ == '__main__':
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets_dir', type=str, default='data/BinauralCuratedDataset',
        help="Path to directory containing all datasets.")
    parser.add_argument(
        '--class_definitions', type=str, default='data/Classes.yaml',
        help="Path to class susbet selection.")
    parser.add_argument(
        '--ontology', type=str, default='data/ontology.json',
        help="Path to ontology definition.")
    parser.add_argument(
        '--fg_output_dir', type=str, default='data/BinauralCuratedDataset/scaper_fmt',
        help="Path to directory to write scaper formatted data.")
    parser.add_argument(
        '--bg_output_dir', type=str, default='data/BinauralCuratedDataset/bg_scaper_fmt',
        help="Path to directory to write scaper formatted data.")
    parser.add_argument(
        '--dry_run', action='store_true', help="Dry run. Do not write any files.")
    args = parser.parse_args()

    datasets = ['FSD50K', 'ESC-50', 'musdb18', 'disco_noise']
    dataset_types = ['train', 'val', 'test']

    for dset in datasets:
        print(f'Collecting dataset {dset}...')
        collector_name = 'data/compile/' + dset.replace('-', '').lower() + '_label_collector.py'
        subprocess.run(['python', collector_name])

    # Construct dict that maps an AudioSet class ID to a classname used in our dataset
    # Also helpful when classnames across datasets are different but same ID is used
    id2classname = {}
    class_data = read_yaml(args.class_definitions)
    ontology = read_json(args.ontology)
    ontology = preprocess_ontology(ontology)

    for class_name, class_list in class_data.items():
        for element in class_list:
            class_id = ontology[element]['id']

            # Map entire subtree to classname
            class_ids = get_subtree(class_id, ontology)

            for cid in class_ids:
                id2classname[cid] = class_name

    print(id2classname)

    for dataset_name in datasets:
        for dataset_type in dataset_types:
            write_scaper_source(dataset_name=dataset_name,
                                dataset_type=dataset_type,
                                base_dir=args.datasets_dir,
                                fg_dest_dir=args.fg_output_dir,
                                bg_dest_dir=args.bg_output_dir,
                                id2classname=id2classname,
                                dry_run=args.dry_run)

    df = pd.DataFrame.from_records(all_samples)
    df.to_csv(os.path.join(args.datasets_dir, 'start_times.csv'))
