## Setup


    # Commands in all sections except the Dataset section are run from repo's toplevel directory
    conda create --name waveformer python=3.8
    conda activate waveformer
    pip install -r requirements.txt

## Training and Evaluation

### Dataset

We use [Scaper](https://github.com/justinsalamon/scaper) toolkit to synthetically generate audio mixtures. Each audio mixture sample is generated on-the-fly during training or evaluation, using Scaper's `generate_from_jams` function on a `.jams` specification file. We provide the `.jams` specification files for all training, validation and evaluation samples (in the step 3 below). The `.jams` specifications are generated using [FSDKaggle2018](https://zenodo.org/record/2552860) and [TAU Urban Acoustic Scenes 2019](https://dcase.community/challenge2019/task-acoustic-scene-classification) datasets as sources for foreground and background sounds, respectively. Steps to create the dataset:

1. Go to the `data` directory:

        cd data

2. Download [FSDKaggle2018](https://zenodo.org/record/2552860), [TAU Urban Acoustic Scenes 2019, Development dataset](https://zenodo.org/record/2589280) and [TAU Urban Acoustic Scenes 2019, Evaluation dataset](https://zenodo.org/record/3063822) datasets using the `data/download.py` script:

        python download.py

3. Make a directory for the dataset and download the `.jams` specifications:

        mkdir BinauralFSDSoundScapes

4. Uncompress FSDKaggle2018 dataset and create scaper source:

        unzip FSDKaggle2018/\*.zip -d FSDKaggle2018
        python fsd_scaper_source_gen.py FSDKaggle2018 ./FSDSoundScapes/FSDKaggle2018 ./FSDSoundScapes/FSDKaggle2018

5. Uncompress TAU Urban Acoustic Scenes 2019 dataset to `FSDSoundScapes` directory:

        unzip TAU-acoustic-sounds/\*.zip -d FSDSoundScapes/TAU-acoustic-sounds/

6. Move `CIPIC_HRTF` to `BinuralFSDSoundScapes` directory:

        mv CIPIC_HRTF BinauralFSDSoundScapes/

7. Generate `.jams` specifications for training, validation and evaluation sets:

        cd ..
        python data/hrtf_split.py
        python data/soundscape_gen.py

### Training

    python -W ignore -m src.training.train experiments/<Experiment dir with config.json> --use_cuda

### Evaluation

Pretrained checkpoints are available at [experiments.zip](https://targetsound.cs.washington.edu/files/experiments.zip). These can be downloaded and uncompressed to appropriate locations using:

    wget https://targetsound.cs.washington.edu/files/experiments.zip
    unzip -o experiments.zip -d experiments

Run evaluation script:

    python -W ignore -m src.training.eval experiments/<Experiment dir with config.json and checkpoints> --use_cuda

### Note

During the sample generation, when the amplitude of mixture sum to greater than 1, peak normalization is used to renormalize the mixtures. This results in a bunch of Scaper warnings during training and evaluation. `-W ignore` flag is used for a clearner output to the console.
