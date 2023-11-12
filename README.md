# Semantic Hearing

[![Gradio demo](https://img.shields.io/badge/DL.ACM-abs-green)](https://dl.acm.org/doi/10.1145/3586183.3606779) [![Gradio demo](https://img.shields.io/badge/DL.ACM-pdf-green)](https://dl.acm.org/doi/pdf/10.1145/3586183.3606779)

This repository provides code for the binaural target sound extraction model proposed in the paper, _Semantic Hearing: Programming Acoustic Scenes with Binaural Hearables_, presented at UIST'23. This model helps us create systems that let you control what you want to hear in the environment, in real-time, using noise-cancelling earbuds & headphones.

https://github.com/vb000/SemanticHearing/assets/16723254/f1b33d8c-179a-4d50-92aa-6a99dde696d0

## Conda environment setup

    conda create --name semhear python=3.8
    conda activate semhear
    pip install -r requirements.txt

## Training

    # Data
    wget -P data https://semantichearing.cs.washington.edu/BinauralCuratedDataset.tar

    # Train
    python -m src.training.train experiments/dc_waveformer --use_cuda

## Evaluation

    # Checkpoint
    wget -P experiments/dc_waveformer https://semantichearing.cs.washington.edu/39.pt

    # Eval
    python -m src.training.eval experiments/dc_waveformer --use_cuda

### BibTeX

```
@inproceedings{10.1145/3586183.3606779,
author = {Veluri, Bandhav and Itani, Malek and Chan, Justin and Yoshioka, Takuya and Gollakota, Shyamnath},
title = {Semantic Hearing: Programming Acoustic Scenes with Binaural Hearables},
year = {2023},
isbn = {9798400701320},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3586183.3606779},
doi = {10.1145/3586183.3606779},
abstract = {Imagine being able to listen to the birds chirping in a park without hearing the chatter from other hikers, or being able to block out traffic noise on a busy street while still being able to hear emergency sirens and car honks. We introduce semantic hearing, a novel capability for hearable devices that enables them to, in real-time, focus on, or ignore, specific sounds from real-world environments, while also preserving the spatial cues. To achieve this, we make two technical contributions: 1) we present the first neural network that can achieve binaural target sound extraction in the presence of interfering sounds and background noise, and 2) we design a training methodology that allows our system to generalize to real-world use. Results show that our system can operate with 20 sound classes and that our transformer-based network has a runtime of 6.56 ms on a connected smartphone. In-the-wild evaluation with participants in previously unseen indoor and outdoor scenarios shows that our proof-of-concept system can extract the target sounds and generalize to preserve the spatial cues in its binaural output. Project page with code: https://semantichearing.cs.washington.edu},
booktitle = {Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology},
articleno = {89},
numpages = {15},
keywords = {Spatial computing, binaural target sound extraction, attention, earable computing, causal neural networks, noise cancellation},
location = {San Francisco, CA, USA},
series = {UIST '23}
}
```
