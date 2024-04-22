# Automated Playlist Description Generation (APDG) System

The Automated Playlist Description Generation (APDG) system is a deep learning model designed to automatically generate titles for playlists based on sequences of track IDs. This project leverages the Spotify Million Playlist Dataset (MPD) and employs Transformer models for understanding and generating meaningful and relevant playlist titles.

## Dataset
The model utilizes the Spotify Million Playlist Dataset (MPD), which contains a vast collection of playlists including track IDs and corresponding titles. Download the data files from [spotify-million-playlist](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files).


## Environment Setup

### Prerequisites
- Python 3.8.5
- PyTorch 1.9.0 (installation varies based on CUDA version, for CUDA 11.1 use `cu111`)

### Installation
Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Training the Model

### Preprocessing and Dataset Splitting
Run the preprocessing script to prepare and split the dataset:

```bash
python preprocess.py
```

### Training Options

For best performance, use the following parameters for training:

```bash
python train_model.py --shuffle True --e_pos False
```
## Evaluation and Inference
To generate playlist descriptions and perform inference, use the following command:
```bash
python infer.py --shuffle True --e_pos False
```

To evaluate the performance, use the follwing command
```bash
python eval.py
```

## Acknowledgments
This project is based on the work done in [ply_title_gen](https://github.com/seungheondoh/ply_title_gen.git).
