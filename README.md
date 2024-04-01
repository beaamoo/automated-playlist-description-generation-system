# Automated Playlist Description Generation (APDG) System

The Automated Playlist Description Generation (APDG) system is a deep learning model designed to automatically generate titles for playlists based on sequences of track IDs. This project leverages the Spotify Million Playlist Dataset (MPD) and employs Transformer models for understanding and generating meaningful and relevant playlist titles.

## Dataset
The model utilizes the Spotify Million Playlist Dataset (MPD), which contains a vast collection of playlists including track IDs and corresponding titles. Download the data files from [spotify-million-playlist](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files).

## File Structure
```
APDG/
│
├── src/                        # Source files for the APDG system
│   ├── data/                   # Data handling scripts
│   │   ├── __init__.py         # Makes directories Python packages
│   │   └── spotify_api.py      # Script to fetch data from the Spotify API
│   │
│   ├── model/                  # Transformer model and utilities
│   │   ├── __init__.py
│   │   ├── transformer.py      # Transformer model implementation
│   │   └── utils.py            # Utility functions, e.g., tokenization
│   │
│   └── main.py                 # Main script to run the APDG system
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── model_exploration.ipynb # Explore model architecture, training, etc.
│   └── api_data_fetching.ipynb # Notebook for exploring Spotify API data
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_spotify_api.py     # Tests for Spotify API data fetching
│   └── test_transformer.py     # Tests for transformer model functionality
│
├── requirements.txt            # Project dependencies
├── setup.py                    # Setup script for the APDG package
└── README.md                   # Project overview and setup instructions
```


### Data Preparation
Before training the model, the dataset is preprocessed and split into training, validation, and test sets. Detailed instructions for data preparation are provided below.

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
python preprocessing.py
```

### Training Options

For best performance, use the following parameters for training:

```bash
python train.py --dataset_type mpd --model transformer --shuffle True --e_pos False
```
## Evaluation and Inference
To evaluate the model and perform inference, use the following commands:
```bash
python eval.py --dataset_type mpd --model transformer --shuffle True --e_pos False

python infer.py --dataset_type mpd --model transformer --shuffle True --e_pos False
```

## Acknowledgments
This project is based on the work done in [ply_title_gen](https://github.com/seungheondoh/ply_title_gen.git).
