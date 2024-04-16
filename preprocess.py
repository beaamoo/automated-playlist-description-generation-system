import re
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import os
import sys
import logging
from collections import Counter
from tokenizers import Tokenizer, SentencePieceBPETokenizer
import torch
import pickle
from src.utils import Vocab

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\[\]\/#!$%\^\*;:{}=\_`~()@<>]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def filter_description(description, songs, min_token_len=3, min_description_len=3, min_tracklist_len=10):
    n_tracks = len(songs)
    tokens = description.split(' ')
    mean_token_len = np.array([len(i) for i in tokens]).mean() if len(tokens) else 0
    
    if mean_token_len >= min_token_len and len(tokens) >= min_description_len and n_tracks >= min_tracklist_len:
        return True
    else:
        return False

def load_and_filter(dataset_name, dataset_dir, filtered_dir, min_token_len=3, min_description_len=3, min_tracklist_len=10):
    tqdm.pandas()

    if dataset_name != 'mpd':
        raise ValueError("Please insert correct dataset name: 'mpd'.")

    fname = os.listdir(dataset_dir)
    logger.info("--- MPD (Million Playlist Dataset)")

    dfs = []
    for file in tqdm(fname):
        if file.startswith("mpd.slice.") and file.endswith(".json"):
            file = json.load(open(os.path.join(dataset_dir,file),'r'))
            df = pd.DataFrame.from_dict(file['playlists'])
            df = df[df['description'].map(lambda r: len(r.split(' ')) > 0 if pd.notna(r) and r != '' else False)]
            df['songs'] = df['tracks'].map(lambda tracks: [track['track_uri'] for track in tracks if len(tracks)>0])
            df['artists'] = df['tracks'].map(lambda tracks: list(set([track['artist_uri'] for track in tracks])))
            df = df.rename(columns={'description': 'plylst_description'})
            dfs.append(df[['pid', 'plylst_description', 'songs', 'artists']])
    playlist = pd.concat(dfs)
    logger.info("got {} playlists in total.".format(len(playlist)))

    logger.info("Normalize Playlist Description...")
    playlist['nrm_plylst_description'] = playlist['plylst_description'].progress_map(normalize_name)
    logger.info("Filter Playlists...")
    filtered_playlist = playlist[playlist.progress_apply(lambda row: filter_description(row['nrm_plylst_description'], row['songs'],\
                                                            min_token_len, min_description_len, min_tracklist_len), axis=1)]
    logger.info("{} playlists are retrieved.".format(len(filtered_playlist)))

    filtered_playlist_dict = filtered_playlist.to_dict(orient='records')
    torch.save(filtered_playlist_dict, os.path.join(filtered_dir, dataset_name+'_filtered.pt'))
    logger.info("Filtered MPD Dataset Saved.")

def data_split(filtered_dir, split_dir, ratio=[0.8, 0.1, 0.1]):
    data_dict = torch.load(filtered_dir)
    df = pd.DataFrame.from_dict(data_dict)

    if not len(ratio)==3 or any(r < 0. or r > 1. for r in ratio) or round(sum(ratio), 5) != 1:
        raise ValueError('Ratio should be three values between 0 and 1, summing to 1.')
    
    min_description_len = df['nrm_plylst_description'].apply(lambda r: len(r.split(' ')) if r else 0).min()
    max_description_len = df['nrm_plylst_description'].apply(lambda r: len(r.split(' ')) if r else 0).max()
    if min_description_len == 0:
        raise Exception('Playlist with no description exists!')

    dfs = {'train': [], 'val': [], 'test':[]}
    for description_len in range(min_description_len, max_description_len+1):
        uni_len_df = df[df['nrm_plylst_description'].map(lambda r: len(r.split(' '))==description_len)]
        train, validate, test = np.split(uni_len_df.sample(frac=1, random_state=33),
                                          [int(ratio[0]*len(uni_len_df)), int((ratio[0]+ratio[1])*len(uni_len_df))])
        dfs['train'].append(train)
        dfs['val'].append(validate)
        dfs['test'].append(test)
    
    for name in dfs:
        merged_dataset = pd.concat(dfs[name])
        merged_dataset_dict = merged_dataset.to_dict(orient='records')
        torch.save(merged_dataset_dict, os.path.join(split_dir, name+'.pt'))
        logger.info("Filtered {} Dataset Saved: total {} playlists.".format(name.upper(), len(merged_dataset)))

def byte_level_BPE_train(train_dir, val_dir, out_dir, out_name, limit_alphabet, vocab_size=1500):
    train_dict = torch.load(train_dir)
    train_df = pd.DataFrame.from_dict(train_dict)
    val_dict = torch.load(val_dir)
    val_df = pd.DataFrame.from_dict(val_dict)
    df = pd.concat([train_df, val_df], axis=0)

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train_from_iterator(
        df['nrm_plylst_description'],
        vocab_size=vocab_size,
        min_frequency=2,
        limit_alphabet=limit_alphabet,
        show_progress=True,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )
    tokenizer.save_model(directory=out_dir, prefix=out_name)
    logger.info("BPE model trained & saved.")

def build_dictionary(dataset_dir, track_out_dir, song_out_dir, out_name):
    data_dict = torch.load(dataset_dir)
    song_list, token_list = [], []
    for instance in data_dict:
        song_list.extend(instance['songs'])
        token_list.extend(instance['nrm_plylst_description'].split())
    s_vocab = Vocab(list_of_tokens=list(Counter(song_list).keys()))
    t_vocab = Vocab(list_of_tokens=list(Counter(token_list).keys()))

    with open(os.path.join(track_out_dir, out_name + "_vocab.pkl"), mode="wb") as io:
        pickle.dump(t_vocab, io)
    
    with open(os.path.join(song_out_dir, out_name + "_vocab.pkl"), mode="wb") as io:
        pickle.dump(s_vocab, io)

def ensure_directories_exist(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

if __name__ == '__main__':
    # Ensure directories exist, and make sure spotify_million_playlist_dataset is downloaded and extracted in root directory
    directories = [
        "./dataset/split/",
        "./dataset/tokenizer/",
        "./dataset/tokenizer/description_split",
        './dataset/tokenizer/track',
        "./dataset/tokenizer/description_bpe"
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    logger.info("----- LOAD AND FILTER DATASETS -----")
    load_and_filter('mpd', "./spotify_million_playlist_dataset/data", "./dataset/split/",\
                    min_token_len=3, min_description_len=3, min_tracklist_len=10)

    logger.info("----- SPLIT DATASETS (TR/VA/TE) -----")
    data_split("./dataset/split/mpd_filtered.pt", "./dataset/split/", ratio=[0.8, 0.1, 0.1])
    
    logger.info("----- BUILD TRACK DICTIONARY -----")
    build_dictionary("./dataset/split/mpd_filtered.pt", './dataset/tokenizer/description_split', './dataset/tokenizer/track', 'mpd')

    logger.info("----- TRAIN SENTENCE LEVEL BPE -----")
    byte_level_BPE_train('./dataset/split/train.pt', './dataset/split/val.pt',\
                         './dataset/tokenizer/description_bpe', 'mpd', vocab_size=1500, limit_alphabet=600)