import re
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import os
import sys
import logging
import torch
import pickle
from src.utils import Vocab
from collections import Counter
from tokenizers import Tokenizer, SentencePieceBPETokenizer

# Function to safely create a directory
def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\[\]\/#!$%\^\*;:{}=\_`~()@<>]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def filter_title(title, songs, min_token_len=3, min_title_len=3, min_tracklist_len=10):
    n_tracks = len(songs)
    tokens = title.split(' ')
    mean_token_len = np.array([len(i) for i in tokens]).mean() if len(tokens) else 0
    
    if mean_token_len >= min_token_len and len(tokens) >= min_title_len and n_tracks >= min_tracklist_len:
        return True
    else:
        return False

def load_and_filter(dataset_dir, filtered_dir, min_token_len=3, min_title_len=3, min_tracklist_len=10):
    tqdm.pandas()
    
    ensure_dir(filtered_dir)# Before processing, ensure the filtered_dir exists

    fname = os.listdir(dataset_dir)
    dfs = []
    for file in tqdm(fname):
        if file.startswith("mpd.slice.") and file.endswith(".json"):
            file = json.load(open(os.path.join(dataset_dir,file),'r'))
            df = pd.DataFrame.from_dict(file['playlists'])
            df = df[df['name'].map(lambda r: len(r.split(' '))>0 if not r=='' else False)]
            df['songs'] = df['tracks'].map(lambda tracks: [track['track_uri'] for track in tracks if len(tracks)>0])
            df['artists'] = df['tracks'].map(lambda tracks: list(set([track['artist_uri'] for track in tracks])))
            df = df.rename(columns={'name': 'plylst_title'})
            dfs.append(df[['pid', 'plylst_title', 'songs', 'artists']])
    playlist = pd.concat(dfs)
    logger.info("got {} playlists in total.".format(len(playlist)))
    
    logger.info("Normalize Playlist Title...")
    playlist['nrm_plylst_title'] = playlist['plylst_title'].progress_map(normalize_name)
    logger.info("Filter Playlists...")
    filtered_playlist = playlist[playlist.progress_apply(lambda row: filter_title(row['nrm_plylst_title'], row['songs'],\
                                                                min_token_len, min_title_len, min_tracklist_len), axis=1)]
    logger.info("{} playlists are retrieved.".format(len(filtered_playlist)))

    filtered_playlist_dict = filtered_playlist.to_dict(orient='records')
    torch.save(filtered_playlist_dict, os.path.join(filtered_dir, 'mpd_filtered.pt'))
    logger.info("Filtered MPD Dataset Saved.")

def data_split(filtered_dir, split_dir, ratio=[0.8, 0.1, 0.1]):
    ensure_dir(split_dir)# Before processing, ensure the split_dir exists
    # df = pd.read_csv(filtered_dir)
    data_dict = torch.load(filtered_dir)
    df = pd.DataFrame.from_dict(data_dict)
    if not len(ratio)==3:
        raise ValueError('Insert ''3'' ratio values for train/val/test dataset.')
    if any(r < 0. or r > 1. for r in ratio) or round(sum(ratio), 5) != 1:
        raise ValueError('Ratio should be values btw 0 and 1, and its sum should be 1.')
    
    min_title_len = df['nrm_plylst_title'].apply(lambda r: len(r.split(' ')) if r!=None else 0).min()
    max_title_len = df['nrm_plylst_title'].apply(lambda r: len(r.split(' ')) if r!=None else 0).max()
    if min_title_len == 0:
        raise Exception('Playlist w/ no title exists!')

    dfs = {'train': [], 'val': [], 'test':[]}
    for title_len in range(min_title_len, max_title_len+1):
        uni_len_df = df[df['nrm_plylst_title'].map(lambda r: len(r.split(' '))==title_len)]

        train, validate, test = np.split(uni_len_df.sample(frac=1, random_state=33),
                                                [int(ratio[0]*len(uni_len_df)), int((ratio[0]+ratio[1])*len(uni_len_df))])
        dfs['train'].append(train)
        dfs['val'].append(validate)
        dfs['test'].append(test)
    
    for name in dfs:
        merged_dataset = pd.concat(dfs[name])

        merged_dataset_dict = merged_dataset.to_dict(orient='records')
        torch.save(merged_dataset_dict, os.path.join(split_dir, name+'.pt'))
        # merged_dataset.to_csv(os.path.join(split_dir, name+'.csv'), index=False)
        logger.info("Filtered {} Dataset Saved: total {} playlists.".format(name.upper(), len(merged_dataset)))

def byte_level_BPE_train(train_dir, val_dir, out_dir, out_name, limit_alphabet=1000, vocab_size=10000):
    # Ensure the output directory exists
    ensure_dir(out_dir)
    
    # Load Dataset
    train_dict = torch.load(train_dir)
    train_df = pd.DataFrame.from_dict(train_dict)
    val_dict = torch.load(val_dir)
    val_df = pd.DataFrame.from_dict(val_dict)
    df = pd.concat([train_df, val_df], axis=0)

    # Initialize an empty tokenizer
    tokenizer = SentencePieceBPETokenizer()

    # Train the tokenizer
    tokenizer.train_from_iterator(
        df['nrm_plylst_title'].tolist(),
        vocab_size=vocab_size,
        min_frequency=2,
        limit_alphabet=limit_alphabet,
        show_progress=True,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )

    # Save the tokenizer model
    tokenizer.save_model(directory=out_dir, prefix=out_name)
    logger.info("BPE model trained & saved.")

def build_dictionary(dataset_dir, track_out_dir, song_out_dir, out_name):
    ensure_dir(track_out_dir)
    ensure_dir(song_out_dir)
    data_dict = torch.load(dataset_dir)
    song_list = []
    token_list = []
    for instance in data_dict:
        song_list.extend(instance['songs'])
    for instance in data_dict:
        token_list.extend(instance['nrm_plylst_title'].split())
    s_counter = Counter(song_list)
    t_counter = Counter(token_list)

    s_vocab = Vocab(list_of_tokens=list(s_counter.keys()))
    t_vocab = Vocab(list_of_tokens=list(t_counter.keys()))

    with open(os.path.join(track_out_dir, out_name + "_vocab.pkl"), mode="wb") as io:
        pickle.dump(t_vocab, io)
    
    with open(os.path.join(song_out_dir, out_name + "_vocab.pkl"), mode="wb") as io:
        pickle.dump(s_vocab, io)
    

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # Directory setup
    dataset_dir = "./spotify_million_playlist_dataset/data"
    filtered_dir = "./dataset/"
    split_dir = "./dataset/split/"
    tokenizer_dir = './dataset/tokenizer'
    track_out_dir = tokenizer_dir + '/title_split'
    song_out_dir = tokenizer_dir + '/track'

    # Ensure directories are created
    ensure_dir(dataset_dir)
    ensure_dir(filtered_dir)
    ensure_dir(split_dir)
    ensure_dir(tokenizer_dir)
    ensure_dir(track_out_dir)
    ensure_dir(song_out_dir)

    logger.info("----- LOAD AND FILTER MPD DATASET -----")
    load_and_filter(dataset_dir, filtered_dir, min_token_len=3, min_title_len=3, min_tracklist_len=10)

    # Adjust the paths according to your data structure
    train_dir = os.path.join(split_dir, 'train.pt')
    val_dir = os.path.join(split_dir, 'val.pt')

    logger.info("----- TRAIN BYTE-LEVEL BPE TOKENIZER -----")
    byte_level_BPE_train(train_dir, val_dir, tokenizer_dir, 'mpd_bpe', limit_alphabet=1000, vocab_size=10000)

    logger.info("----- SPLIT DATASETS (TR/VA/TE) -----")
    data_split(os.path.join(filtered_dir, 'mpd_filtered.pt'), split_dir, ratio=[0.8, 0.1, 0.1])
    
    logger.info("----- BUILD TRACK DICTIONARY -----")
    build_dictionary(os.path.join(filtered_dir, 'mpd_filtered.pt'), track_out_dir, song_out_dir, 'mpd')
