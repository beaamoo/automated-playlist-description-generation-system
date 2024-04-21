import os
import sys
from argparse import ArgumentParser, Namespace
from src.model.transformer import Transformer
from src.utils import Vocab

import numpy as np
import pickle
import torch
from tokenizers import Tokenizer, SentencePieceBPETokenizer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

import pandas as pd
import json
from tqdm import tqdm

from torch.autograd import Variable

class _tokenizer():
    def __init__(self):
        self.tokenizer_dir = './dataset/tokenizer/description_bpe'
        self.model = SentencePieceBPETokenizer(os.path.join(self.tokenizer_dir, "mpd-vocab.json"),\
                                            os.path.join(self.tokenizer_dir, "mpd-merges.txt"))
        
        self.encoder = self.model.get_vocab()
    def encode(self, target_str, is_pretokenized=False, add_special_tokens=True):
        return self.model.encode(target_str, pair=None, is_pretokenized=False, add_special_tokens=True).ids

    def decode(self, target_ids, skip_special_tokens=True):
        return self.model.decode(target_ids, skip_special_tokens=True)

def description_tokenize(text, context_length, description_vocab):
    token = ["<sos>"] + text.split() + ["<eos>"]
    all_tokens = [description_vocab.token_to_idx[i] for i in token]
    text = torch.zeros(context_length, dtype=torch.long)
    if len(all_tokens) < context_length:
        text[:len(all_tokens)] = torch.tensor(all_tokens)
    else:
        text[:context_length-1] = torch.tensor(all_tokens[:context_length-1])
        text[-1] = all_tokens[-1]
    return text

def song_tokenize(song, dataset_type, context_length, song_vocab, shuffle=False):
    song_token = ["<sos>"] + song + ["<eos>"]
    all_tokens = [song_vocab.token_to_idx[i] for i in song_token]
    song_seq = torch.zeros(context_length, dtype=torch.long)
    if len(all_tokens) < context_length:
        song_seq[:len(all_tokens)] = torch.tensor(all_tokens)
    else:
        song_seq[:context_length-1] = torch.tensor(all_tokens[:context_length-1])
        song_seq[-1] = all_tokens[-1]
    return song_seq

def decode_song(song_seq):
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    decoded_song_list = []
    song_dict={}
    data_dir = "./spotify_million_playlist_dataset/mpd/data"
    fname = os.listdir(data_dir)
    for file in tqdm(fname):
        if file.startswith("mpd.slice.") and file.endswith(".json"):
            one_playlist = json.load(open(os.path.join(data_dir,file),'r'))
            for ply in one_playlist['playlists']:
                if ply['tracks']==[] or ply['name']=='':
                    continue
                for track in ply['tracks']:
                    track_dict = {}
                    track_dict['description'] = track['track_name']
                    track_dict['artist'] = track['artist_name']
                    song_dict[track['track_uri']] = track_dict
                
    for sid in song_seq:
        song_idx = song_vocab.idx_to_token[sid]
        if song_idx not in special_tokens:
            decoded_song_list.append(song_dict[song_idx])
    return decoded_song_list


def _generation(dataset_type, model_type, songs, model, context_length, song_vocab, description_vocab, device):
    song_seq = song_tokenize(songs, dataset_type, context_length, song_vocab)
    sos_token = description_tokenize('', context_length, description_vocab)
    song_seq = song_seq.to(device)
    sos_token = sos_token.to(device)
    if model_type == 'rnn':
        song_seq = torch.unsqueeze(song_seq, 1)
        sos_token = torch.unsqueeze(sos_token, 1)[0,:]
    elif model_type == 'transfomer':
        song_seq = torch.unsqueeze(song_seq, 0)
        sos_token = torch.unsqueeze(sos_token, 0)[:,0]
    
    max_len=200
    attentions = torch.zeros(max_len, 1, context_length).to(device)
    with torch.no_grad():
        if model_type == 'rnn':
            encoder_output, hidden = model.encoder(song_seq)
        elif model_type == 'transfomer':
            src_mask = model.make_src_mask(song_seq)
            enc_src = model.encoder(song_seq, src_mask)

    #vocab_size = model.decoder.output_size
    if model_type == 'rnn':
        outputs = []
        hidden = hidden[:model.decoder.n_layers]
        output = Variable(sos_token)  # sos
    elif model_type == 'transfomer':
        trg_indexes = [sos_token]

    if model_type == 'rnn':
        for t in range(1, max_len):
            with torch.no_grad():
                output, hidden, attn_weights = model.decoder(output, hidden, encoder_output)
            pred_token = output.argmax(1).item()
            outputs.append(pred_token)
            top1 = output.data.max(1)[1]
            output = Variable(top1)
            if pred_token == 2:
                break
            attentions[t-1] = attn_weights
    elif model_type == 'transfomer':
        for t in range(1, max_len):
            with torch.no_grad():
                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
                trg_mask = model.make_trg_mask(trg_tensor)
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)
            if pred_token == 2:
                break
    
    if model_type == 'rnn':
        generative_description = [description_vocab.idx_to_token[i] for i in outputs]
        return generative_description, attentions[:len(generative_description)-1]
    elif model_type == 'transfomer':
        generative_description = [description_vocab.idx_to_token[i] for i in trg_indexes[1:]]
        return generative_description, attention.squeeze(0)[:,1:len(generative_description)]

def main(args):
    save_path = f"transformer_pt/{args.tokenzier}/s:{args.shuffle}_epos:{args.e_pos}"
    description_tokenizer = _tokenizer()
    song_vocab = pickle.load(open(os.path.join("./dataset/tokenizer/track", args.dataset_type + "_vocab.pkl"), mode="rb"))
    description_vocab = pickle.load(open(os.path.join("./dataset/tokenizer/description_split", args.dataset_type + "_vocab.pkl"), mode="rb"))
    if args.tokenzier == "white":
        input_size = len(song_vocab) 
        output_size= len(description_vocab)
    else:
        raise ValueError("Current model only support white space tokenizer")

    if args.model == "transfomer":
        model = Transformer(
            input_size = input_size, 
            output_size= output_size,
            hidden_size= args.embed_size,
            e_layers= args.e_layers, 
            d_layers= args.d_layers, 
            heads = args.heads,
            pf_dim = args.hidden_size,
            dropout= args.dropout, 
            e_pos = args.e_pos,
            device = args.gpus
        )

    device = f"cuda:{args.gpus}"
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"), map_location=torch.device(device))
    new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
    new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    fl = torch.load(os.path.join(args.split_path, args.dataset_type, "test.pt"))
    inference = {}
    counter = 0  # Add a counter
    for item in tqdm(fl):
        gen_t, _ = _generation(args.dataset_type, args.model, item['songs'], model, args.context_length, song_vocab, description_vocab, args.gpus)
        inference[item['pid']] = {
            "ground_truth": item['nrm_plylst_description'],
            "prediction": " ".join(gen_t)
        }
        counter += 1  # Increment the counter
        if counter >= 50:  # Break the loop if counter is 5 or more
            break

    with open(os.path.join(save_path, "inference.json"), mode="w", encoding='utf-8') as io:
        json.dump(inference, io, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--split_path", default="./dataset/split", type=str)
    parser.add_argument("--tid", default="0", type=str)
    parser.add_argument("--model", default="transfomer", type=str)
    parser.add_argument("--tokenzier", default="white", type=str)
    parser.add_argument("--context_length", default=64, type=int)
    parser.add_argument("--shuffle", default=True, type=bool)
    # model
    parser.add_argument("--embed_size", default=128, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--e_layers", default=3, type=int)
    parser.add_argument("--d_layers", default=3, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    parser.add_argument("--e_pos", default=False, type=bool)
    parser.add_argument("--d_pos", default=True, type=bool)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--pf_dim", default=256, type=int)

    parser.add_argument("--teacher_forcing_ratio", default=0.5, type=float)
    # pipeline
    parser.add_argument("--gpus", default=0, type=int)

    args = parser.parse_args()
    args.dataset_type = "mpd"  # Hardcode the dataset type to 'mpd'
    main(args)