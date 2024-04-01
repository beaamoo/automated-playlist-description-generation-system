import random
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PlyDataset(Dataset):
    def __init__(self, split_path, dataset_type, tokenizer, context_length, split, title_tokenizer, title_vocab, song_vocab, track_vocab, shuffle):
        # Initialization code here
        self.split_path = split_path
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.split = split
        self.title_tokenizer = title_tokenizer
        self.title_vocab = title_vocab
        self.song_vocab = song_vocab
        self.track_vocab = track_vocab  # Use track_vocab as needed within your dataset
        self.shuffle = shuffle

    def load_data(self):
        """
        Loads and optionally augments the data based on the split and shuffle flag.
        """
        data_path = os.path.join(self.split_path, self.dataset_type, f"{self.split.lower()}.pt")
        data = torch.load(data_path)

        if self.split == "TRAIN" and self.shuffle:
            augmented_data = self.augment_data(data)
            random.shuffle(augmented_data)
            return augmented_data

        return data

    def augment_data(self, data):
        """
        Doubles the dataset size by creating a shuffled version of each playlist.
        """
        augmented_data = []
        for instance in data:
            shuffled_instance = instance.copy()
            random.shuffle(shuffled_instance['songs'])
            augmented_data.append(instance)
            augmented_data.append(shuffled_instance)
        return augmented_data

    def __getitem__(self, index):
        """
        Returns the tokenized song sequence and title sequence for the given index.
        """
        instance = self.data[index]
        song_seq = self._tokenize_songs(instance['songs'])
        title_seq = self._tokenize_title(instance['nrm_plylst_title'])
        return song_seq, title_seq

    def _tokenize_title(self, text):
        """
        Tokenizes the playlist title based on the specified tokenizer.
        """
        if self.tokenizer == "bpe":
            tokens = [1] + self.title_tokenizer.encode(text) + [2]
        else:
            tokens = [self.title_vocab.get(token, self.title_vocab["<unk>"]) 
                      for token in ["<sos>"] + text.split() + ["<eos>"]]
        return self._pad_sequence(tokens)

    def _tokenize_songs(self, songs):
        """
        Tokenizes the song sequence.
        """
        tokens = [self.song_vocab.get(song, self.song_vocab["<unk>"]) 
                  for song in ["<sos>"] + songs + ["<eos>"]]
        return self._pad_sequence(tokens)

    def _pad_sequence(self, tokens):
        """
        Pads or truncates the token sequence to the specified context length.
        """
        padded_seq = torch.zeros(self.context_length, dtype=torch.long)
        effective_length = min(len(tokens), self.context_length)
        padded_seq[:effective_length] = torch.tensor(tokens[:effective_length])
        return padded_seq

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.data)
