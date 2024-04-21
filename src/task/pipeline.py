import pickle
from typing import Callable, Optional
import os
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from ..data import Ply_Dataset


class PlyPipeline(LightningDataModule):
    def __init__(self, split_path, dataset_type, tokenzier, context_length, description_tokenizer, description_vocab, song_vocab, shuffle, batch_size, num_workers) -> None:
        super(PlyPipeline, self).__init__()
        self.dataset_builder = Ply_Dataset
        self.split_path = split_path
        self.dataset_type = dataset_type
        self.tokenzier = tokenzier
        self.context_length = context_length
        self.description_tokenizer = description_tokenizer
        self.description_vocab = description_vocab
        self.song_vocab = song_vocab
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = PlyPipeline.get_dataset(
                self.dataset_builder,
                self.split_path,
                self.dataset_type,
                self.tokenzier,
                self.context_length,
                "TRAIN",
                self.description_tokenizer,
                self.description_vocab,
                self.song_vocab,
                self.shuffle
            )

            self.val_dataset = PlyPipeline.get_dataset(
                self.dataset_builder,
                self.split_path,
                self.dataset_type,
                self.tokenzier,
                self.context_length,
                "VALID",
                self.description_tokenizer,
                self.description_vocab,
                self.song_vocab,
                self.shuffle
            )

        if stage == "test" or stage is None:
            self.test_dataset = PlyPipeline.get_dataset(
                self.dataset_builder,
                self.split_path,
                self.dataset_type,
                self.tokenzier,
                self.context_length,
                "TEST",
                self.description_tokenizer,
                self.description_vocab,
                self.song_vocab,
                False,
            )

    def train_dataloader(self) -> DataLoader:
        return PlyPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return PlyPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return PlyPipeline.get_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, split_path, dataset_type, tokenzier, context_length, split, description_tokenizer, description_vocab, song_vocab, shuffle) -> Dataset:
        dataset = dataset_builder(split_path, dataset_type, tokenzier, context_length, split, description_tokenizer, description_vocab, song_vocab, shuffle)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)