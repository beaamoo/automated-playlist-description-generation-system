from argparse import ArgumentParser
import os
from pathlib import Path
import pickle
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.model.rnn import RNN_Attn
from src.model.transformer import Transformer
from src.task.pipeline import PlyPipeline
from src.utils import Vocab
import wandb

def setup_tokenizer():
    """Initializes the tokenizer for the MPD dataset."""
    tokenizer_dir = Path('./dataset/tokenizer/title_bpe')
    vocab_path = tokenizer_dir / "mpd-vocab.json"
    merges_path = tokenizer_dir / "mpd-merges.txt"
    return SentencePieceBPETokenizer(vocab_path, merges_path)

def get_logger_and_callbacks(args):
    """Sets up WandbLogger, ModelCheckpoint, and EarlyStopping based on the provided arguments."""
    wandb_logger = WandbLogger()
    checkpoint_dir = Path(f"exp/mpd/{args.model}")
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename="best", save_top_k=1, monitor="val_loss", mode='min')
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=20, mode="min")
    return wandb_logger, checkpoint_callback, early_stop_callback

def main():
    """Main training function."""
    parser = ArgumentParser()
    parser.add_argument("--model", default="rnn", choices=["rnn", "transformer"], help="Model to use.")
    parser.add_argument("--max_epochs", default=100, type=int, help="Maximum number of epochs.")
    parser.add_argument("--gpus", default=0, type=int, help="Number of GPUs to use.")
    parser.add_argument("--reproduce", action='store_true', help="Ensure reproducibility.")
    args = parser.parse_args()

    if args.reproduce:
        seed_everything(42)
    
    wandb.init(project="MPD-Model-Training")

    # Loading vocab and initializing tokenizer specifically for the MPD dataset
    vocab = pickle.load(open("./dataset/tokenizer/mpd_vocab.pkl", "rb"))
    tokenizer = setup_tokenizer()

    # Model selection based on command line argument
    model_cls = RNN_Attn if args.model == "rnn" else Transformer
    model = model_cls(input_size=len(vocab), output_size=len(vocab))

    logger, checkpoint_callback, early_stop_callback = get_logger_and_callbacks(args)

    # Data pipeline for MPD dataset
    pipeline = PlyPipeline(tokenizer=tokenizer, vocab=vocab, dataset_type="mpd")

    trainer = Trainer(max_epochs=args.max_epochs, gpus=args.gpus, logger=logger, callbacks=[early_stop_callback, checkpoint_callback], deterministic=True)

    trainer.fit(model, datamodule=pipeline)

if __name__ == "__main__":
    main()
