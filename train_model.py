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
from src.model.transformer import Transformer  # Ensure this import matches your file and class names
from src.task.pipeline import PlyPipeline
import wandb
from tokenizers import Tokenizer, SentencePieceBPETokenizer
from src.utils import Vocab
from pytorch_lightning.callbacks import ProgressBar

class TokenizerWrapper:
    def __init__(self):
        tokenizer_dir = './dataset/tokenizer/'  # Use string path for consistency
        vocab_file = os.path.join(tokenizer_dir, 'mpd_bpe-vocab.json')
        merges_file = os.path.join(tokenizer_dir, 'mpd_bpe-merges.txt')
        self.tokenizer = SentencePieceBPETokenizer(vocab_file, merges_file)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

def get_wandb_logger():
    return WandbLogger(project="MPD-Model-Training", log_model="all")

def setup_callbacks():
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode='min',
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )
    return checkpoint_callback, early_stop_callback

def main(args):
    if args.reproduce:
        seed_everything(42)

    tokenizer = TokenizerWrapper()
    title_vocab = pickle.load(open(Path("./dataset/tokenizer/title_split/mpd_vocab.pkl"), "rb"))
    track_vocab = pickle.load(open(Path("./dataset/tokenizer/track/mpd_vocab.pkl"), "rb"))

    # Adjust these placeholder values as needed for your setup
    split_path = "./dataset/split/"
    dataset_type = "mpd"  # Or whatever your dataset type is
    context_length = 64  # This is a placeholder; set it based on your model's needs
    shuffle = True  # Adjust based on whether you want to shuffle the dataset during training

    # Initialize the appropriate model, as before
    model_cls = RNN_Attn if args.model == "rnn" else Transformer
    model = model_cls(
        input_size=len(track_vocab),  # Adjust based on actual model input requirements
        output_size=len(title_vocab),
        hidden_size=args.hidden_size,
        e_layers=args.num_layers,  # Assuming you want the same 'num_layers' for encoder and decoder
        d_layers=args.num_layers,  # You might adjust this if different
        heads=args.heads,
        pf_dim=args.pf_dim,
        dropout=args.dropout,
        e_pos=True,  # Assuming you want positional encoding; adjust based on your needs
    )

    logger = get_wandb_logger()
    checkpoint_callback, early_stop_callback = setup_callbacks()

    # Updated instantiation of PlyPipeline with all required arguments
    pipeline = PlyPipeline(
        split_path=split_path,
        dataset_type=dataset_type,
        tokenizer=tokenizer,  # Make sure this matches corrected argument name in PlyPipeline
        context_length=context_length,
        title_tokenizer=tokenizer,  # Assuming title_tokenizer and tokenizer can be the same
        title_vocab=title_vocab,
        song_vocab=track_vocab,
        track_vocab=track_vocab,  # Assuming you've updated PlyPipeline to accept track_vocab
        shuffle=shuffle,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    progress_bar = ProgressBar()

    trainer = Trainer(
    max_epochs=args.max_epochs,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
    deterministic=True
)

    trainer.fit(model, datamodule=pipeline)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "transformer"], default="rnn", help="Model type.")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads in multi-head attention.")
    parser.add_argument("--pf_dim", type=int, default=512, help="Dimension of the position-wise feedforward layer.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs.")
    parser.add_argument("--embed_size", type=int, default=256, help="Embedding size for the RNN model.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size for RNN/Transformer.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers.")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use.")
    parser.add_argument("--reproduce", action="store_true", help="Ensure reproducibility.")
    args = parser.parse_args()

    main(args)

