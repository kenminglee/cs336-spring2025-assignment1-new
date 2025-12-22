from dataclasses import dataclass
import os
from typing import Literal
import random
import time

import torch
import numpy as np
from jaxtyping import Int 
import tyro
import wandb
from tqdm import tqdm

from cs336_basics import ROOT_DIR
from cs336_basics.bpe.tokenization import Tokenizer
from cs336_basics.nn import TransformerLM, cross_entropy_loss, AdamW, load_checkpoint, save_checkpoint, cosine_lr_schedule, generate_text, nucleus_sampling, clip_gradient


def tokenize_dataset(
        dataset_path:str, 
        tokenizer: Tokenizer, 
        tokenized_dataset_path: str, # where to save tokenized dataset to
        split_token:bytes=b"<|endoftext|>", 
        chunk_size:int=4096
    ):
    def find_document_boundaries():
        with open(dataset_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            pos = 0
            document_boundaries = [0]
            while pos < file_size:
                f.seek(pos)
                chunk = f.read(chunk_size)
                found_at = chunk.find(split_token)
                if found_at!=-1:
                    document_boundaries.append(pos+found_at+len(split_token)+1)
                    pos = document_boundaries[-1]
                else:
                    pos += chunk_size
        return document_boundaries


    doc_boundaries = find_document_boundaries()
    num_tokens = 0
    with open(dataset_path, "rb") as f:
        for index in tqdm(range(len(doc_boundaries)-1), desc="Counting num tokens in dataset"):
            f.seek(doc_boundaries[index])
            raw_bytes = f.read(doc_boundaries[index+1]-doc_boundaries[index])
            num_tokens += len(tokenizer.encode(raw_bytes.decode("utf-8", errors="ignore")))
    
    tokens_mm = np.memmap(
        tokenized_dataset_path,
        dtype=np.uint16,
        mode="w+",
        shape=(num_tokens,)
    )
    idx = 0
    with open(dataset_path, "rb") as f:
        for index in tqdm(range(len(doc_boundaries)-1), desc="tokenizing to file"):
            f.seek(doc_boundaries[index])
            raw_bytes = f.read(doc_boundaries[index+1]-doc_boundaries[index])
            tokenIDs = tokenizer.encode(raw_bytes.decode("utf-8", errors="ignore"))
            tokens_mm[idx:idx+len(tokenIDs)] = tokenIDs
            idx += len(tokenIDs)
    tokens_mm.flush()
    return tokens_mm

def sample_batch(
    dataset: Int[np.ndarray, " num_tokens "], 
    batch_size: int, 
    context_length: int, 
    device: str,
    rng: np.random.Generator | None = None
) -> tuple[
    Int[torch.Tensor, "batch_size context_length"], 
    Int[torch.Tensor, "batch_size context_length"]
]:
    
    if rng is None:
        rng = np.random.default_rng()
    starting_indices = rng.choice(len(dataset)-context_length, size=batch_size)
    offsets = np.arange(context_length)
    data_in = torch.tensor(dataset[starting_indices[:,None]+offsets], device=device, dtype=torch.long)
    data_out = torch.tensor(dataset[starting_indices[:,None]+(offsets+1)], device=device, dtype=torch.long)
    return data_in, data_out


@dataclass
class Args:
    """
    Arguments for LM training
    """
    seed: int = 0
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, use CPU"""

    resume: bool = False
    """if toggled, will resume training"""
    checkpoint_path: str | None = None
    """[Only used when resume is true] Path to .pt or .pth file"""
    

    dataset: Literal["TinyStories", "OpenWebText", "TinyStories-Sample-Small", "TinyStories-Sample-5M"] = "TinyStories-Sample-Small"
    """Dataset, ranked from smallest to largest: TinyStories-Sample-Small, TinyStories-Sample-5M, TinyStories, OpenWebText"""
    wandb_project_name: str = "cs336-a1"
    """the wandb's project name"""
    run_name: str | None = None
    """Optional run name; ignored when resuming a run"""
    notes: str | None = None
    """Optional notes on this run (logged to WandB); ignored when resuming a run"""

    context_length: int = 256
    d_model : int = 512
    d_ff: int = 1344
    """Dimension of feed-forward MLP"""
    theta: int = 10000
    """RoPE theta parameter"""
    num_layers: int = 4
    """Number of prenorm-transformer block"""
    num_heads: int = 16
    """Number of transformer heads"""

    lr: float = 1e-3
    """Overriden if cosine-lr-schedule is used"""
    weight_decay: float = 0.01
    beta_1: float = 0.9
    beta_2: float = 0.999

    num_checkpoints: int = 5
    save_path: str = os.path.join(ROOT_DIR, "../runs")
    batch_size: int = 64
    training_steps: int = 1500
    """Number of training steps. Each training step involves batch_size x context_length number of tokens"""
    log_freq: int = 5
    """How often to log metrics to WandB (steps/log)"""
    param_log_freq: int = 50
    """How often to log gradients and weights to WandB (steps/log)"""

    cosine_lr_schedule: bool = False
    """Toggle to use Cosine LR schedule"""
    max_lr: float = 1.0
    """[Only used when cosine_lr_schedule is true] Max LR """
    min_lr: float = 1e-5
    """[Only used when cosine_lr_schedule is true] Min LR """
    warmup_iters: int = 100
    """[Only used when cosine_lr_schedule is true] Number of warmup iterations to take to get to Max LR """
    cosine_cycle_iters: int = 1000
    """[Only used when cosine_lr_schedule is true] Number of iterations to use cosine annealing from Max LR to Min LR. After cosine_cycle_iters, go back to only Min LR """

    softmax_temp: float = 1.0
    """temperature for softmax; only used when generating text during validation"""
    top_p: float = 0.9
    """P value for top-P sampling; only used when generating text during validation. Must be between 0 and 1"""

    max_gradient_norm: float = -1
    """Clip norm of gradient to this value. No clipping when set to -1."""



def train(args: Args):
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rng = np.random.default_rng(args.seed)
    valid_rng = np.random.default_rng(args.seed)
    torch_gen = torch.Generator(device=args.device)
    torch_gen.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = args.torch_deterministic


    if "TinyStories" in args.dataset:
        vocab_path = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-train-vocab.json")
        merges_path = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-train-merges.txt")
        valid_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-valid.txt")
        tokenized_valid_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStories-valid.bin")
        if args.dataset=="TinyStories-Sample-Small":
            train_dataset_path = os.path.join(ROOT_DIR, "../tests/fixtures/tinystories_sample.txt")
            tokenized_train_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStories-Sample-Small.bin")
        elif args.dataset=="TinyStories-Sample-5M":
            train_dataset_path = os.path.join(ROOT_DIR, "../tests/fixtures/tinystories_sample_5M.txt")
            tokenized_train_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStories-Sample-5M.bin")
        else:
            train_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-train.txt")
            tokenized_train_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStories-train.bin")
    else:
        vocab_path = os.path.join(ROOT_DIR, "../data/owt-train-vocab.json")
        merges_path = os.path.join(ROOT_DIR, "../data/owt-train-merges.txt")
        train_dataset_path = os.path.join(ROOT_DIR, "../data/owt_train.txt")
        valid_dataset_path = os.path.join(ROOT_DIR, "../data/owt_valid.txt")
        tokenized_train_dataset_path = os.path.join(ROOT_DIR, "../data/owt-train.bin")
        tokenized_valid_dataset_path = os.path.join(ROOT_DIR, "../data/owt-valid.bin")

    assert os.path.isfile(vocab_path) and os.path.isfile(merges_path), "Vocab or merges file not found! Please run cs336_basics/bpe/train.py to train a BPE."
    assert os.path.isfile(train_dataset_path) and os.path.isfile(valid_dataset_path), "Training/Validation dataset not found! Please follow the instructions in Readme.md to download the datasets."

    tokenizer = Tokenizer.from_files(
        vocab_path, merges_path, ["<|endoftext|>"]
    )
    
    if not os.path.isfile(tokenized_train_dataset_path):
        print("Tokenized training dataset not found! tokenizing dataset now...")
        train_data = tokenize_dataset(train_dataset_path, tokenizer, tokenized_train_dataset_path)
    else:
        train_data = np.memmap(
            tokenized_train_dataset_path,
            dtype=np.uint16,
            mode="r"
        )
    
    if not os.path.isfile(tokenized_valid_dataset_path):
        print("Tokenized validation dataset not found! tokenizing dataset now...")
        validation_data = tokenize_dataset(valid_dataset_path, tokenizer, tokenized_valid_dataset_path)
    else:
        validation_data = np.memmap(
            tokenized_valid_dataset_path,
            dtype=np.uint16,
            mode="r"
        )
    
    model = TransformerLM(
        vocab_size = len(tokenizer._vocab),
        context_length = args.context_length,
        num_layers = args.num_layers,
        d_model = args.d_model,
        num_heads = args.num_heads,
        rope_theta = args.theta,
        d_ff = args.d_ff,
        device=args.device,
        dtype = torch.float32
    )
    optimizer = AdamW(
        model.parameters(), 
        lr:=args.lr, 
        args.weight_decay, 
        (args.beta_1, args.beta_2)
    )
    if args.resume:
        assert args.checkpoint_path is not None, "To resume a training run, please pass --checkpoint-path flag"
        assert os.path.isfile(args.checkpoint_path), f'"{args.checkpoint_path}" is not a valid filepath!'
        info = {}
        print(f'Resuming training run from "{args.checkpoint_path}"... ')
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer, info)
        run_name = info["run_name"]
        save_path = os.path.join(args.save_path, run_name)
        run = wandb.init(
            id=info["wandb_id"], 
            resume="must", 
            dir=os.path.join(save_path, "wandb"),
            resume_from=f"{info['wandb_id']}?_step={start_step}"
        )
    else:
        if args.run_name is None:
            run_name = f"{args.dataset.lower()}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
        else:
            run_name = f"{args.run_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
        start_step = 1
        save_path = os.path.join(args.save_path, run_name)
        run = wandb.init(
            project=args.wandb_project_name,
            dir=os.path.join(save_path, "wandb"),
            config=vars(args),
            group=args.dataset,
            name=run_name,
            save_code=False,
            notes=args.notes
        )
    # model.compile()
    checkpoint = args.training_steps // args.num_checkpoints
    run.watch(model, log="all", log_freq=args.param_log_freq)
    validation_table = wandb.Table(columns=["step","prompt", "prediction"], log_mode="INCREMENTAL")
    for step in tqdm(range(start_step, args.training_steps+1), desc="Training step"):
        if args.cosine_lr_schedule:
            lr = cosine_lr_schedule(step, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
            optimizer.update_lr(lr)

        x, y = sample_batch(train_data, args.batch_size, args.context_length, args.device, rng=train_rng)
        pred = model(x)
        loss = cross_entropy_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        if args.max_gradient_norm!=-1:
            clip_gradient(model.parameters(), args.max_gradient_norm)
        optimizer.step()

        if step%args.log_freq==0:
            run.log({"train/loss": loss.detach().cpu(), "train/lr":lr}, step=step)

        if step%checkpoint==0:
            with torch.no_grad():
                x, y = sample_batch(validation_data, args.batch_size, args.context_length, args.device, rng=valid_rng)
                valid_loss = cross_entropy_loss(model(x), y)

                text = generate_text("Once upon a time", tokenizer, model, torch_gen, max_num_tokens=200, device=args.device, sampling_fn=nucleus_sampling(args.top_p), softmax_temperature=args.softmax_temp)
                validation_table.add_data(step, "Once upon a time", text)

                run.log({"valid/loss": valid_loss.detach().cpu(), "valid/generation":validation_table}, step=step)

            save_checkpoint(model, optimizer, step, os.path.join(save_path, f"model_{step}.pt"), wandb_id=run.id, run_name=run_name)

        
if __name__=="__main__":
    args = tyro.cli(Args)
    train(args)


