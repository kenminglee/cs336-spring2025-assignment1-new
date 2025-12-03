import regex as re
import os
from collections import Counter
import time

from cs336_basics.bpe.serialization import write_vocab_to_file, write_merges_to_file
from cs336_basics.bpe.train.common import train
from cs336_basics.bpe.train.pretokenization import pretokenize, pretokenize_str, NON_WHITESPACE_PRE_TOKENIZER
from cs336_basics import ROOT_DIR


def bpe_example(
    text: str,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"]
) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:
    pretokens: Counter[str] = pretokenize_str(text, special_tokens, pretokenize_regex=NON_WHITESPACE_PRE_TOKENIZER)
    return train(pretokens, vocab_size, special_tokens)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_processes:int = 4
) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:
    pretokens: Counter[str] = pretokenize(input_path, num_processes, special_tokens)
    return train(pretokens, vocab_size, special_tokens)

def train_bpe_tinystories(
    num_processes:int = 32,
    output_dir:str=os.path.join(ROOT_DIR, "../data")
):
    start_time = time.time()
    fp = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-train.txt")
    assert os.path.isfile(fp), f"{fp} does not exist!"
    vocab, merges = train_bpe(fp, 10000, special_tokens=["<|endoftext|>"], num_processes=num_processes)
    elapsed_time = time.time()-start_time
    print(f"Finished training on TinyStories dataset, spent {int(elapsed_time // 60)} minutes and {elapsed_time % 60} seconds, now saving to disk...")
    write_vocab_to_file(vocab, os.path.join(output_dir, "TinyStoriesV2-GPT4-train-vocab.json"))
    write_merges_to_file(merges, os.path.join(output_dir, "TinyStoriesV2-GPT4-train-merges.txt"))

def train_bpe_owt(
    num_processes:int = 32,
    output_dir:str=os.path.join(ROOT_DIR, "../data")
):
    start_time = time.time()
    fp = os.path.join(ROOT_DIR, "../data/owt_train.txt")
    assert os.path.isfile(fp), f"{fp} does not exist!"
    vocab, merges = train_bpe(fp, 32000, special_tokens=["<|endoftext|>"], num_processes=num_processes)
    elapsed_time = time.time()-start_time
    print(f"Finished training on owt dataset, spent {int(elapsed_time // 60)} minutes and {elapsed_time % 60} seconds, now saving to disk...")
    write_vocab_to_file(vocab, os.path.join(output_dir, "owt-train-vocab.json"))
    write_merges_to_file(merges, os.path.join(output_dir, "owt-train-merges.txt"))

if __name__=="__main__":
    print("Running bpe_example test:")
    vocab, merge = bpe_example(
        '''
        low low low low low
        lower lower widest widest widest
        newest newest newest newest newest newest
        ''',
        257+6,
        special_tokens = ["<|endoftext|>"]
    )
    assert len(vocab.items()&{257: b'st', 258: b'est', 259: b'ow', 260: b'low', 261: b'west', 262: b'ne'}.items())==6, "Vocab does not match!"
    assert merge==[(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est'), (b'n', b'e')]
    print("bpe_example test passed!")

    print('Please run "uv run pytest tests/test_train_bpe.py" to test train_bpe')

    # print("Training BPE on TinyStories")
    # train_bpe_tinystories()

    # print("Training BPE on owt")
    # train_bpe_owt()
    
