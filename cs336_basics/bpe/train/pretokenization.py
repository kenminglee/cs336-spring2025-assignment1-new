from collections import Counter
import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
from functools import partial

from cs336_basics import ROOT_DIR

# This function is important as it ensures we don't end up with boundaries that cut in the middle of a split_special_token
# which is bound to happen if we naively split the large file into equal chunks
# the following function is unchanged from https://github.com/stanford-cs336/assignment1-basics/blob/430e2c844e29f8aad8f9330e8706db9cb508241f/cs336_basics/pretokenization_example.py#L5
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

GPT2_PRE_TOKENIZER = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pretokenize_chunk(
    start:int,
    end:int,
    input_path: str | os.PathLike, 
    special_tokens: list[str],
    pretokenize_regex: re.Pattern[str] = GPT2_PRE_TOKENIZER
)->Counter[str]:
    special_tok_pat = re.compile("| ".join(re.escape(tok) for tok in special_tokens))
    pretokens:list[str] = []
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    minichunks = re.split(special_tok_pat, chunk)
    for minichunk in minichunks:
        pretokens.extend([match.group() for match in re.finditer(pretokenize_regex, minichunk)])
    return Counter(pretokens)


NON_WHITESPACE_PRE_TOKENIZER = re.compile(r"\S+")

# used for when we want to pretokenize a string (single process)
def pretokenize_str(
    text: str,
    special_tokens:list[str],
    pretokenize_regex: re.Pattern[str] = NON_WHITESPACE_PRE_TOKENIZER
)-> Counter[str]:
    special_tok_pat = re.compile("| ".join(re.escape(tok) for tok in special_tokens))
    chunks = re.split(special_tok_pat, text)
    pretokens:list[str] = []
    for chunk in chunks:
        pretokens.extend([match.group() for match in re.finditer(pretokenize_regex, chunk)])
    return Counter(pretokens)

# used for when we want to pretokenize a large file (multiple processes)
def pretokenize(
    input_path: str | os.PathLike, 
    desired_num_processes:int, 
    special_tokens: list[str],)->Counter[str]:
    with open(input_path, "rb") as f:
        special_tokens_bytes:list[bytes] = [s.encode("utf-8") for s in special_tokens]
        # In the case of multiple special tokens, find the best one to split on
        # where "best" is defined by the token that is most numerous and balanced so that
        # we get the closest number of boundaries for our desired_num_processes
        # e.g., if we want to split across 4 processes, the ideal number of boundaries is 4+1=5
        max_chunk_boundaries = []
        max_chunks_found = 0
        for special_tok in special_tokens_bytes:
            boundaries = find_chunk_boundaries(f, desired_num_processes, special_tok)
            if len(boundaries)==desired_num_processes+1: # perfect split token found, can break
                max_chunks_found = len(boundaries)-1
                max_chunk_boundaries = boundaries
                break
            elif len(boundaries)>max_chunks_found:
                max_chunks_found = len(boundaries)-1
                max_chunk_boundaries = boundaries
    print("Num processes for pretokenization -- Desired:", desired_num_processes, "Actual:", max_chunks_found)
    worker_fn = partial(pretokenize_chunk, input_path=input_path, special_tokens=special_tokens) 
    workers_arg = list(zip(max_chunk_boundaries[:-1], max_chunk_boundaries[1:]))
    with Pool(processes=max_chunks_found) as pool:
        results = pool.starmap(worker_fn, workers_arg)
    return sum(results, Counter())




# ## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token

if __name__=="__main__":
    num_processes = 32
    results = pretokenize(os.path.join(ROOT_DIR, "../tests/fixtures/corpus.en"), 32, ["<|endoftext|>"])
    # print(results)