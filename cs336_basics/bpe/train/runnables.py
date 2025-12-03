import regex as re

from cs336_basics.bpe.train.common import train
from cs336_basics.bpe.train.pretokenization import pretokenize

NON_WHITESPACE = re.compile(r"\S+")

def bpe_example(
    text: str,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"]
) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:

    special_tok_pat = re.compile("| ".join(re.escape(tok) for tok in special_tokens))
    chunks = re.split(special_tok_pat)
    