import regex as re
from collections import Counter

from cs336_basics.bpe.train.common import train
from cs336_basics.bpe.train.pretokenization import pretokenize, pretokenize_str, NON_WHITESPACE_PRE_TOKENIZER


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