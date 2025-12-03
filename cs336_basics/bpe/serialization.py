import os
from functools import lru_cache
from pathlib import Path
import json

# Function is obtained from tests/common.py
@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def write_merges_to_file(merges: list[tuple[bytes, bytes]], txt_filepath:str):
    fp = Path(txt_filepath)
    assert fp.suffix == ".txt", "filepath must be a valid path to a .txt file"
    fp.parent.mkdir(parents=True, exist_ok=True) # automatically create all intermediate dir if not exist
    
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    lines = [
        f'{"".join([gpt2_byte_encoder[i] for i in merge_token_1])} {"".join([gpt2_byte_encoder[i] for i in merge_token_2])}\n'
        for merge_token_1, merge_token_2 in merges
    ]

    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

def write_vocab_to_file(vocab: dict[int, bytes], json_filepath:str):
    fp = Path(json_filepath)
    assert fp.suffix == ".json", "filepath must be a valid path to a .json file"
    fp.parent.mkdir(parents=True, exist_ok=True) # automatically create all intermediate dir if not exist
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    data = {"".join([gpt2_byte_encoder[b] for b in token]):id_ for id_, token in vocab.items()}

    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# function obtained largely from tests/test_train_bpe.py
def read_merges_from_file(txt_filepath:str)->list[tuple[bytes,bytes]]:
    assert os.path.isfile(txt_filepath), f"Error reading merges: {txt_filepath} does not exist!"

    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(txt_filepath, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    return merges

# function obtained largely from tests/test_train_bpe.py
def read_vocab_from_file(json_filepath:str) -> dict[int, bytes]:
    assert os.path.isfile(json_filepath), f"Error reading merges: {json_filepath} does not exist!"
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(json_filepath, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    return vocab

if __name__=="__main__":
    print("Testing read/write merges")
    test_merges = [(b"a",b"e"),(bytes([32]),b"o"),(bytes([0]),b"5"),(bytes([32,0]), bytes([1,32]))]
    tmp_dir = "./tmp_dir/"
    write_merges_to_file(test_merges, os.path.join(tmp_dir, "merges.txt"))
    retrieved_merges = read_merges_from_file(os.path.join(tmp_dir, "merges.txt"))
    assert test_merges==retrieved_merges
    print("Test passed! ")

    print("Testing read/write vocab")
    test_vocab = {
        0:b"<|endoftext|>",
        1:b"!",
        2:b"\"",
        3:b"#",
        4:bytes([0]),
        5:bytes([32,10])
    }
    write_vocab_to_file(test_vocab, os.path.join(tmp_dir, "vocab.json"))
    retrieved_vocab = read_vocab_from_file(os.path.join(tmp_dir, "vocab.json"))
    assert test_vocab==retrieved_vocab
    print("Test passed!")

    import shutil
    shutil.rmtree(tmp_dir)