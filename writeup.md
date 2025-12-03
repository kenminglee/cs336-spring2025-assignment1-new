# Problem (`unicode1`): Understanding Unicode

(a) `chr(0)` returns `'\x00'`.

(b) The `repr` representation of `chr(0)` (i.e., `print(chr(0))`), is `"'\\x00'"`.

(c) 
```
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

# Problem (`unicode2`): Unicode Encodings

Note on how UTF-8 encoding works:
- Basically all characters (aka codepoints) are converted to a sequence of byte values (i.e., 0-255). 
- Ascii characters like A-Z or a-z fit into one byte (so one 0-255 number suffice to represent an ascii character).
- Other characters will need to be encoded in 2, 3 or even 4 bytes. They are encoded as such:
    - 1 byte: `0xxxxxxx`
    - 2 bytes: `110xxxxx 10xxxxxx`
    - 3 bytes: `1110xxxx 10xxxxxx 10xxxxxx`
    - 4 bytes: `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`
    - The amount of ones in the first byte (from the left) tells you how many of the following bytes still belong to the same character. All bytes that belong to the sequence start with 10 in binary. To encode the character you convert its codepoint to binary and fill in the x’s.

1. Generally speaking, the advantage of training on UTF-8 encoded bytes (2-bytes, or 8 bits per token), is that our values are much denser (less wastage, since all english characters falls into the range of 0-255). The disadvantage is that for characters outside the range of 0-255, it takes multiple 2-byte tokens to represent a single character, so then a sentence may result in a long sequence of tokens.
Additionally, the probability of outputting invalid characters may be larger?

2. This function does not work because some characters take multiple bytes to be represented!
```
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'

>>> decode_utf8_bytes_to_str_wrong("hello你好".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
  File "<stdin>", line 2, in <listcomp>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```

3. `01111111 10111111` is an invalid two-byte sequence that does not decode to any Unicode character. That's because the second byte starts with a `10`, which is a pattern to represent continuation byte (see above for explaination). However, the 1st byte starts with `0`, which means the unicode character only contains 1 byte. Hence the 2nd byte is a stray continuation byte (that is invalid).  (a) 

# Note on encoding-to/decoding-from bytes:

UTF-8/UTF-16/UTF-32 are encoding specs, they determine how strings are encoded in bytes, and vice versa.

1. To encode a `str` into `bytes`, do `<str>.encode('utf-8')`:
```
>>> "hi".encode("utf-8")
b'hi'
>>> "你好".encode("utf-8")
b'\xe4\xbd\xa0\xe5\xa5\xbd'
```

2. To convert `bytes` into `str`, do `<bytes>.decode('utf-8')`:
```
>>> b'hi'.decode('utf-8')
'hi'
```

3. Note that `utf-8` is used by default:

Encoding:
```
>>> 'hi'.encode()
b'hi'
>>> 'hi'.encode('utf-16')
b'\xff\xfeh\x00i\x00'
>>> 'hi'.encode('utf-8')
b'hi'
```

Decoding:
```
>>> b'hi'.decode('utf-8')
'hi'
>>> b'hi'.decode('utf-16')
'楨'
>>> b'hi'.decode()
'hi'
```

4. Iterating a sequence of `bytes` gives us a list of integers (representing the value of each byte in base 10):
```
>>> [i for i in "hi".encode("utf-8")]
[104, 105]
```

5. Hence, to convert a string into a list of bytes, do:
```
>>> [i for i in "hi".encode("utf-8")]
[104, 105]
>>> [bytes([i]) for i in "hi".encode("utf-8")]
[b'h', b'i']
```

Note: `bytes([i])` creates a `bytes` with value `i`, while `bytes(i)` creates `i` bytes with value 0:
```
>>> bytes([2])
b'\x02'
>>> bytes(2)
b'\x00\x00'
```


