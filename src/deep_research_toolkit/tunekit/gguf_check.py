"""GGUF control-token validator (design doc §6.2/§6.3/§9): a hard promotion
gate against the unsloth#5070/#5386 class of bug -- GGUF export silently
demoting `<start_of_turn>`, `<end_of_turn>`, BOS, or EOS from CONTROL to
NORMAL token_type, which makes a served model never stop (or start)
generating a turn correctly.

This module reads ONLY the GGUF header and the metadata key-value section --
no tensor-info or tensor-data parsing at all, per design doc §6.2 ("a minimal
GGUF metadata-KV reader... no tensor parsing"). The metadata section always
comes immediately after the header and before tensor info in the GGUF binary
layout, so the reader can (and does) stop before ever touching a tensor byte.

GGUF VERSION TARGETED: spec version 3 (the version every current llama.cpp/
Unsloth GGUF export uses; ggml-org/ggml `docs/gguf.md`). Version 2 files
(pre-2024, u32 KV/tensor counts instead of u64) are NOT supported by this
minimal reader -- see `read_gguf_metadata`'s version check.
"""
from __future__ import annotations

import struct

GGUF_MAGIC = b"GGUF"

#: The GGUF spec version this reader targets (see module docstring).
SUPPORTED_GGUF_VERSION = 3

# GGUF metadata value types (ggml-org/ggml docs/gguf.md `gguf_metadata_value_type`).
_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

#: struct.unpack format for every SCALAR value type (STRING and ARRAY are
#: handled separately -- they are variable-length, not fixed-size scalars).
_SCALAR_STRUCT_FMT = {
    _TYPE_UINT8: "<B", _TYPE_INT8: "<b",
    _TYPE_UINT16: "<H", _TYPE_INT16: "<h",
    _TYPE_UINT32: "<I", _TYPE_INT32: "<i",
    _TYPE_FLOAT32: "<f", _TYPE_BOOL: "<?",
    _TYPE_UINT64: "<Q", _TYPE_INT64: "<q",
    _TYPE_FLOAT64: "<d",
}

# llama.cpp `llama_token_type` / GGUF `tokenizer.ggml.token_type` values.
TOKEN_TYPE_UNDEFINED = 0
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4
TOKEN_TYPE_UNUSED = 5
TOKEN_TYPE_BYTE = 6

#: The exact tokens unsloth#5070/#5386 corrupted -- Gemma's turn-boundary
#: control tokens. Checked by literal string against `tokenizer.ggml.tokens`.
REQUIRED_CONTROL_TOKENS = ("<start_of_turn>", "<end_of_turn>")


class GGUFParseError(ValueError):
    """Raised on a malformed/truncated GGUF byte stream, or an unsupported
    GGUF version -- never a raw struct.error or IndexError."""


class _ByteReader:
    """Tiny cursor over an in-memory byte string. Every read is bounds
    checked so a truncated/corrupt file raises GGUFParseError instead of
    silently reading garbage or throwing an opaque struct.error."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise GGUFParseError(
                f"unexpected end of GGUF data at offset {self._pos} (wanted {n} more bytes)")
        chunk = self._data[self._pos:self._pos + n]
        self._pos += n
        return chunk

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def read_gguf_string(self) -> str:
        """A GGUF string is a u64 length prefix followed by that many UTF-8
        bytes (NOT null-terminated)."""
        length = self.read_u64()
        return self.read(length).decode("utf-8", errors="replace")

    def read_value(self, value_type: int):
        if value_type == _TYPE_STRING:
            return self.read_gguf_string()
        if value_type == _TYPE_ARRAY:
            elem_type = self.read_u32()
            count = self.read_u64()
            return [self.read_value(elem_type) for _ in range(count)]
        fmt = _SCALAR_STRUCT_FMT.get(value_type)
        if fmt is None:
            raise GGUFParseError(f"unsupported GGUF metadata value type {value_type}")
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]


def read_gguf_metadata(data: bytes) -> dict:
    """Parses the GGUF header + metadata key-value section ONLY -- stops
    before the tensor-info section, so no tensor bytes are ever read (design
    doc §6.2: "no tensor parsing"). Returns `{metadata_key: value}`.

    Raises GGUFParseError on a bad magic, an unsupported version (only
    `SUPPORTED_GGUF_VERSION` is handled), or truncated/malformed data."""
    r = _ByteReader(data)
    magic = r.read(4)
    if magic != GGUF_MAGIC:
        raise GGUFParseError(f"not a GGUF file: bad magic {magic!r} (expected {GGUF_MAGIC!r})")

    version = r.read_u32()
    if version != SUPPORTED_GGUF_VERSION:
        raise GGUFParseError(
            f"unsupported GGUF version {version} (this reader targets version "
            f"{SUPPORTED_GGUF_VERSION} only)")

    r.read_u64()  # tensor_count -- unused; this reader never parses tensor info
    kv_count = r.read_u64()

    metadata: dict = {}
    for _ in range(kv_count):
        key = r.read_gguf_string()
        value_type = r.read_u32()
        metadata[key] = r.read_value(value_type)
    return metadata


def validate_control_tokens(metadata: dict) -> dict:
    """Asserts `<start_of_turn>`/`<end_of_turn>`/BOS/EOS are token_type
    CONTROL, not NORMAL (the unsloth#5070/#5386 corruption class). Report-
    only: never raises -- returns `{"ok": bool, "errors": [...], "checked":
    {name: token_type}}` so a caller (CLI or promote.py) decides how to act
    on a failure.

    Looks up `<start_of_turn>`/`<end_of_turn>` by exact string match against
    `tokenizer.ggml.tokens`; BOS/EOS are looked up by id via
    `tokenizer.ggml.bos_token_id`/`tokenizer.ggml.eos_token_id`. Missing
    metadata (not a GGUF built with a tokenizer, or a key renamed upstream)
    is itself an error, not a silent skip -- a promotion gate must not treat
    "couldn't check" the same as "checked and fine"."""
    errors: list[str] = []
    checked: dict[str, int | None] = {}

    tokens = metadata.get("tokenizer.ggml.tokens")
    token_types = metadata.get("tokenizer.ggml.token_type")
    if tokens is None or token_types is None:
        return {
            "ok": False,
            "errors": ["missing tokenizer.ggml.tokens or tokenizer.ggml.token_type metadata "
                      "-- cannot validate control tokens"],
            "checked": checked,
        }

    index_by_token = {tok: i for i, tok in enumerate(tokens)}

    def _check(name: str, index: int | None) -> None:
        if index is None:
            errors.append(f"{name} not found")
            checked[name] = None
            return
        if not (0 <= index < len(token_types)):
            errors.append(f"{name} (id {index}) is out of range of tokenizer.ggml.token_type")
            checked[name] = None
            return
        ttype = token_types[index]
        checked[name] = ttype
        if ttype != TOKEN_TYPE_CONTROL:
            errors.append(
                f"{name} (id {index}) has token_type {ttype}, expected CONTROL "
                f"({TOKEN_TYPE_CONTROL}) -- unsloth#5070/#5386-class corruption")

    for tok in REQUIRED_CONTROL_TOKENS:
        _check(repr(tok), index_by_token.get(tok))

    for name, id_key in (("BOS", "tokenizer.ggml.bos_token_id"),
                         ("EOS", "tokenizer.ggml.eos_token_id")):
        token_id = metadata.get(id_key)
        if token_id is None:
            errors.append(f"missing {id_key}")
            checked[name] = None
            continue
        _check(name, token_id)

    return {"ok": not errors, "errors": errors, "checked": checked}


def validate_gguf_file(path) -> dict:
    """Convenience wrapper: reads `path` off disk, parses its metadata, and
    validates control tokens in one call. Raises GGUFParseError if the file
    isn't a well-formed GGUF v3 header; returns the report dict from
    `validate_control_tokens` otherwise (report-only from there on)."""
    with open(path, "rb") as f:
        data = f.read()
    metadata = read_gguf_metadata(data)
    return validate_control_tokens(metadata)
