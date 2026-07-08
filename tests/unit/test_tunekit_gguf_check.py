"""tunekit.gguf_check: a minimal GGUF v3 metadata-KV reader (no tensor
parsing) plus the control-token validator guarding against the
unsloth#5070/#5386 corruption class (CONTROL tokens silently demoted to
NORMAL). Every GGUF byte blob here is constructed by hand in this file --
no fixture binary, no real model."""
from __future__ import annotations

import struct

import pytest

from deep_research_toolkit.tunekit.gguf_check import (
    GGUF_MAGIC,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    GGUFParseError,
    read_gguf_metadata,
    validate_control_tokens,
    validate_gguf_file,
)

_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_STRING = 8
_TYPE_ARRAY = 9


def _string_kv(key: str, value: str) -> bytes:
    key_b = key.encode("utf-8")
    val_b = value.encode("utf-8")
    return (struct.pack("<Q", len(key_b)) + key_b
            + struct.pack("<I", _TYPE_STRING)
            + struct.pack("<Q", len(val_b)) + val_b)


def _u32_kv(key: str, value: int) -> bytes:
    key_b = key.encode("utf-8")
    return (struct.pack("<Q", len(key_b)) + key_b
            + struct.pack("<I", _TYPE_UINT32)
            + struct.pack("<I", value))


def _string_array_kv(key: str, values: list[str]) -> bytes:
    key_b = key.encode("utf-8")
    body = struct.pack("<I", _TYPE_STRING) + struct.pack("<Q", len(values))
    for v in values:
        v_b = v.encode("utf-8")
        body += struct.pack("<Q", len(v_b)) + v_b
    return (struct.pack("<Q", len(key_b)) + key_b
            + struct.pack("<I", _TYPE_ARRAY) + body)


def _int32_array_kv(key: str, values: list[int]) -> bytes:
    key_b = key.encode("utf-8")
    body = struct.pack("<I", _TYPE_INT32) + struct.pack("<Q", len(values))
    for v in values:
        body += struct.pack("<i", v)
    return (struct.pack("<Q", len(key_b)) + key_b
            + struct.pack("<I", _TYPE_ARRAY) + body)


#: A tiny vocab covering: two ordinary tokens, the two turn-boundary control
#: tokens, and BOS/EOS -- enough to exercise every check
#: validate_control_tokens performs.
_TOKENS = ["<pad>", "hello", "<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "world"]
_BOS_ID = 4
_EOS_ID = 5


def _build_gguf(token_types: list[int], *, version: int = 3, magic: bytes = GGUF_MAGIC,
                extra_kvs: bytes = b"", bos_id: int = _BOS_ID, eos_id: int = _EOS_ID,
                include_tokenizer_kvs: bool = True) -> bytes:
    """Hand-builds a GGUF byte blob with just enough metadata KVs to drive
    read_gguf_metadata + validate_control_tokens: `general.architecture` (an
    ordinary string KV, to prove multi-KV parsing works), plus the
    tokenizer.ggml.* keys the validator reads. tensor_count is 0 and no
    tensor-info bytes follow -- read_gguf_metadata must never try to read
    them (design doc §6.2: "no tensor parsing")."""
    kvs = _string_kv("general.architecture", "gemma4") + extra_kvs
    if include_tokenizer_kvs:
        kvs += _string_array_kv("tokenizer.ggml.tokens", _TOKENS)
        kvs += _int32_array_kv("tokenizer.ggml.token_type", token_types)
        kvs += _u32_kv("tokenizer.ggml.bos_token_id", bos_id)
        kvs += _u32_kv("tokenizer.ggml.eos_token_id", eos_id)
        kv_count = 5
    else:
        kv_count = 1

    header = magic + struct.pack("<I", version) + struct.pack("<Q", 0) + struct.pack("<Q", kv_count)
    return header + kvs


def _all_control_token_types() -> list[int]:
    # index: 0 <pad>=NORMAL, 1 hello=NORMAL, 2 <start_of_turn>=CONTROL,
    # 3 <end_of_turn>=CONTROL, 4 <bos>=CONTROL, 5 <eos>=CONTROL, 6 world=NORMAL
    return [TOKEN_TYPE_NORMAL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL]


# ---------------------------------------------------------------------------
# read_gguf_metadata
# ---------------------------------------------------------------------------

def test_read_gguf_metadata_parses_scalar_and_array_kvs():
    data = _build_gguf(_all_control_token_types())
    metadata = read_gguf_metadata(data)
    assert metadata["general.architecture"] == "gemma4"
    assert metadata["tokenizer.ggml.tokens"] == _TOKENS
    assert metadata["tokenizer.ggml.token_type"] == _all_control_token_types()
    assert metadata["tokenizer.ggml.bos_token_id"] == _BOS_ID
    assert metadata["tokenizer.ggml.eos_token_id"] == _EOS_ID


def test_read_gguf_metadata_stops_before_tensor_section():
    # No tensor-info bytes are appended at all; a parser that (incorrectly)
    # tried to read tensor info here would raise GGUFParseError on the
    # truncated read. It must not.
    data = _build_gguf(_all_control_token_types())
    read_gguf_metadata(data)  # does not raise


def test_read_gguf_metadata_bad_magic_raises():
    data = _build_gguf(_all_control_token_types(), magic=b"NOPE")
    with pytest.raises(GGUFParseError, match="bad magic"):
        read_gguf_metadata(data)


def test_read_gguf_metadata_unsupported_version_raises():
    data = _build_gguf(_all_control_token_types(), version=2)
    with pytest.raises(GGUFParseError, match="unsupported GGUF version"):
        read_gguf_metadata(data)


def test_read_gguf_metadata_truncated_data_raises():
    data = _build_gguf(_all_control_token_types())
    with pytest.raises(GGUFParseError, match="unexpected end"):
        read_gguf_metadata(data[:-5])


# ---------------------------------------------------------------------------
# validate_control_tokens: the pass case
# ---------------------------------------------------------------------------

def test_validate_control_tokens_all_correct():
    metadata = read_gguf_metadata(_build_gguf(_all_control_token_types()))
    report = validate_control_tokens(metadata)
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["checked"]["'<start_of_turn>'"] == TOKEN_TYPE_CONTROL
    assert report["checked"]["'<end_of_turn>'"] == TOKEN_TYPE_CONTROL
    assert report["checked"]["BOS"] == TOKEN_TYPE_CONTROL
    assert report["checked"]["EOS"] == TOKEN_TYPE_CONTROL


# ---------------------------------------------------------------------------
# validate_control_tokens: the unsloth#5070/#5386 corruption class
# ---------------------------------------------------------------------------

def test_validate_control_tokens_flags_start_of_turn_demoted_to_normal():
    types = _all_control_token_types()
    types[2] = TOKEN_TYPE_NORMAL  # <start_of_turn> corrupted
    report = validate_control_tokens(read_gguf_metadata(_build_gguf(types)))
    assert report["ok"] is False
    assert any("<start_of_turn>" in e and "CONTROL" in e for e in report["errors"])


def test_validate_control_tokens_flags_end_of_turn_demoted_to_normal():
    types = _all_control_token_types()
    types[3] = TOKEN_TYPE_NORMAL  # <end_of_turn> corrupted
    report = validate_control_tokens(read_gguf_metadata(_build_gguf(types)))
    assert report["ok"] is False
    assert any("<end_of_turn>" in e for e in report["errors"])


def test_validate_control_tokens_flags_eos_demoted_to_normal():
    types = _all_control_token_types()
    types[_EOS_ID] = TOKEN_TYPE_NORMAL  # EOS corrupted -- the unsloth#5386 bug exactly
    report = validate_control_tokens(read_gguf_metadata(_build_gguf(types)))
    assert report["ok"] is False
    assert any("EOS" in e for e in report["errors"])


def test_validate_control_tokens_flags_bos_demoted_to_normal():
    types = _all_control_token_types()
    types[_BOS_ID] = TOKEN_TYPE_NORMAL
    report = validate_control_tokens(read_gguf_metadata(_build_gguf(types)))
    assert report["ok"] is False
    assert any("BOS" in e for e in report["errors"])


def test_validate_control_tokens_missing_tokenizer_metadata_fails_not_skips():
    data = _build_gguf(_all_control_token_types(), include_tokenizer_kvs=False)
    metadata = read_gguf_metadata(data)
    report = validate_control_tokens(metadata)
    assert report["ok"] is False
    assert report["errors"]


def test_validate_control_tokens_missing_start_of_turn_token_reported():
    tokens_without_marker = [t for t in _TOKENS if t != "<start_of_turn>"]
    types = [TOKEN_TYPE_NORMAL] * len(tokens_without_marker)
    key_b = b"tokenizer.ggml.tokens"
    body = struct.pack("<I", _TYPE_STRING) + struct.pack("<Q", len(tokens_without_marker))
    for v in tokens_without_marker:
        v_b = v.encode("utf-8")
        body += struct.pack("<Q", len(v_b)) + v_b
    tokens_kv = struct.pack("<Q", len(key_b)) + key_b + struct.pack("<I", _TYPE_ARRAY) + body
    types_kv = _int32_array_kv("tokenizer.ggml.token_type", types)
    header = GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 4)
    data = (header + _string_kv("general.architecture", "gemma4") + tokens_kv + types_kv
           + _u32_kv("tokenizer.ggml.bos_token_id", 0) + _u32_kv("tokenizer.ggml.eos_token_id", 1))
    report = validate_control_tokens(read_gguf_metadata(data))
    assert report["ok"] is False
    assert any("<start_of_turn>" in e and "not found" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# validate_gguf_file: file-level convenience wrapper
# ---------------------------------------------------------------------------

def test_validate_gguf_file_reads_from_disk(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(_build_gguf(_all_control_token_types()))
    report = validate_gguf_file(path)
    assert report["ok"] is True


def test_validate_gguf_file_detects_corruption_from_disk(tmp_path):
    types = _all_control_token_types()
    types[_EOS_ID] = TOKEN_TYPE_NORMAL
    path = tmp_path / "corrupted.gguf"
    path.write_bytes(_build_gguf(types))
    report = validate_gguf_file(path)
    assert report["ok"] is False
