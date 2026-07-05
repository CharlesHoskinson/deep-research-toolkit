import hashlib

from deep_research_toolkit.common.hashing import content_hash, file_hash


def test_content_hash_is_stable_prefixed_and_content_sensitive():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc").startswith("sha256:")
    assert content_hash("abc") != content_hash("abd")
    assert len(content_hash("abc", length=8).split(":")[1]) == 8


def test_file_hash_is_full_sha256_of_bytes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    assert file_hash(p) == hashlib.sha256(b"hello world").hexdigest()
