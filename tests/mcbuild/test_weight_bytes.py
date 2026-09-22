"""item K: ``run_reader.safetensors_total_bytes`` reads weight sizes from
safetensors METADATA only (index json or file header); never a weight download."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark.bineval import run_reader
from benchmark.bineval.run_reader import safetensors_header_bytes, safetensors_total_bytes


def test_local_index_total_size(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 55_600_000_000}, "weight_map": {}}), encoding="utf-8"
    )
    assert safetensors_total_bytes(str(tmp_path)) == 55_600_000_000


def test_local_single_file_header(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    save_file({"a": torch.zeros(3, 4, dtype=torch.float32), "b": torch.ones(5, dtype=torch.bfloat16)},
              str(tmp_path / "model.safetensors"))
    expected = 3 * 4 * 4 + 5 * 2
    assert safetensors_header_bytes(tmp_path / "model.safetensors") == expected
    assert safetensors_total_bytes(str(tmp_path)) == expected
    # the header is read, not the whole file: the first 8 bytes state its length
    with (tmp_path / "model.safetensors").open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    assert set(header) >= {"a", "b"}


def test_local_dir_without_safetensors_is_none(tmp_path: Path) -> None:
    assert safetensors_total_bytes(str(tmp_path)) is None


def test_hub_index_then_single_file_header(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    idx = tmp_path / "model.safetensors.index.json"
    idx.write_text(json.dumps({"metadata": {"total_size": 123}}), encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, fn, **kw: str(idx))
    assert safetensors_total_bytes("org/sharded") == 123

    def no_index(repo, fn, **kw):
        raise EntryNotFoundError("no index")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_index)
    meta = SimpleNamespace(tensors={
        "a": SimpleNamespace(data_offsets=(0, 40)),
        "b": SimpleNamespace(data_offsets=(40, 100)),
    })
    monkeypatch.setattr(huggingface_hub, "parse_safetensors_file_metadata", lambda repo, fn, **kw: meta)
    assert safetensors_total_bytes("org/single") == 100

    def boom(repo, fn, **kw):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "parse_safetensors_file_metadata", boom)
    assert safetensors_total_bytes("org/unreachable") is None


def test_helper_is_importable_without_torch_side_effects() -> None:
    assert callable(run_reader.safetensors_total_bytes)
    with pytest.raises(FileNotFoundError):
        safetensors_header_bytes(Path("does/not/exist.safetensors"))
