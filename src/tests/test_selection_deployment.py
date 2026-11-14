"""Model preparation and selected-image annotation interfaces, fully offline."""
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import pytest
import torch

from src.paths import PROJECT
from src.model_registry import model_specs


def test_selection_model_has_explicit_preparation_scope():
    assert model_specs()["selection"] == {"repo_id": "facebook/dinov2-small", "role": "frozen_public_grayscale_shape_gate", "scope": "selection"}
    output = subprocess.check_output([sys.executable, "-X", "utf8", "tools/prepare_models.py", "--scope", "selection"], cwd=PROJECT, text=True, encoding="utf-8")
    plan = json.loads(output)
    assert plan["download"] is False
    assert [item["key"] for item in plan["models"]] == ["selection"]


def test_shape_backend_uses_prepared_selection_directory(tmp_path, monkeypatch):
    from src import model_registry
    from src.visual.shape_features import LocalShapeBackend
    directory = tmp_path / "selection"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"offline-placeholder")
    (directory / "config.json").write_text("{}", encoding="utf-8")
    keys, calls = [], []
    def resolve(key):
        keys.append(key)
        return str(directory)
    class Model:
        def float(self): return self
        def eval(self): return self
        def requires_grad_(self, enabled): return self
        def to(self, device):
            calls.append(("device", str(device)))
            return self
    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor", Path(path), kwargs))
            return object()
    class Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", Path(path), kwargs))
            return Model()
    monkeypatch.setattr(model_registry, "resolve_model", resolve)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoImageProcessor=Processor, AutoModel=Loader, __version__="offline"))
    backend = LocalShapeBackend({}, tmp_path / "cache", device="cpu")
    assert keys == ["selection"]
    assert calls[0][1] == directory
    assert calls[0][2]["local_files_only"] is True
    assert ("device", "cpu") in calls
    assert backend.snapshot["preprocessing"]["foreground_segmentation"] is False
    backend.close()


def test_public_annotator_uses_sam2_and_moves_inputs_to_requested_device(tmp_path, monkeypatch):
    from src.visual import public_annotations
    directory = tmp_path / "sam2"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"offline-placeholder")
    (directory / "download_manifest.json").write_text("{}", encoding="utf-8")
    ocr_package = tmp_path / "ocr_package"
    ocr_package.mkdir()
    (ocr_package / "model.onnx").write_bytes(b"offline-placeholder")
    keys, calls = [], []
    def resolve(key):
        keys.append(key)
        return str(directory)
    class Batch(dict):
        def to(self, device):
            calls.append(("batch_device", str(device)))
            return self
    foreground = np.zeros((16, 16), dtype=bool)
    foreground[5:11, 5:11] = True
    mask = torch.tensor(foreground).reshape(1, 1, 1, 16, 16)
    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor", Path(path), kwargs))
            return cls()
        def __call__(self, **kwargs):
            return Batch(pixel_values=torch.zeros(1, 3, 16, 16), original_sizes=torch.tensor([[16, 16]]))
        def post_process_masks(self, masks, sizes):
            assert masks.device.type == sizes.device.type == "cpu"
            return [mask[0]]
    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", Path(path), kwargs))
            return cls(), {}
        def eval(self): return self
        def requires_grad_(self, enabled): return self
        def to(self, device):
            calls.append(("model_device", str(device)))
            return self
        def __call__(self, **kwargs):
            return types.SimpleNamespace(pred_masks=mask, iou_scores=torch.ones(1, 1, 1))
    class OCR:
        def __init__(self, **kwargs): pass
        def __call__(self, *args, **kwargs): return [], 0
    monkeypatch.setattr(public_annotations, "resolve_model", resolve)
    monkeypatch.setattr(public_annotations.importlib.metadata, "version", lambda name: "offline")
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(Sam2Processor=Processor, Sam2Model=Model))
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=OCR, __file__=str(ocr_package / "__init__.py")))
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(setNumThreads=lambda number: None))
    annotator = public_annotations.PublicImageAnnotator(device="cpu")
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    image[foreground] = [120, 30, 10]
    path = tmp_path / "source.png"
    Image.fromarray(image).save(path)
    from src.visual.image_utils import sha256_file
    result = annotator.propose({"id": "PUBLIC0001", "image": str(path), "image_sha256": sha256_file(path), "title": "Synthetic public input"}, tmp_path / "annotation")
    assert keys == ["sam2"]
    assert ("model_device", "cpu") in calls
    assert ("batch_device", "cpu") in calls
    assert annotator.snapshot["device"] == "cpu"
    assert result["text_regions_reviewed"] is False
    assert Path(result["foreground_mask"]).is_file()
    annotator.close()


def test_old_webshop_routes_are_not_registered():
    source = (PROJECT / "src/webshop/full_webshop.py").read_text(encoding="utf-8")
    assert "/perlage-disk/health" not in source
    assert "/perlage-assets/" not in source
    assert "/ambilage-disk/health" in source
    assert "/ambilage-assets/" in source


def test_public_annotator_rejects_unavailable_cuda_instead_of_falling_back(monkeypatch):
    from src.visual.public_annotations import PublicImageAnnotator
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(setNumThreads=lambda number: None))
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=object))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(Sam2Processor=object, Sam2Model=object))
    with pytest.raises(RuntimeError, match="禁止静默回退"):
        PublicImageAnnotator(device="cuda")
