"""Frozen public input resolution and generation dispatch, without real model inference."""
import json
from pathlib import Path
import shutil

import pytest

from src.paths import EVALUATION_BASE
from src.visual.audit import read_json
from src.visual.catalog_training_guard import catalog_training_guard
from src.experiments import base_workflow

GROUPS = tuple(read_json(EVALUATION_BASE / "manifest.json")["groups"])


@pytest.mark.parametrize("group", GROUPS)
def test_nine_groups_resolve_public_inputs_under_training_guard(group, tmp_path):
    with catalog_training_guard(allowed_output_roots=[tmp_path, EVALUATION_BASE / "groups"],
            private_paths=[EVALUATION_BASE / "shared_memory", EVALUATION_BASE / "evaluation_inputs"]) as audit:
        value = base_workflow.public_group(EVALUATION_BASE, group)
        assert Path(value["source"]["image"]).is_file()
        assert Path(value["target"]["image"]).is_file()
        assert value["definition"]["source_category"]
        assert value["definition"]["source_public_key"].startswith("Title: ")
        assert "FeatureText: " in value["definition"]["source_public_key"]
        assert value["target_text"] == value["definition"]["target_feature_text"]
        assert not any(key.startswith("perlage") for key in value["definition"])
        assert audit["blocked_accesses"] == []
        with pytest.raises(PermissionError):
            (EVALUATION_BASE / "shared_memory/shared_memory.json").read_text(encoding="utf-8")
    assert len(audit["blocked_accesses"]) == 1


def _copy_public_base(destination):
    destination.mkdir()
    shutil.copy2(EVALUATION_BASE / "manifest.json", destination / "manifest.json")
    shutil.copytree(EVALUATION_BASE / "public_metadata", destination / "public_metadata")
    shutil.copytree(EVALUATION_BASE / "groups/07_shampoo", destination / "groups/07_shampoo")
    return destination


@pytest.mark.parametrize("relative", ["groups/07_shampoo/definition.json", "public_metadata/product_metadata.json", "public_metadata/product_feature_review.json"])
def test_public_group_rejects_changed_frozen_metadata(tmp_path, relative):
    base = _copy_public_base(tmp_path / "base")
    path = base / relative
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="摘要"):
        base_workflow.public_group(base, "07_shampoo")


def test_base_copy_does_not_need_global_public_metadata(tmp_path, monkeypatch):
    from src.visual import target_features
    base = _copy_public_base(tmp_path / "base")
    monkeypatch.setattr(target_features, "CATALOG_DB_PATH", tmp_path / "absent-global-metadata.json")
    monkeypatch.setattr(target_features, "REVIEW_PATH", tmp_path / "absent-global-review.json")
    value = base_workflow.public_group(base, "07_shampoo")
    assert value["source"]["id"] == "B09B63YYCW"
    assert value["definition"]["source_category"] == "Shampoo & Conditioner"
    assert "BELLISSO" in value["definition"]["source_public_key"]
    assert "Aveda" not in value["definition"]["source_public_key"]


@pytest.mark.parametrize("method,steps,config", [
    ("ambilage", None, {"steps": 7}),
    ("ambilage", 3, {"steps": 7}),
])
def test_steps_and_method_config_reach_generation_without_loading_models(tmp_path, monkeypatch, method, steps, config):
    from src.visual import optimization
    from src.experiments import pipeline
    output = tmp_path / "output"
    monkeypatch.setattr(base_workflow, "OUTPUT", output)
    monkeypatch.setattr(pipeline, "source_snapshot", lambda: {"implementation": "offline-test"})
    seen = {}
    class StopBeforeModelLoad(RuntimeError):
        pass
    def capture(source, target, destination, **kwargs):
        seen.update(kwargs)
        raise StopBeforeModelLoad()
    monkeypatch.setattr(optimization, "generate_attack", capture)
    with pytest.raises(StopBeforeModelLoad):
        base_workflow.generate_group(EVALUATION_BASE, "07_shampoo", method, output / "run", steps=steps,
                                     ambilage_config=config, device="cpu", epsilon_levels=8)
    assert seen.get("steps") == steps
    assert ("steps" in seen) == (steps is not None)
    if method == "ambilage":
        assert seen["ambilage_config"].steps == config['steps']
        assert seen["ambilage_config"].surrogate_set == 'lucid'
        assert seen["definition"]["source_category"] == "Shampoo & Conditioner"
        assert "source_public_key" in seen["definition"]
    else:
        assert seen["epsilon_levels"] == 8
        assert "ambilage_config" not in seen
    assert not (output / "run/FROZEN_IMAGE.json").exists()


def test_g14_evaluator_uses_prepared_model_key_without_loading_weights(tmp_path, monkeypatch):
    from src.experiments.shared_evaluation import make_eval_encoder
    from src import model_registry
    from src.visual.encoders import clip, joint
    model = tmp_path / "eval_g14"
    model.mkdir()
    (model / "open_clip_pytorch_model.safetensors").write_text("test-placeholder", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    keys = []
    def resolve(key):
        keys.append(key)
        assert key == "eval_g14"
        return str(model)
    monkeypatch.setattr(model_registry, "resolve_model", resolve)
    seen = {}
    class FakeEncoder:
        snapshot = {}
    def encoder(config, project):
        seen.update(config)
        return FakeEncoder()
    monkeypatch.setattr(clip, "MatchedClipEncoder", encoder)
    monkeypatch.setattr(joint, "ChunkedClipEncoder", lambda value: value)
    result = make_eval_encoder("cpu", EVALUATION_BASE / "evaluation_encoder.json")
    assert keys == ["eval_g14"]
    assert seen["surrogate"]["weight_path"] == str((model / "open_clip_pytorch_model.safetensors").resolve())
    assert seen["surrogate"]["tokenizer_path"] == str(model)
    assert result.snapshot["evaluation_protocol"]["matches_training_surrogate"] is False


def test_gme_evaluator_uses_prepared_model_key_without_starting_worker(tmp_path, monkeypatch):
    from src.experiments.shared_evaluation import make_eval_encoder
    from src import model_registry
    from src.visual.encoders import joint
    model = tmp_path / "eval_gme"
    model.mkdir()
    keys = []
    def resolve(key):
        keys.append(key)
        assert key == "eval_gme"
        return str(model)
    monkeypatch.setattr(model_registry, "resolve_model", resolve)
    seen = {}
    class FakeEncoder:
        snapshot = {"provider": "gme"}
    def encoder(**options):
        seen.update(options)
        return FakeEncoder()
    monkeypatch.setattr(joint, "GMEProcessEncoder", encoder)
    result = make_eval_encoder("cpu", EVALUATION_BASE.parents[1] / "src/visual/config/evaluator.gme.example.json")
    assert keys == ["eval_gme"]
    assert seen["model"] == str(model)
    assert seen["python"] is None
    assert result.snapshot["evaluation_protocol"]["matches_training_surrogate"] is False
