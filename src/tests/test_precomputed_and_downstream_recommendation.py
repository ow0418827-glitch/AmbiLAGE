"""Unit tests for PrecomputedEncoderStore, OfflinePrecomputedEvalEncoder, and MiniMax M3 downstream recommendation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.experiments.downstream_recommendation import (
    SBERTGHRCalculator,
    build_minimax_m3_messages,
    call_minimax_m3,
    compute_recommendation_metrics,
    evaluate_downstream_job,
    norm_text,
    parse_recommendation_response,
    sanitize_memory_entry,
)
from src.visual.image_utils import sha256_file
from src.visual.precomputed_eval_store import (
    OfflinePrecomputedEvalEncoder,
    PrecomputedEncoderStore,
    _atomic_save_npy,
    canonical_text_key,
    format_ghr_target_sentence,
    text_key_sha256,
)
from src.visual.visual_retrieval import pillow_decoder_signature


def _make_test_image(path: Path, color: tuple[int, int, int]) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
    return path, sha256_file(path)


def _deterministic_vec(seed_str: str, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(seed_str)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _build_fixture_precomputed_store(
    store_root: Path,
    slug: str,
    image_shas: list[str],
    texts: list[str],
    *,
    target_img_sha: str | None = None,
    query_vec_override: np.ndarray | None = None,
) -> PrecomputedEncoderStore:
    dim = 8
    enc_dir = store_root / slug
    enc_dir.mkdir(parents=True, exist_ok=True)

    sorted_imgs = sorted(set(image_shas))
    img_vecs = []
    for sha in sorted_imgs:
        if sha == target_img_sha and query_vec_override is not None:
            img_vecs.append(query_vec_override.copy())
        else:
            img_vecs.append(_deterministic_vec(f"img:{sha}", dim))
    img_matrix = np.vstack(img_vecs).astype(np.float32)

    txt_map = {}
    audits = {}
    for t in texts:
        nt = canonical_text_key(t)
        t_sha = text_key_sha256(nt)
        txt_map[t_sha] = (
            query_vec_override.copy()
            if query_vec_override is not None
            else _deterministic_vec(f"txt:{t_sha}", dim)
        )
        audits[t_sha] = {
            "text_sha256": t_sha,
            "input_tokens": len(nt.split()),
            "encoded_tokens": len(nt.split()),
            "chunks": 1,
            "payload_tokens_per_chunk": 75,
            "chunk_token_counts": [len(nt.split())],
            "truncated": False,
        }
    sorted_txts = sorted(txt_map)
    txt_matrix = np.vstack([txt_map[s] for s in sorted_txts]).astype(np.float32)

    img_npy_sha = _atomic_save_npy(enc_dir / "image_vectors.npy", img_matrix)
    txt_npy_sha = _atomic_save_npy(enc_dir / "text_vectors.npy", txt_matrix)

    manifest = {
        "schema_version": 1,
        "encoder_slug": slug,
        "encoder_label": "Fixture-Encoder",
        "surrogate_pair": ["ViT-L-14", "openai"],
        "weight_sha256": "a" * 64,
        "vector_dim": dim,
        "image_count": len(sorted_imgs),
        "text_count": len(sorted_txts),
        "image_vectors_sha256": img_npy_sha,
        "text_vectors_sha256": txt_npy_sha,
        "image_decoder": pillow_decoder_signature(),
        "core_encoder_snapshot": {
            "provider": "clip",
            "loader": "standalone_eval_clip",
            "name": "ViT-L-14",
            "pretrained": "openai",
            "precision": "float32",
        },
        "image_shas": sorted_imgs,
        "text_shas": sorted_txts,
        "text_audits": audits,
    }
    (enc_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PrecomputedEncoderStore(slug, store_root)


def test_precomputed_store_and_offline_encoder_zero_gpu(tmp_path):
    img1, sha1 = _make_test_image(tmp_path / "img1.png", (200, 10, 10))
    img2, sha2 = _make_test_image(tmp_path / "img2.png", (10, 200, 10))
    q_text = "Best electric toothbrush for sensitive gums"

    store = _build_fixture_precomputed_store(
        tmp_path / "store",
        "01_clip_l14",
        [sha1, sha2],
        [q_text, "Product feature text"],
    )
    assert store.covers_all([sha1, sha2], [q_text])
    assert not store.covers_all([sha1, "missing_sha"], [q_text])

    offline_enc = OfflinePrecomputedEvalEncoder(
        store,
        batch_size=16,
        device_label="cuda",
        evaluation_protocol={"setting": "held_out_public_encoder"},
    )
    v1 = offline_enc.encode(img1)
    assert v1.shape == (8,) and np.isclose(np.linalg.norm(v1), 1.0)
    vt = offline_enc.encode_text(q_text)
    assert vt.shape == (8,) and np.isclose(np.linalg.norm(vt), 1.0)
    assert offline_enc.last_encoding_audit["text_sha256"] == text_key_sha256(q_text)

    vj = offline_enc.encode_joint(q_text, img1, image_weight=0.5)
    assert vj.shape == (8,) and np.isclose(np.linalg.norm(vj), 1.0)
    offline_enc.close()


def test_sanitize_memory_entry_and_interleaved_messages_no_leakage(tmp_path):
    entries = []
    for i in range(5):
        img_p, img_sha = _make_test_image(tmp_path / f"item_{i}.png", (20 * i, 100, 150))
        entries.append({
            "Id": f"groups9_reference_09_toothbrush_B00{i}",
            "EntryType": "error" if i == 0 else "normal",
            "RecordKind": "purchase_memory",
            "SourceGroup": "09_toothbrush",
            "FinalASIN": f"B00000000{i}",
            "Brand": f"Brand{i}" if i > 0 else "",
            "Attributes": [f"Attr_{i}_{j}" for j in range(8)],
            "FeatureText": f"Summary feature note for item {i}.",
            "ObservedPurchase": {"secret": "internal_audit"},
            "ReferenceNote": "poisoned reference note",
            "ImageSubstitution": {"protocol": "controlled_image_substitution"},
            "PrimaryImage": {
                "ASIN": f"B00000000{i}",
                "Title": f"Clean Product Title Number {i}",
                "LocalPath": img_p.name,
                "SHA256": img_sha,
            },
        })

    bank_path = tmp_path / "modified.json"
    bank_path.write_text(json.dumps(entries), encoding="utf-8")

    s0 = sanitize_memory_entry(entries[0], bank_path, rank=1)
    assert s0["brand"] == "Unknown"
    assert len(s0["attributes"]) == 6  # truncated to max 6 attributes

    messages, sanitized_audit, content_audit = build_minimax_m3_messages(
        entries,
        "I need a soft toothbrush for daily oral care.",
        memory_path=bank_path,
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    user_blocks = messages[1]["content"]
    # 1 header + 5 * (1 text + 1 image) + 1 final query = 12 blocks
    assert len(user_blocks) == 12
    for rank_idx in range(5):
        t_blk = user_blocks[1 + 2 * rank_idx]
        i_blk = user_blocks[2 + 2 * rank_idx]
        assert t_blk["type"] == "text"
        assert f"Memory #{rank_idx + 1} (Rank {rank_idx + 1}):" in t_blk["text"]
        assert i_blk["type"] == "image_url"
        assert i_blk["image_url"]["url"].startswith("data:image/png;base64,")

    serialized_payload = json.dumps(content_audit)
    for forbidden in ("groups9_reference_", "B00000000", "ImageSubstitution", "ObservedPurchase", "poisoned reference"):
        assert forbidden not in serialized_payload


def test_parse_recommendation_response_three_tier_tolerance():
    # Tier 1: <think> tag + markdown ```json fence
    raw_1 = """<think>Analyzing top 5 memories...</think>
```json
{
  "reasoning": "Memory #1 matches the user's request for a U-shaped silicone toothbrush.",
  "recommend": "I highly recommend the Tthxqing U shaped Toothbrush, Food Grade Soft Silicone Whole Mouth Toothbrush.",
  "recommended_product_name": "Tthxqing U shaped Toothbrush, Food Grade Soft Silicone Whole Mouth Toothbrush"
}
```"""
    p1 = parse_recommendation_response(raw_1)
    assert p1["parse_status"] == "json_ok"
    assert "Tthxqing U shaped Toothbrush" in p1["recommend"]
    assert p1["recommended_product_name"].startswith("Tthxqing")

    # Tier 2: Malformed/truncated JSON falling back to regex
    raw_2 = '{"reasoning": "Fits well.", "recommend": "You should choose the SUNBA 3MP Starlight Camera.", "recommended_product_name": null'
    p2 = parse_recommendation_response(raw_2)
    assert p2["parse_status"] == "fallback_regex"
    assert p2["recommend"] == "You should choose the SUNBA 3MP Starlight Camera."
    assert p2["recommended_product_name"] is None

    # Tier 3: Plain unstructured text falling back to raw
    raw_3 = "Based on your memories, I suggest getting the Cinch Men's Bronze Label Jeans."
    p3 = parse_recommendation_response(raw_3)
    assert p3["parse_status"] == "fallback_raw"
    assert "Cinch Men's Bronze Label Jeans" in p3["recommend"]
    assert p3["recommended_product_name"] is None


def test_ghr_sentence_and_dual_recommendation_metrics(tmp_path):
    # Brand deduplication when title already starts with brand
    assert (
        format_ghr_target_sentence("SUNBA", "SUNBA 3MP Outdoor PTZ Camera")
        == "I recommend the SUNBA 3MP Outdoor PTZ Camera."
    )
    assert (
        format_ghr_target_sentence("Buy-Rite", "Chelsea Styling Chair for Salons")
        == "I recommend the Buy-Rite Chelsea Styling Chair for Salons."
    )
    assert (
        format_ghr_target_sentence("", "Window Treatment Semi-Sheer Curtain")
        == "I recommend the Window Treatment Semi-Sheer Curtain."
    )

    # Stub SBERT calculator where identical/similar texts have high cosine
    def _stub_embed(text: str) -> np.ndarray:
        tokens = set(norm_text(text).split())
        vocab = ["recommend", "tthxqing", "shaped", "toothbrush", "food", "grade", "soft", "silicone", "oral", "other"]
        vec = np.asarray([1.0 if w in tokens else 0.05 for w in vocab], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    sbert = SBERTGHRCalculator(store_root=tmp_path, encoder_fn=_stub_embed)
    target_title = "Tthxqing U shaped Toothbrush, Food Grade Soft Silicone Whole Mouth Toothbrush"
    target_brand = "Tthxqing"
    top5_titles = [
        target_title,
        "Oral-B Pro 1000 CrossAction Electric Toothbrush",
        "Philips Sonicare ProtectiveClean 5100",
        "Colgate Extra Clean Full Head Manual Toothbrush",
        "Waterpik Cordless Advanced Water Flosser",
    ]

    # Case A: Exact match in recommend text
    m_exact = compute_recommendation_metrics(
        f"For your needs, I recommend the {target_title} because of its gentle silicone bristles.",
        target_title,
        target_title=target_title,
        target_brand=target_brand,
        top5_titles=top5_titles,
        sbert_calculator=sbert,
    )
    assert m_exact["rec_exact_hit"] is True
    assert m_exact["rec_abbrev_hit"] is False
    assert m_exact["rec_any_hit"] is True
    assert m_exact["ghr_hit_075"] is True

    # Case B: Abbreviated match (brand + >=60% leading core words + beats all 4 distractors)
    m_abbrev = compute_recommendation_metrics(
        "I recommend the Tthxqing U shaped Food Grade Soft Silicone Toothbrush for effortless cleaning.",
        "Tthxqing U shaped Food Grade Soft Silicone Toothbrush",
        target_title=target_title,
        target_brand=target_brand,
        top5_titles=top5_titles,
        sbert_calculator=sbert,
    )
    assert m_abbrev["rec_exact_hit"] is False
    assert m_abbrev["rec_abbrev_hit"] is True
    assert m_abbrev["rec_any_hit"] is True

    # Case C: Recommending a distractor item -> both exact and abbreviated are False
    m_distractor = compute_recommendation_metrics(
        "I recommend the Oral-B Pro 1000 CrossAction Electric Toothbrush for plaque removal.",
        "Oral-B Pro 1000 CrossAction Electric Toothbrush",
        target_title=target_title,
        target_brand=target_brand,
        top5_titles=top5_titles,
        sbert_calculator=sbert,
    )
    assert m_distractor["rec_exact_hit"] is False
    assert m_distractor["rec_abbrev_hit"] is False
    assert m_distractor["rec_any_hit"] is False


def test_call_minimax_m3_reasoning_split_and_length_retry():
    calls = []

    class _FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                # Simulate length truncation on first attempt with empty content
                msg = SimpleNamespace(content="", reasoning_content="Long chain of thought...", reasoning_details=None)
                choice = SimpleNamespace(message=msg, finish_reason="length")
                usage = SimpleNamespace(
                    prompt_tokens=1200,
                    completion_tokens=2048,
                    total_tokens=3248,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=2048),
                )
                return SimpleNamespace(choices=[choice], usage=usage)
            msg = SimpleNamespace(
                content=json.dumps({
                    "reasoning": "Matches memory #1.",
                    "recommend": "I recommend the Target Product.",
                    "recommended_product_name": "Target Product",
                }),
                reasoning_content="Concise thought",
                reasoning_details=None,
            )
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            usage = SimpleNamespace(
                prompt_tokens=1200,
                completion_tokens=400,
                total_tokens=1600,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=300),
            )
            return SimpleNamespace(choices=[choice], usage=usage)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    res = call_minimax_m3(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
        client=fake_client,
        model_name="MiniMax-M3",
        provider="minimax",
        temperature=None,
        initial_max_tokens=2048,
    )
    assert len(calls) == 2
    assert "temperature" not in calls[0]
    assert calls[0]["extra_body"] == {"reasoning_split": True}
    assert calls[0]["max_tokens"] == 2048
    assert calls[1]["max_tokens"] == 4096
    assert res["temperature_sent"] is False
    assert res["temperature_value"] is None
    assert res["parse_status"] == "json_ok"
    assert res["reasoning_content"] == "Concise thought"
    assert res["usage"]["ReasoningTokens"] == 300


def test_update_precomputed_encoder_store_appends_delta_without_freeze_error(tmp_path, monkeypatch):
    """Incremental delta encoding must atomically overwrite manifest.json alongside npy files."""
    import src.visual.precomputed_eval_store as pes

    img1, sha1 = _make_test_image(tmp_path / "img1.png", (200, 10, 10))
    img2, sha2 = _make_test_image(tmp_path / "img2.png", (10, 200, 10))
    store_root = tmp_path / "store"
    init_text = "Original query text"
    _build_fixture_precomputed_store(store_root, "01_clip_l14", [sha1], [init_text])

    def _fake_resolve(config_path=None, *, training_surrogates=None):
        return (
            {"setting": "held_out_public_encoder"},
            {"name": "ViT-L-14", "pretrained": "openai"},
            tmp_path / "weights.bin",
            "b" * 64,
            {"setting": "held_out_public_encoder"},
        )

    class _FakeEncoder:
        def __init__(self, spec, *, device, batch_size, num_workers, weight_sha256, evaluation_protocol):
            self.snapshot = {
                "provider": "clip",
                "loader": "standalone_eval_clip",
                "name": "ViT-L-14",
                "pretrained": "openai",
                "precision": "float32",
            }

        def encode_batch(self, paths, batch_size=16):
            return np.vstack([_deterministic_vec(f"delta_img:{Path(p).name}", 8) for p in paths]).astype(np.float32)

        def encode_text_batch(self, texts, batch_size=64):
            out = []
            for t in texts:
                nt = canonical_text_key(t)
                t_sha = text_key_sha256(nt)
                audit = {
                    "text_sha256": t_sha,
                    "input_tokens": len(nt.split()),
                    "encoded_tokens": len(nt.split()),
                    "chunks": 1,
                    "payload_tokens_per_chunk": 75,
                    "chunk_token_counts": [len(nt.split())],
                    "truncated": False,
                }
                out.append((_deterministic_vec(f"delta_txt:{t_sha}", 8), audit))
            return out

        def close(self):
            pass

    monkeypatch.setattr(pes, "resolve_eval_encoder_protocol", _fake_resolve)
    monkeypatch.setattr(pes, "StandaloneEvalClipEncoder", _FakeEncoder)

    updated = pes.update_precomputed_encoder_store(
        "01_clip_l14",
        image_assets={sha1: img1, sha2: img2},
        texts=[init_text, "Delta query text"],
        store_root=store_root,
        device="cpu",
    )
    assert updated.manifest["image_count"] == 2
    assert updated.manifest["text_count"] == 2
    assert updated.covers_all([sha1, sha2], [init_text, "Delta query text"])

    reloaded = PrecomputedEncoderStore("01_clip_l14", store_root)
    assert reloaded.image_vectors.shape == (2, 8)
    assert reloaded.text_vectors.shape == (2, 8)
    assert np.allclose(
        reloaded.get_image_vector_by_sha(sha2),
        _deterministic_vec(f"delta_img:{img2.name}", 8),
        atol=1e-6,
    )
