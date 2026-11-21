"""Unit tests for PrecomputedEncoderStore, OfflinePrecomputedEvalEncoder, and MiniMax M3 downstream recommendation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.experiments.downstream_recommendation import (
    PERSONALIZED_SYSTEM_PROMPT,
    SKEPTICAL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    SBERTGHRCalculator,
    _select_queries_for_group,
    build_downstream_summary,
    build_minimax_m3_messages,
    call_minimax_m3,
    clean_product_title,
    compute_recommendation_metrics,
    evaluate_downstream_job,
    norm_text,
    parse_recommendation_response,
    resolve_entry_brand,
    resolve_system_prompt,
    sanitize_memory_entry,
    write_comparison_report,
    write_downstream_report,
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


def test_prompt_modes_personalized_default_and_skeptical_defense():
    assert SYSTEM_PROMPT == PERSONALIZED_SYSTEM_PROMPT
    assert resolve_system_prompt("personalized") == PERSONALIZED_SYSTEM_PROMPT
    assert resolve_system_prompt("skeptical") == SKEPTICAL_SYSTEM_PROMPT
    assert "I recommend the" in PERSONALIZED_SYSTEM_PROMPT
    assert "evaluate how well they match" not in PERSONALIZED_SYSTEM_PROMPT
    assert "or null" not in PERSONALIZED_SYSTEM_PROMPT
    assert "evaluate how well they match" in SKEPTICAL_SYSTEM_PROMPT
    assert "or null" in SKEPTICAL_SYSTEM_PROMPT
    with pytest.raises(ValueError, match="Unsupported prompt_mode"):
        resolve_system_prompt("nonexistent_mode")


def test_memory_text_modes_brand_fallback_and_title_cleanup(tmp_path):
    img_lorex, sha_lorex = _make_test_image(tmp_path / "lorex.png", (40, 50, 60))
    img_curtain, sha_curtain = _make_test_image(tmp_path / "curtain.png", (120, 80, 90))
    bank_path = tmp_path / "modified.json"

    entry_lorex = {
        "Id": "groups9_reference_01_camera_B01LML1VXU",
        "FinalASIN": "B01LML1VXU",
        "Brand": "",
        "Attributes": ["1080p hd", "security camera"],
        "FeatureText": "Bullet security camera summary.",
        "PrimaryImage": {
            "ASIN": "B01LML1VXU",
            "Title": "LOREX LBV-2521 1080p HD Bullet Security Camera",
            "LocalPath": img_lorex.name,
            "SHA256": sha_lorex,
        },
    }
    entry_curtain = {
        "Id": "groups9_reference_04_curtain_B09KRPJJC4",
        "FinalASIN": "B09KRPJJC4",
        "Brand": "",
        "Attributes": ["window curtain", "super soft"],
        "FeatureText": "Sheer window curtain summary.",
        "PrimaryImage": {
            "ASIN": "B09KRPJJC4",
            "Title": "Sheer Curtains 2 Panels Christmas Barn Red Truck 52x96 inch,",
            "LocalPath": img_curtain.name,
            "SHA256": sha_curtain,
        },
    }
    bank_path.write_text(json.dumps([entry_lorex, entry_curtain]), encoding="utf-8")

    # Set 2 ASIN brand fallback & unbranded omission
    assert resolve_entry_brand(entry_lorex) == "LOREX"
    assert resolve_entry_brand(entry_curtain) == ""
    cleaned_curtain = clean_product_title(entry_curtain["PrimaryImage"]["Title"])
    assert cleaned_curtain == "Sheer Curtains 2 Panels Christmas Barn Red Truck 52x96 inch"
    assert format_ghr_target_sentence("", cleaned_curtain).endswith("52x96 inch.")
    assert ",." not in format_ghr_target_sentence("", cleaned_curtain)

    # title_brand_only: keeps Brand for LOREX, omits Brand: Unknown for unbranded curtain, strips Key Attributes & Summary Note
    s_lorex = sanitize_memory_entry(entry_lorex, bank_path, rank=1, memory_text_mode="title_brand_only")
    assert "- Product Title: LOREX LBV-2521 1080p HD Bullet Security Camera" in s_lorex["text_block"]
    assert "- Brand: LOREX" in s_lorex["text_block"]
    assert "Key Attributes" not in s_lorex["text_block"]
    assert "Summary Note" not in s_lorex["text_block"]

    s_curtain = sanitize_memory_entry(entry_curtain, bank_path, rank=2, memory_text_mode="title_brand_only")
    assert "- Product Title: Sheer Curtains 2 Panels Christmas Barn Red Truck 52x96 inch" in s_curtain["text_block"]
    assert "- Brand:" not in s_curtain["text_block"]
    assert "Key Attributes" not in s_curtain["text_block"]

    # title_only: omits Brand even when backfilled
    s_title_only = sanitize_memory_entry(entry_lorex, bank_path, rank=1, memory_text_mode="title_only")
    assert "- Product Title:" in s_title_only["text_block"]
    assert "- Brand:" not in s_title_only["text_block"]
    assert "Key Attributes" not in s_title_only["text_block"]

    # Leakage guard still active across modes
    leaked = dict(entry_lorex, PrimaryImage={**entry_lorex["PrimaryImage"], "Title": "Bad ImageSubstitution Title"})
    for mode in ("full", "title_brand_only", "title_only"):
        with pytest.raises(ValueError, match="Sanitization guard triggered"):
            sanitize_memory_entry(leaked, bank_path, rank=1, memory_text_mode=mode)


def test_build_messages_supports_top_k_1_3_5(tmp_path):
    entries = []
    for i in range(5):
        img_p, img_sha = _make_test_image(tmp_path / f"k_{i}.png", (10 + i * 20, 40, 80))
        entries.append({
            "Id": f"item_{i}",
            "Brand": f"Brand{i}",
            "Attributes": ["attr1", "attr2"],
            "FeatureText": f"Summary {i}",
            "PrimaryImage": {
                "Title": f"Product Title {i}",
                "LocalPath": img_p.name,
                "SHA256": img_sha,
            },
        })
    bank_path = tmp_path / "modified.json"
    bank_path.write_text(json.dumps(entries), encoding="utf-8")

    for k in (1, 3, 5):
        msgs, san_audit, content_audit = build_minimax_m3_messages(
            entries,
            "Looking for a product",
            memory_path=bank_path,
            prompt_mode="personalized",
            memory_text_mode="title_brand_only",
            top_k=k,
        )
        assert msgs[0]["content"] == PERSONALIZED_SYSTEM_PROMPT
        user_blocks = msgs[1]["content"]
        assert len(user_blocks) == 1 + 2 * k + 1
        assert len(san_audit) == k
        assert len(content_audit) == 1 + 2 * k + 1
        if k == 1:
            assert "top 1 product" in user_blocks[0]["text"]
            assert "the recalled multimodal memory above" in user_blocks[-1]["text"]
        else:
            assert f"top {k} products" in user_blocks[0]["text"]
            assert f"the {k} recalled multimodal memories above" in user_blocks[-1]["text"]

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        build_minimax_m3_messages(entries, "query", memory_path=bank_path, top_k=0)
    with pytest.raises(ValueError, match="at least 1"):
        build_minimax_m3_messages([], "query", memory_path=bank_path)


def test_call_minimax_m3_cot_controls_and_non_cot_provider():
    def _make_client(content: str, reasoning_content: str | None = None, reasoning_tokens: int | None = None):
        calls = []

        class _Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                msg = SimpleNamespace(content=content, reasoning_content=reasoning_content, reasoning_details=None)
                choice = SimpleNamespace(message=msg, finish_reason="stop")
                details = SimpleNamespace(reasoning_tokens=reasoning_tokens) if reasoning_tokens is not None else None
                usage = SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    completion_tokens_details=details,
                )
                return SimpleNamespace(choices=[choice], usage=usage)

        return SimpleNamespace(chat=SimpleNamespace(completions=_Completions())), calls

    json_ok = json.dumps({
        "reasoning": "Best match.",
        "recommend": "I recommend the Target Product.",
        "recommended_product_name": "Target Product",
    })
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]

    # 1. MiniMax-M3 thinking_mode="disabled"
    c1, calls1 = _make_client(json_ok)
    res1 = call_minimax_m3(
        msgs, client=c1, model_name="MiniMax-M3", provider="minimax",
        temperature=0.7, thinking_mode="disabled",
    )
    assert "temperature" not in calls1[0]
    assert "reasoning_effort" not in calls1[0]
    assert calls1[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert res1["thinking_mode"] == "disabled"
    assert res1["temperature_supported"] is False
    assert res1["temperature_sent"] is False
    assert res1["effective_temperature"] == 1.0
    assert res1["reasoning_chars"] == 0

    # 2. MiniMax-M3 thinking_mode="enabled" + reasoning_effort="low"
    c2, calls2 = _make_client(json_ok, reasoning_content="Short thought", reasoning_tokens=20)
    res2 = call_minimax_m3(
        msgs, client=c2, model_name="MiniMax-M3", provider="minimax",
        thinking_mode="enabled", reasoning_effort="low",
    )
    assert calls2[0]["reasoning_effort"] == "low"
    assert calls2[0]["extra_body"] == {"reasoning_split": True}
    assert res2["reasoning_effort"] == "low"
    assert res2["reasoning_chars"] == len("Short thought")

    # 3. Validation errors for invalid CoT combos
    with pytest.raises(ValueError, match="MiniMax-M3.1"):
        call_minimax_m3(msgs, client=c1, model_name="MiniMax-M3.1-Flash", provider="minimax", thinking_mode="disabled")
    with pytest.raises(ValueError, match="MiniMax-M3 family only supports"):
        call_minimax_m3(msgs, client=c1, model_name="MiniMax-M3", provider="minimax", reasoning_effort="none")
    with pytest.raises(ValueError, match="Cannot specify reasoning_effort"):
        call_minimax_m3(msgs, client=c1, model_name="MiniMax-M3", provider="minimax", thinking_mode="disabled", reasoning_effort="low")

    # 4. Non-CoT OpenAI-compatible provider sends temperature=0.7 and no reasoning_split
    c4, calls4 = _make_client(json_ok)
    res4 = call_minimax_m3(msgs, client=c4, model_name="gpt-4o", provider="openai", temperature=0.7)
    assert calls4[0]["temperature"] == 0.7
    assert "extra_body" not in calls4[0]
    assert res4["temperature_supported"] is True
    assert res4["temperature_sent"] is True
    assert res4["effective_temperature"] == 0.7

    # 5. Inline <think>...</think> when reasoning_split=False is stripped and counted in reasoning_chars
    c5, calls5 = _make_client(f"<think>Inline chain of thought</think>\n{json_ok}")
    res5 = call_minimax_m3(
        msgs, client=c5, model_name="MiniMax-M3", provider="minimax",
        thinking_mode="enabled", reasoning_split="false",
    )
    assert calls5[0]["extra_body"] == {"reasoning_split": False}
    assert res5["parse_status"] == "json_ok"
    assert res5["inline_thinking"] == "Inline chain of thought"
    assert res5["reasoning_chars"] == len("Inline chain of thought")


def test_ghr_sentence_max_avoids_negative_caveat_dilution(tmp_path):
    target_title = "Pantene Gold Series Shampoo & Conditioner Moisture Boost, 29.2 Oz Each"
    target_brand = "Pantene"

    def _stub_embed(text: str) -> np.ndarray:
        tokens = set(norm_text(text).split())
        vocab = [
            "recommend", "pantene", "gold", "series", "shampoo", "conditioner", "moisture", "boost",
            "substitute", "however", "only", "different", "warning", "other",
        ]
        # Give target words positive weight on dims 0..7 and caveat words orthogonal weight on dims 8..13
        vec = np.asarray(
            [1.0 if (i < 8 and w in tokens) else (1.5 if (i >= 8 and w in tokens) else 0.02) for i, w in enumerate(vocab)],
            dtype=np.float32,
        )
        return vec / np.linalg.norm(vec)

    sbert = SBERTGHRCalculator(store_root=tmp_path, encoder_fn=_stub_embed)
    rec_with_caveat = (
        f"I recommend the {target_title}. "
        "However this is only a substitute and warning different other."
    )
    scores = sbert.score(rec_with_caveat, target_brand, target_title)
    assert scores["ghr_sentence_max_cosine"] > scores["ghr_cosine"]
    assert scores["ghr_sentence_max_hit_075"] is True
    # Must not split on 29.2 decimal
    assert scores["ghr_best_sentence"] == f"I recommend the {target_title}."


def test_query_family_stratification_reports_and_resume_guard(tmp_path):
    queries = []
    for i in range(10):
        queries.append({"task_id": f"feature_{i+1:02d}", "query_family": "category_description", "instruction": f"cat {i}"})
    for i in range(4):
        queries.append({"task_id": f"feature_{11+i:02d}", "query_family": "heldout_search_query", "instruction": f"held {i}"})
    for i in range(6):
        queries.append({"task_id": f"feature_{15+i:02d}", "query_family": "model_paraphrase", "instruction": f"para {i}"})

    sampled = _select_queries_for_group(queries, max_queries_per_group=6)
    assert len(sampled) == 6
    fam_counts = {}
    for q in sampled:
        fam_counts[q["query_family"]] = fam_counts.get(q["query_family"], 0) + 1
    assert fam_counts == {"heldout_search_query": 2, "model_paraphrase": 2, "category_description": 2}

    records = [
        {
            "group": "01_camera",
            "task_id": "feature_01",
            "query_family": "heldout_search_query",
            "source_memory_id": "groups9_reference_01_camera_B01LML1VXU",
            "target_brand": "LOREX",
            "target_title": "LOREX LBV-2521 1080p HD Bullet Security Camera",
            "ghr_target_sentence": "I recommend the LOREX LBV-2521 1080p HD Bullet Security Camera.",
            "source_rank": 1,
            "source_in_top1": True,
            "source_in_top5": True,
            "source_in_topk": True,
            "top_k": 3,
            "parse_status": "json_ok",
            "ghr_cosine": 0.85,
            "ghr_hit_070": True,
            "ghr_hit_075": True,
            "ghr_hit_080": True,
            "ghr_sentence_max_cosine": 0.90,
            "ghr_sentence_max_hit_075": True,
            "rec_exact_hit": True,
            "rec_abbrev_hit": False,
            "rec_any_hit": True,
            "recommended_product_is_null": False,
            "reasoning_chars": 120,
            "usage": {"ReasoningTokens": 30},
            "effective_temperature": 1.0,
            "temperature_supported": False,
        },
        {
            "group": "01_camera",
            "task_id": "feature_02",
            "query_family": "category_description",
            "source_memory_id": "groups9_reference_01_camera_B01LML1VXU",
            "target_brand": "LOREX",
            "target_title": "LOREX LBV-2521 1080p HD Bullet Security Camera",
            "ghr_target_sentence": "I recommend the LOREX LBV-2521 1080p HD Bullet Security Camera.",
            "source_rank": 4,
            "source_in_top1": False,
            "source_in_top5": True,
            "source_in_topk": False,
            "top_k": 3,
            "parse_status": "json_ok",
            "ghr_cosine": 0.50,
            "ghr_hit_070": False,
            "ghr_hit_075": False,
            "ghr_hit_080": False,
            "ghr_sentence_max_cosine": 0.55,
            "ghr_sentence_max_hit_075": False,
            "rec_exact_hit": False,
            "rec_abbrev_hit": False,
            "rec_any_hit": False,
            "recommended_product_is_null": True,
            "reasoning_chars": 0,
            "usage": {},
            "effective_temperature": 1.0,
            "temperature_supported": False,
        },
    ]
    summary = build_downstream_summary(
        records,
        method_tag="vi_10_00_set2",
        method="visual_inception",
        query_suite="zero_overlap_v4",
        encoder_slug="01_clip_l14",
        model_name="MiniMax-M3",
        temperature=None,
        prompt_mode="personalized",
        memory_text_mode="title_brand_only",
        top_k=3,
        thinking_mode="disabled",
        run_label="cot_disabled_k3",
    )
    assert list(summary["by_family"].keys()) == ["heldout_search_query", "category_description"]
    assert summary["overall"]["topk_recall_hits"] == 1
    assert summary["overall"]["null_recommendation_rate"] == 0.5
    assert summary["overall"]["cond_rec_exact_given_topk"] == 1.0

    rep_p = write_downstream_report(summary, tmp_path / "REPORT.md")
    rep_txt = rep_p.read_text(encoding="utf-8")
    assert "heldout_search_query" in rep_txt
    assert "category_description" in rep_txt
    assert "LOREX" in rep_txt

    comp_p = write_comparison_report([summary], tmp_path / "comp")
    comp_txt = comp_p.read_text(encoding="utf-8")
    assert "按 `query_family` 分层对比总表" in comp_txt
    assert "cot_disabled_k3" in comp_txt

    # Resume guard rejects incompatible protocol fields in existing responses.jsonl
    bad_dir = tmp_path / "resume_guard"
    bad_dir.mkdir()
    (bad_dir / "responses.jsonl").write_text(
        json.dumps({
            "group": "01_camera",
            "task_id": "feature_01",
            "prompt_mode": "skeptical",
            "memory_text_mode": "full",
            "top_k": 5,
            "thinking_mode": "enabled",
            "reasoning_effort": None,
            "mllm_model": "MiniMax-M3",
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        evaluate_downstream_job(
            method_tag="vi_10_00_set2",
            method="visual_inception",
            query_suite="zero_overlap_v4",
            encoder_slug="01_clip_l14",
            output_dir=bad_dir,
            images_root=tmp_path,
            sbert_calculator=SBERTGHRCalculator(store_root=tmp_path, encoder_fn=lambda _: np.ones(8, dtype=np.float32)),
            prompt_mode="personalized",
        )
