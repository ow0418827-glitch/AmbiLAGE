"""Precomputed 3-encoder vector store and Zero-GPU offline evaluation encoder.

Precomputes and persists normalized float32 embeddings for:
  1. Memory bank images (clean base pools + frozen modified/attack images) and
     memory product feature/title texts across evaluation bases.
  2. Test text query suites (training_oracle_v4, overlap_30_v4, zero_overlap_v4,
     and base formal evaluation_inputs) across all 9 product groups.
  3. GHR target recommendation sentences on sentence-transformers/all-MiniLM-L6-v2.

Once precomputed, both upstream retrieval evaluation (fast_evaluate_panel.py /
fast_stage2_panel.py / shared_evaluation.py) and downstream MiniMax M3 recommendation
evaluation execute in pure CPU NumPy mode without loading encoder models or using GPU.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from src.experiments import shared_evaluation
from src.experiments.visual_inception_category_evaluation import (
    QUERY_SUITE_ROOTS,
    TRANSFER_PANEL_ENCODERS,
)
from src.paths import (
    EVALUATION_BASE,
    MODEL_ROOT,
    OUTPUT,
    PRIVATE,
    PROJECT,
    RESOURCES,
    VISUAL_CONFIGS,
    load_evaluation_bases,
    resolve_evaluation_base,
)
from src.visual.audit import digest, read_json, save_json
from src.visual.encoders.backbone import resolve_hf_cached_weight
from src.visual.encoders.eval_clip import StandaloneEvalClipEncoder
from src.visual.encoders.joint import _text_digest, normalized, validated_image_weight
from src.visual.image_utils import sha256_file
from src.visual.visual_retrieval import (
    memory_index_text,
    pillow_decoder_signature,
    resolve_image,
)

PRECOMPUTED_STORE_SCHEMA = 1
DEFAULT_PRECOMPUTED_STORE_DIR = OUTPUT / "cache" / "precomputed_eval_vectors"

ENCODER_SPEC_BY_SLUG: dict[str, dict[str, Any]] = {
    slug: {
        "slug": slug,
        "label": label,
        "surrogate_pair": list(pair),
        "matched_config": str((VISUAL_CONFIGS / matched_cfg).resolve()),
        "heldout_config": str((VISUAL_CONFIGS / heldout_cfg).resolve()),
    }
    for slug, label, pair, matched_cfg, heldout_cfg in TRANSFER_PANEL_ENCODERS
}


def canonical_text_key(text: str | None) -> str:
    """Match StandaloneEvalClipEncoder._tokenize_text_chunks text normalization."""
    return str(text or " ")


def text_key_sha256(text: str | None) -> str:
    """Compute SHA256 matching StandaloneEvalClipEncoder text_sha256."""
    return _text_digest(canonical_text_key(text))


def resolve_eval_encoder_protocol(
    config_path: str | Path | None = None,
    *,
    training_surrogates: list[dict] | None = None,
) -> tuple[dict, dict, Path, str, dict]:
    """Validate evaluator configuration and compute evaluation_protocol without loading model weights."""
    private = PRIVATE / "evaluator.json"
    settings_path = Path(
        config_path or (private if private.is_file() else VISUAL_CONFIGS / "evaluator.example.json")
    ).resolve()
    settings = read_json(settings_path)
    setting = settings.get("setting")
    if setting not in ("held_out_public_encoder", "matched_public_encoder"):
        raise ValueError("评估编码器须显式标明独立留出或同代理评估设置")
    if setting == "matched_public_encoder" and config_path is None:
        raise ValueError("同代理评估只能由本次实验显式配置启用")
    if settings.get("provider") == "gme":
        raise ValueError("Precomputed CLIP/SigLIP store 仅支持 CLIP/OpenCLIP/SigLIP 编码器")

    actual_context = training_surrogates is not None
    training_specs = (
        shared_evaluation._actual_training_identities(training_surrogates)
        if actual_context
        else shared_evaluation._builtin_training_identities()
    )
    identity_scope = (
        "actual_frozen_surrogates" if actual_context else "all_builtin_surrogate_configurations"
    )
    identity_audit = dict(
        training_identity_scope=identity_scope,
        training_surrogates_sha256=digest(training_specs),
    )

    spec = dict(settings["surrogate"])
    matching = [
        i
        for i, m in enumerate(training_specs)
        if (m["name"], m["pretrained"]) == (spec["name"], spec["pretrained"])
    ]
    if matching and setting == "held_out_public_encoder":
        raise ValueError("正式留出检索器与训练代理相同，不能作为黑盒迁移评估")
    if not matching and setting == "matched_public_encoder":
        raise ValueError("同代理评估配置与实际训练代理身份不符")

    if spec.get("weight_path"):
        raw_weight = Path(spec["weight_path"])
        if raw_weight.is_absolute():
            weight = raw_weight
        else:
            rel_settings = settings_path.parent / raw_weight
            rel_model_root = MODEL_ROOT / raw_weight
            weight = rel_settings if rel_settings.is_file() else rel_model_root
        if not weight.is_file() and weight.name in (
            "open_clip_model.safetensors",
            "open_clip_pytorch_model.safetensors",
        ):
            weight = next(
                (
                    weight.parent / name
                    for name in (
                        "open_clip_model.safetensors",
                        "open_clip_pytorch_model.safetensors",
                    )
                    if (weight.parent / name).is_file()
                ),
                weight,
            )
    else:
        from src.model_registry import resolve_model

        try:
            directory = Path(resolve_model("eval_g14")) if spec["name"] == "ViT-g-14" else None
        except FileNotFoundError:
            directory = None
        weight = next(
            (
                directory / fn
                for fn in (
                    "open_clip_model.safetensors",
                    "open_clip_pytorch_model.safetensors",
                    "open_clip_pytorch_model.bin",
                )
                if directory is not None and (directory / fn).is_file()
            ),
            None,
        )
        if weight is None:
            weight = next(
                (
                    p
                    for fn in (
                        "open_clip_model.safetensors",
                        "open_clip_pytorch_model.safetensors",
                        "open_clip_pytorch_model.bin",
                    )
                    if (p := resolve_hf_cached_weight(spec["hf_repo"], filename=fn)) is not None
                ),
                None,
            )
        if weight is None and spec.get("hf_repo"):
            try:
                repo_dir = Path(resolve_model(spec["hf_repo"]))
            except (FileNotFoundError, KeyError, ValueError):
                repo_dir = None
            weight = next(
                (
                    repo_dir / fn
                    for fn in (
                        "open_clip_model.safetensors",
                        "open_clip_pytorch_model.safetensors",
                        "open_clip_pytorch_model.bin",
                    )
                    if repo_dir is not None and (repo_dir / fn).is_file()
                ),
                None,
            )

    if weight is None or not weight.is_file():
        raise FileNotFoundError("缺少评估检索器的本地权重，请先显式准备")

    chosen_hash = sha256_file(weight)
    if actual_context or setting == "matched_public_encoder":
        weight_matches = (
            [i for i, row in enumerate(training_specs) if row.get("weight_sha256") == chosen_hash]
            if actual_context
            else []
        )
        if weight_matches and setting == "held_out_public_encoder":
            raise ValueError("正式留出检索器与训练代理使用相同权重，不能作为黑盒迁移评估")
        matching = [
            i
            for i in matching
            if (
                training_specs[i]["weight_sha256"]
                if actual_context
                else shared_evaluation._builtin_weight_sha256(training_specs[i])
            )
            == chosen_hash
        ]
        if setting == "matched_public_encoder" and not matching:
            raise ValueError("同代理评估的实际权重与训练代理不符")

    spec["weight_path"] = str(weight.resolve())
    if spec.get("tokenizer_path") and not Path(spec["tokenizer_path"]).is_absolute():
        rel_settings = (settings_path.parent / spec["tokenizer_path"]).resolve()
        rel_model_root = (MODEL_ROOT / spec["tokenizer_path"]).resolve()
        spec["tokenizer_path"] = str(rel_settings if rel_settings.exists() else rel_model_root)
    if not spec.get("tokenizer_path") and (weight.parent / "tokenizer_config.json").is_file():
        spec["tokenizer_path"] = str(weight.parent)

    evaluation_protocol = dict(
        setting=setting,
        matches_training_surrogate=bool(matching),
        matching_training_surrogate_indices=matching,
        black_box_transfer_claimed=setting == "held_out_public_encoder",
        tf32_enabled=False,
        precision="float32",
        **identity_audit,
    )
    return settings, spec, weight.resolve(), chosen_hash, evaluation_protocol


def infer_encoder_slug_from_config(config_path: str | Path | None) -> str | None:
    """Resolve transfer panel encoder slug ('01_clip_l14', etc.) from an evaluator config JSON."""
    if config_path is None:
        return None
    p = Path(config_path).resolve()
    if not p.is_file():
        return None
    doc = read_json(p)
    surrogate = doc.get("surrogate") or {}
    pair = (surrogate.get("name"), surrogate.get("pretrained"))
    for slug, info in ENCODER_SPEC_BY_SLUG.items():
        if tuple(info["surrogate_pair"]) == pair:
            return slug
    return None


class PrecomputedEncoderStore:
    """Read-only or updatable disk store of precomputed image and text vectors for one encoder."""

    def __init__(
        self,
        encoder_slug: str,
        store_root: str | Path = DEFAULT_PRECOMPUTED_STORE_DIR,
    ) -> None:
        if encoder_slug not in ENCODER_SPEC_BY_SLUG:
            raise ValueError(f"Unknown transfer panel encoder slug: {encoder_slug}")
        self.encoder_slug = encoder_slug
        self.store_root = Path(store_root).resolve()
        self.encoder_dir = self.store_root / encoder_slug
        self.manifest_path = self.encoder_dir / "manifest.json"
        self.image_vectors_path = self.encoder_dir / "image_vectors.npy"
        self.text_vectors_path = self.encoder_dir / "text_vectors.npy"

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Precomputed vector store not found for {encoder_slug} at {self.manifest_path}. "
                "Run `python tools/precompute_eval_vectors.py` first."
            )
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("schema_version") != PRECOMPUTED_STORE_SCHEMA:
            raise ValueError(f"Unsupported precomputed vector store schema: {self.manifest.get('schema_version')}")
        if self.manifest.get("encoder_slug") != encoder_slug:
            raise ValueError("Precomputed store encoder_slug mismatch")

        if sha256_file(self.image_vectors_path) != self.manifest.get("image_vectors_sha256"):
            raise ValueError(f"Image vector matrix SHA256 mismatch in {self.image_vectors_path}")
        if sha256_file(self.text_vectors_path) != self.manifest.get("text_vectors_sha256"):
            raise ValueError(f"Text vector matrix SHA256 mismatch in {self.text_vectors_path}")

        self.image_vectors = np.load(self.image_vectors_path, allow_pickle=False)
        self.text_vectors = np.load(self.text_vectors_path, allow_pickle=False)
        self.image_shas: list[str] = list(self.manifest["image_shas"])
        self.text_shas: list[str] = list(self.manifest["text_shas"])
        if self.image_vectors.shape[0] != len(self.image_shas):
            raise ValueError("Image vector row count does not match manifest image_shas")
        if self.text_vectors.shape[0] != len(self.text_shas):
            raise ValueError("Text vector row count does not match manifest text_shas")

        self._image_index: dict[str, int] = {sha: idx for idx, sha in enumerate(self.image_shas)}
        self._text_index: dict[str, int] = {sha: idx for idx, sha in enumerate(self.text_shas)}
        self.text_audits: dict[str, dict] = dict(self.manifest.get("text_audits") or {})
        self.core_encoder_snapshot: dict[str, Any] = copy.deepcopy(self.manifest["core_encoder_snapshot"])
        self.weight_sha256: str = self.manifest["weight_sha256"]

    @classmethod
    def exists(cls, encoder_slug: str, store_root: str | Path = DEFAULT_PRECOMPUTED_STORE_DIR) -> bool:
        enc_dir = Path(store_root).resolve() / encoder_slug
        return (
            (enc_dir / "manifest.json").is_file()
            and (enc_dir / "image_vectors.npy").is_file()
            and (enc_dir / "text_vectors.npy").is_file()
        )

    def has_image_sha(self, image_sha256: str) -> bool:
        return image_sha256 in self._image_index

    def has_text(self, text: str) -> bool:
        return text_key_sha256(text) in self._text_index

    def missing_image_shas(self, image_shas: Iterable[str]) -> list[str]:
        return sorted({sha for sha in image_shas if sha not in self._image_index})

    def missing_texts(self, texts: Iterable[str]) -> list[str]:
        missing_by_sha: dict[str, str] = {}
        for t in texts:
            norm_t = canonical_text_key(t)
            t_sha = text_key_sha256(norm_t)
            if t_sha not in self._text_index and t_sha not in missing_by_sha:
                missing_by_sha[t_sha] = norm_t
        return [missing_by_sha[k] for k in sorted(missing_by_sha)]

    def covers_all(self, image_shas: Iterable[str], texts: Iterable[str]) -> bool:
        return not self.missing_image_shas(image_shas) and not self.missing_texts(texts)

    def get_image_vector_by_sha(self, image_sha256: str) -> np.ndarray:
        idx = self._image_index.get(image_sha256)
        if idx is None:
            raise KeyError(f"[{self.encoder_slug}] Image SHA256 not found in precomputed store: {image_sha256}")
        return self.image_vectors[idx].copy()

    def get_text_vector(self, text: str) -> tuple[np.ndarray, dict]:
        norm_t = canonical_text_key(text)
        t_sha = text_key_sha256(norm_t)
        idx = self._text_index.get(t_sha)
        if idx is None:
            raise KeyError(f"[{self.encoder_slug}] Text not found in precomputed store: {norm_t[:80]!r}")
        audit = copy.deepcopy(self.text_audits.get(t_sha))
        return self.text_vectors[idx].copy(), audit

    def rank_bank_queries(
        self,
        bank_entries: Sequence[dict],
        queries: Sequence[dict | str],
        *,
        pool_path: str | Path,
        key_mode: str = "image",
        key_image_weight: float = 1.0,
        index_text: str = "product_feature",
        source_memory_id: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Pure CPU Zero-GPU retrieval over a memory bank for a list of queries.

        Returns exact cosine similarity scores, 1-indexed source_rank, and Top-K entries
        matching VisualMemoryRetriever.rank bit-for-bit.
        """
        if key_mode not in ("image", "joint"):
            raise ValueError("key_mode must be 'image' or 'joint'")
        weight = 1.0 if key_mode == "image" else validated_image_weight(key_image_weight)
        pool_path = Path(pool_path).resolve()

        bank_vecs: list[np.ndarray] = []
        entry_ids: list[str] = []
        for entry in bank_entries:
            entry_ids.append(entry["Id"])
            rec = entry.get("RetrievalImage") if entry.get("RetrievalImage") is not None else entry.get("PrimaryImage")
            img_sha = rec.get("SHA256") if isinstance(rec, dict) else None
            if not img_sha:
                img_path = resolve_image(entry, pool_path)
                img_sha = sha256_file(img_path)
            img_vec = self.get_image_vector_by_sha(img_sha)
            if key_mode == "image" or weight == 1.0:
                vec = img_vec
            else:
                txt = memory_index_text(entry, index_text)
                if not str(txt or "").strip():
                    vec = img_vec
                else:
                    txt_vec, _ = self.get_text_vector(txt)
                    vec = normalized(img_vec + txt_vec) if weight == 0.5 else normalized(weight * img_vec + (1.0 - weight) * txt_vec)
            bank_vecs.append(vec)

        bank_matrix = np.vstack(bank_vecs).astype(np.float32, copy=False)
        source_idx = entry_ids.index(source_memory_id) if source_memory_id is not None else None

        query_results: list[dict[str, Any]] = []
        score_rows: list[np.ndarray] = []
        for q_idx, q in enumerate(queries):
            if isinstance(q, str):
                instruction = q
                task_id = f"query_{q_idx + 1:02d}"
                q_meta = {}
            else:
                instruction = q["instruction"]
                task_id = q.get("task_id") or q.get("id") or f"query_{q_idx + 1:02d}"
                q_meta = {k: v for k, v in q.items() if k not in ("instruction", "task_id")}
            q_vec, q_audit = self.get_text_vector(instruction)
            scores_f32 = bank_matrix @ q_vec
            scores_f64 = scores_f32.astype(np.float64)
            score_rows.append(scores_f64)

            order = np.argsort(-scores_f64, kind="stable").tolist()
            top_indices = order[:top_k]
            source_rank = (order.index(source_idx) + 1) if source_idx is not None else None
            source_score = float(scores_f64[source_idx]) if source_idx is not None else None

            query_results.append({
                "task_id": task_id,
                "instruction": instruction,
                "source_rank": source_rank,
                "source_score": source_score,
                "source_in_top5": (source_rank <= 5) if source_rank is not None else None,
                "top_indices": top_indices,
                "top_memory_ids": [entry_ids[i] for i in top_indices],
                "top_scores": [float(scores_f64[i]) for i in top_indices],
                "top_entries": [bank_entries[i] for i in top_indices],
                "query_encoding_audit": q_audit,
                **q_meta,
            })

        return {
            "encoder_slug": self.encoder_slug,
            "key_mode": key_mode,
            "key_image_weight": weight,
            "index_text": index_text,
            "memory_ids": entry_ids,
            "source_memory_id": source_memory_id,
            "scores": np.vstack(score_rows) if score_rows else np.zeros((0, len(entry_ids)), dtype=np.float64),
            "queries": query_results,
        }


class OfflinePrecomputedEvalEncoder:
    """Zero-GPU drop-in replacement for StandaloneEvalClipEncoder backed by PrecomputedEncoderStore."""

    supports_image_weight = True

    def __init__(
        self,
        store: PrecomputedEncoderStore,
        *,
        batch_size: int = 16,
        device_label: str = "cuda",
        evaluation_protocol: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.batch_size = max(1, int(batch_size))
        self.closed = False
        self.last_encoding_audit: dict[str, Any] | None = None
        self._stat_hashes: dict[tuple[str, int, int], str] = {}

        # Match StandaloneEvalClipEncoder.snapshot structure & keys so model_sha256 is identical.
        snap = copy.deepcopy(store.core_encoder_snapshot)
        snap["batch_size"] = self.batch_size
        snap["device"] = str(device_label)
        if evaluation_protocol is not None:
            snap["evaluation_protocol"] = copy.deepcopy(evaluation_protocol)
        self.snapshot = snap

    def _assert_open(self) -> None:
        if self.closed:
            raise RuntimeError("OfflinePrecomputedEvalEncoder is closed")

    def _sha(self, path: str | Path) -> str:
        p = Path(path).resolve()
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        val = self._stat_hashes.get(key)
        if val is None:
            val = sha256_file(p)
            self._stat_hashes[key] = val
        return val

    def encode(self, path: str | Path) -> np.ndarray:
        self._assert_open()
        img_sha = self._sha(path)
        return self.store.get_image_vector_by_sha(img_sha)

    def encode_batch(self, paths: Iterable[str | Path], *, batch_size: int | None = None) -> list[np.ndarray]:
        self._assert_open()
        return [self.encode(p) for p in paths]

    def encode_text(self, text: str) -> np.ndarray:
        self._assert_open()
        vec, audit = self.store.get_text_vector(text)
        self.last_encoding_audit = audit
        return vec

    def encode_text_batch(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 64,
    ) -> list[tuple[np.ndarray, dict]]:
        self._assert_open()
        return [self.store.get_text_vector(t) for t in texts]

    def encode_joint(self, text: str, path: str | Path, image_weight: float = 0.5) -> np.ndarray:
        self._assert_open()
        weight = validated_image_weight(image_weight)
        if not str(text or "").strip():
            if weight == 0:
                raise ValueError("image_weight=0 requires nonempty text")
            self.last_encoding_audit = {
                "text_sha256": _text_digest(str(text or "")),
                "input_tokens": 0,
                "encoded_tokens": 0,
                "chunks": 0,
                "truncated": False,
            }
            return self.encode(path)
        if weight == 0:
            return self.encode_text(text)
        image = self.encode(path)
        if weight == 1:
            self.last_encoding_audit = {
                "text_sha256": _text_digest(str(text)),
                "text_skipped": True,
                "reason": "image_weight_is_one",
                "encoded_tokens": 0,
                "chunks": 0,
                "truncated": False,
            }
            return image
        text_vector = self.encode_text(text)
        if weight == 0.5:
            return normalized(image + text_vector)
        return normalized(weight * image + (1 - weight) * text_vector)

    def close(self) -> None:
        self.closed = True


def _atomic_save_npy(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        with tmp.open("wb") as f:
            np.save(f, np.asarray(array, dtype=np.float32), allow_pickle=False)
        tmp.replace(path)
    finally:
        if tmp.is_file():
            tmp.unlink()
    return sha256_file(path)


def update_precomputed_encoder_store(
    encoder_slug: str,
    *,
    image_assets: Mapping[str, Path],
    texts: Iterable[str],
    store_root: str | Path = DEFAULT_PRECOMPUTED_STORE_DIR,
    device: str = "cuda",
    batch_size: int = 16,
    text_batch_size: int = 64,
    num_workers: int = 4,
    metadata_extra: dict[str, Any] | None = None,
) -> PrecomputedEncoderStore:
    """Encode any missing images or texts for one encoder on GPU and persist into store_root."""
    if encoder_slug not in ENCODER_SPEC_BY_SLUG:
        raise ValueError(f"Unknown encoder slug: {encoder_slug}")
    spec_info = ENCODER_SPEC_BY_SLUG[encoder_slug]
    store_root = Path(store_root).resolve()
    enc_dir = store_root / encoder_slug
    enc_dir.mkdir(parents=True, exist_ok=True)

    existing_img_map: dict[str, np.ndarray] = {}
    existing_txt_map: dict[str, np.ndarray] = {}
    existing_audits: dict[str, dict] = {}
    core_snapshot: dict[str, Any] | None = None
    weight_sha: str | None = None

    if PrecomputedEncoderStore.exists(encoder_slug, store_root):
        current = PrecomputedEncoderStore(encoder_slug, store_root)
        for sha, vec in zip(current.image_shas, current.image_vectors):
            existing_img_map[sha] = vec.copy()
        for sha, vec in zip(current.text_shas, current.text_vectors):
            existing_txt_map[sha] = vec.copy()
        existing_audits.update(current.text_audits)
        core_snapshot = current.core_encoder_snapshot
        weight_sha = current.weight_sha256

    missing_img_shas = sorted(sha for sha in image_assets if sha not in existing_img_map)
    requested_texts_by_sha: dict[str, str] = {}
    for t in texts:
        norm_t = canonical_text_key(t)
        t_sha = text_key_sha256(norm_t)
        if t_sha not in existing_txt_map and t_sha not in requested_texts_by_sha:
            requested_texts_by_sha[t_sha] = norm_t
    missing_txt_shas = sorted(requested_texts_by_sha)

    if missing_img_shas or missing_txt_shas or core_snapshot is None:
        _, spec, _, resolved_weight_sha, _ = resolve_eval_encoder_protocol(
            spec_info["matched_config"], training_surrogates=None
        )
        encoder = StandaloneEvalClipEncoder(
            spec,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            weight_sha256=resolved_weight_sha,
            evaluation_protocol=None,
        )
        try:
            snap = copy.deepcopy(encoder.snapshot)
            snap.pop("evaluation_protocol", None)
            core_snapshot = snap
            weight_sha = resolved_weight_sha

            if missing_img_shas:
                ordered_paths = [Path(image_assets[sha]).resolve() for sha in missing_img_shas]
                new_img_vecs = encoder.encode_batch(ordered_paths, batch_size=batch_size)
                for sha, vec in zip(missing_img_shas, new_img_vecs):
                    existing_img_map[sha] = np.asarray(vec, dtype=np.float32)

            if missing_txt_shas:
                ordered_texts = [requested_texts_by_sha[sha] for sha in missing_txt_shas]
                new_txt_pairs = encoder.encode_text_batch(ordered_texts, batch_size=text_batch_size)
                for sha, (vec, audit) in zip(missing_txt_shas, new_txt_pairs):
                    existing_txt_map[sha] = np.asarray(vec, dtype=np.float32)
                    existing_audits[sha] = audit
        finally:
            encoder.close()

    sorted_img_shas = sorted(existing_img_map)
    dim = int(next(iter(existing_img_map.values())).shape[0]) if existing_img_map else (
        int(next(iter(existing_txt_map.values())).shape[0]) if existing_txt_map else 768
    )
    img_matrix = (
        np.vstack([existing_img_map[s] for s in sorted_img_shas]).astype(np.float32, copy=False)
        if sorted_img_shas
        else np.zeros((0, dim), dtype=np.float32)
    )
    sorted_txt_shas = sorted(existing_txt_map)
    txt_matrix = (
        np.vstack([existing_txt_map[s] for s in sorted_txt_shas]).astype(np.float32, copy=False)
        if sorted_txt_shas
        else np.zeros((0, dim), dtype=np.float32)
    )

    img_npy_sha = _atomic_save_npy(enc_dir / "image_vectors.npy", img_matrix)
    txt_npy_sha = _atomic_save_npy(enc_dir / "text_vectors.npy", txt_matrix)

    manifest = {
        "schema_version": PRECOMPUTED_STORE_SCHEMA,
        "encoder_slug": encoder_slug,
        "encoder_label": spec_info["label"],
        "surrogate_pair": spec_info["surrogate_pair"],
        "weight_sha256": weight_sha,
        "vector_dim": dim,
        "image_count": len(sorted_img_shas),
        "text_count": len(sorted_txt_shas),
        "image_vectors_sha256": img_npy_sha,
        "text_vectors_sha256": txt_npy_sha,
        "image_decoder": pillow_decoder_signature(),
        "core_encoder_snapshot": core_snapshot,
        "image_shas": sorted_img_shas,
        "text_shas": sorted_txt_shas,
        "text_audits": {s: existing_audits[s] for s in sorted_txt_shas if s in existing_audits},
    }
    if metadata_extra:
        manifest.update(metadata_extra)
    save_json(enc_dir / "manifest.json", manifest)
    return PrecomputedEncoderStore(encoder_slug, store_root)


def make_offline_or_standalone_eval_encoder(
    encoder_slug: str | None = None,
    *,
    device: str = "cuda",
    config_path: str | Path | None = None,
    training_surrogates: list[dict] | None = None,
    batch_size: int = 16,
    text_batch_size: int = 64,
    num_workers: int = 4,
    required_image_assets: Mapping[str, Path] | None = None,
    required_texts: Iterable[str] | None = None,
    store_root: str | Path = DEFAULT_PRECOMPUTED_STORE_DIR,
    require_offline: bool = False,
    auto_update_store: bool = True,
) -> OfflinePrecomputedEvalEncoder | StandaloneEvalClipEncoder:
    """Return Zero-GPU OfflinePrecomputedEvalEncoder whenever precomputed vectors cover required assets."""
    slug = encoder_slug or infer_encoder_slug_from_config(config_path)
    _, _, _, _, evaluation_protocol = resolve_eval_encoder_protocol(
        config_path, training_surrogates=training_surrogates
    )

    if slug is not None and PrecomputedEncoderStore.exists(slug, store_root):
        store = PrecomputedEncoderStore(slug, store_root)
        req_imgs = list(required_image_assets.keys()) if required_image_assets else []
        req_txts = list(required_texts) if required_texts else []
        missing_imgs = store.missing_image_shas(req_imgs)
        missing_txts = store.missing_texts(req_txts)
        if not missing_imgs and not missing_txts:
            print(
                f"[OFFLINE PRECOMPUTED {slug}] 100% vector hit "
                f"(images={len(req_imgs)}/{store.manifest['image_count']}, "
                f"texts={len(req_txts)}/{store.manifest['text_count']}) -> ZERO model load, ZERO GPU!",
                flush=True,
            )
            return OfflinePrecomputedEvalEncoder(
                store,
                batch_size=batch_size,
                device_label=device,
                evaluation_protocol=evaluation_protocol,
            )
        if require_offline:
            raise RuntimeError(
                f"[OFFLINE PRECOMPUTED {slug}] Missing {len(missing_imgs)} images and "
                f"{len(missing_txts)} texts in strict offline mode. Run tools/precompute_eval_vectors.py first."
            )
        if auto_update_store and required_image_assets is not None:
            print(
                f"[PRECOMPUTED UPDATE {slug}] Encoding only delta assets: "
                f"missing_images={len(missing_imgs)}, missing_texts={len(missing_txts)}",
                flush=True,
            )
            store = update_precomputed_encoder_store(
                slug,
                image_assets=required_image_assets,
                texts=req_txts,
                store_root=store_root,
                device=device,
                batch_size=batch_size,
                text_batch_size=text_batch_size,
                num_workers=num_workers,
            )
            return OfflinePrecomputedEvalEncoder(
                store,
                batch_size=batch_size,
                device_label=device,
                evaluation_protocol=evaluation_protocol,
            )

    if require_offline:
        raise FileNotFoundError(
            f"Precomputed vector store for {slug or config_path} does not exist at {store_root} "
            "and require_offline=True."
        )

    from src.visual.encoders.eval_clip import make_standalone_eval_encoder

    return make_standalone_eval_encoder(
        device=device,
        config_path=config_path,
        training_surrogates=training_surrogates,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def collect_all_project_precompute_assets(
    *,
    bases: Iterable[str | Path] | None = None,
    query_suites: Iterable[str] | None = None,
    extra_image_roots: Iterable[str | Path] | None = None,
    include_all_output_frozen: bool = True,
) -> dict[str, Any]:
    """Scan evaluation bases, frozen attack images, and test query suites for all unique images & texts."""
    image_assets: dict[str, Path] = {}
    texts: set[str] = {" "}
    covered_bases: dict[str, dict[str, Any]] = {}
    covered_suites: dict[str, dict[str, Any]] = {}
    covered_frozen_roots: list[str] = []

    if bases is None:
        cfg = load_evaluation_bases()
        base_names = list((cfg.get("bases") or {}).keys()) or ["original_9groups"]
        resolved_bases = [(name, resolve_evaluation_base(name)) for name in base_names]
    else:
        resolved_bases = [(str(b), resolve_evaluation_base(b)) for b in bases]

    for b_label, base_path in resolved_bases:
        if not (base_path / "manifest.json").is_file():
            continue
        manifest = read_json(base_path / "manifest.json")
        pool_rel = manifest.get("shared_memory", {}).get("pool_file")
        if not pool_rel:
            continue
        pool_path = (base_path / pool_rel).resolve()
        pool_doc = read_json(pool_path)
        entries = pool_doc.get("entries", [])
        base_img_count = 0
        for entry in entries:
            img_path = resolve_image(entry, pool_path)
            if img_path is not None and img_path.is_file():
                rec = entry.get("RetrievalImage") if entry.get("RetrievalImage") is not None else entry.get("PrimaryImage")
                img_sha = (rec or {}).get("SHA256") or sha256_file(img_path)
                image_assets[img_sha] = img_path
                base_img_count += 1
            for mode in ("product_feature", "title"):
                try:
                    t_val = memory_index_text(entry, mode)
                    if t_val:
                        texts.add(t_val)
                except ValueError:
                    pass

        # Also collect base formal evaluation_inputs queries
        base_q_count = 0
        for group_key in manifest.get("groups", {}):
            q_file = base_path / "evaluation_inputs" / group_key / "queries.json"
            if q_file.is_file():
                q_doc = read_json(q_file)
                for instr in q_doc.get("instructions", []):
                    if isinstance(instr, str) and instr.strip():
                        texts.add(instr)
                        base_q_count += 1

        covered_bases[b_label] = {
            "base_path": str(base_path),
            "pool_path": str(pool_path),
            "pool_sha256": sha256_file(pool_path),
            "entry_count": len(entries),
            "image_count": base_img_count,
            "formal_query_count": base_q_count,
        }

    # Collect test query suites (training_oracle_v4, overlap_30_v4, zero_overlap_v4, etc.)
    suites_to_scan = list(query_suites) if query_suites is not None else [
        "training_oracle_v4",
        "overlap_30_v4",
        "zero_overlap_v4",
    ]
    for suite_name in suites_to_scan:
        suite_root = QUERY_SUITE_ROOTS.get(suite_name)
        if suite_root is None or not Path(suite_root).is_dir():
            continue
        suite_root = Path(suite_root).resolve()
        manifest_file = suite_root / "manifest.json"
        if not manifest_file.is_file():
            continue
        s_manifest = read_json(manifest_file)
        suite_q_count = 0
        for g_key, g_info in (s_manifest.get("groups") or {}).items():
            if isinstance(g_info, dict) and g_info.get("path"):
                q_path = (suite_root / g_info["path"]).resolve()
            else:
                q_path = (suite_root / g_key / "queries.json").resolve()
            if q_path.is_file():
                q_doc = read_json(q_path)
                for instr in q_doc.get("instructions", []):
                    if isinstance(instr, str) and instr.strip():
                        texts.add(instr)
                        suite_q_count += 1
        covered_suites[suite_name] = {
            "query_root": str(suite_root),
            "manifest_sha256": sha256_file(manifest_file),
            "query_count": suite_q_count,
        }

    # Collect frozen attack images
    frozen_files: list[Path] = []
    if include_all_output_frozen and OUTPUT.is_dir():
        frozen_files.extend(sorted(OUTPUT.glob("**/FROZEN_IMAGE.json")))
    if extra_image_roots:
        for root in extra_image_roots:
            rp = Path(root).resolve()
            if rp.is_file() and rp.name == "FROZEN_IMAGE.json":
                frozen_files.append(rp)
            elif rp.is_dir():
                frozen_files.extend(sorted(rp.glob("**/FROZEN_IMAGE.json")))

    seen_frozen = set()
    for fpath in frozen_files:
        fpath = fpath.resolve()
        if fpath in seen_frozen:
            continue
        seen_frozen.add(fpath)
        try:
            fdoc = read_json(fpath)
            if fdoc.get("status") != "frozen":
                continue
            img_str = fdoc.get("image")
            if not img_str:
                continue
            img_p = Path(img_str)
            if not img_p.is_file():
                continue
            img_p = img_p.resolve()
            img_sha = fdoc.get("image_sha256") or sha256_file(img_p)
            if sha256_file(img_p) != img_sha:
                continue
            image_assets[img_sha] = img_p
            if fpath.is_relative_to(OUTPUT.resolve()):
                parts = fpath.relative_to(OUTPUT.resolve()).parts
                if len(parts) >= 3:
                    root_tag = str(Path(*parts[:-2]))
                    if root_tag not in covered_frozen_roots:
                        covered_frozen_roots.append(root_tag)
        except Exception:
            continue

    return {
        "image_assets": image_assets,
        "texts": sorted(texts),
        "covered_bases": covered_bases,
        "covered_query_suites": covered_suites,
        "covered_frozen_roots": sorted(covered_frozen_roots),
    }


def format_ghr_target_sentence(brand: str | None, title: str) -> str:
    """Construct the Visual Inception Appendix A.3 adversarial goal string G_{adv,g}.

    Template: "I recommend the {Brand} {Title}."
    If Title already starts with Brand (case-insensitive) or Brand is empty/Unknown,
    do not repeat the brand name.
    """
    clean_title = " ".join(str(title or "").split()).strip().rstrip(".")
    clean_brand = " ".join(str(brand or "").split()).strip()
    if not clean_brand or clean_brand.lower() == "unknown":
        return f"I recommend the {clean_title}."
    if clean_title.lower().startswith(clean_brand.lower()):
        return f"I recommend the {clean_title}."
    return f"I recommend the {clean_brand} {clean_title}."


def precompute_sbert_ghr_targets(
    store_root: str | Path = DEFAULT_PRECOMPUTED_STORE_DIR,
    *,
    sbert_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Precompute and save SBERT (all-MiniLM-L6-v2) target embeddings for GHR on CPU."""
    store_root = Path(store_root).resolve()
    store_root.mkdir(parents=True, exist_ok=True)
    local_sbert = Path(sbert_model_path) if sbert_model_path else (MODEL_ROOT / "all-MiniLM-L6-v2")
    model_id = str(local_sbert.resolve()) if local_sbert.is_dir() else "sentence-transformers/all-MiniLM-L6-v2"

    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(model_id, device="cpu")
    cfg = load_evaluation_bases()
    base_names = list((cfg.get("bases") or {}).keys()) or ["original_9groups"]

    sentences_by_key: dict[str, dict[str, Any]] = {}
    unique_sentences: list[str] = []

    for bname in base_names:
        bpath = resolve_evaluation_base(bname)
        if not (bpath / "manifest.json").is_file():
            continue
        manifest = read_json(bpath / "manifest.json")
        pool_path = (bpath / manifest["shared_memory"]["pool_file"]).resolve()
        pool = read_json(pool_path).get("entries", [])
        by_id = {e["Id"]: e for e in pool}
        for group, mapping in (manifest.get("shared_memory", {}).get("source_mapping") or {}).items():
            mid = mapping["source_memory_id"]
            entry = by_id[mid]
            title = entry["PrimaryImage"]["Title"]
            brand = entry.get("Brand") or entry["PrimaryImage"].get("Brand") or ""
            sentence = format_ghr_target_sentence(brand, title)
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)
            sentences_by_key[f"{bname}::{group}"] = {
                "base": bname,
                "group": group,
                "source_memory_id": mid,
                "brand": brand,
                "title": title,
                "ghr_target_sentence": sentence,
            }

    vectors = sbert.encode(unique_sentences, normalize_embeddings=True, convert_to_numpy=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    vec_by_sentence = {s: vectors[i].tolist() for i, s in enumerate(unique_sentences)}

    for info in sentences_by_key.values():
        s = info["ghr_target_sentence"]
        info["vector"] = vec_by_sentence[s]

    payload = {
        "schema_version": 1,
        "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "targets": sentences_by_key,
        "by_sentence": vec_by_sentence,
    }
    out_file = store_root / "sbert_ghr_targets.json"
    save_json(out_file, payload)
    return payload


def precompute_all_eval_vectors(
    store_root: str | Path = DEFAULT_PRECOMPUTED_STORE_DIR,
    *,
    encoder_slugs: Sequence[str] | None = None,
    bases: Iterable[str | Path] | None = None,
    query_suites: Iterable[str] | None = None,
    extra_image_roots: Iterable[str | Path] | None = None,
    include_all_output_frozen: bool = True,
    batch_sizes: Mapping[str, int] | None = None,
    text_batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
    precompute_sbert: bool = True,
) -> dict[str, Any]:
    """Precompute and store all memory bank images & test query texts across the 3 encoders."""
    store_root = Path(store_root).resolve()
    assets = collect_all_project_precompute_assets(
        bases=bases,
        query_suites=query_suites,
        extra_image_roots=extra_image_roots,
        include_all_output_frozen=include_all_output_frozen,
    )
    slugs = list(encoder_slugs) if encoder_slugs is not None else list(ENCODER_SPEC_BY_SLUG.keys())
    default_bs = {"01_clip_l14": 32, "02_openclip_bigG14": 16, "03_siglip_so400m": 16}
    if batch_sizes:
        default_bs.update({str(k): int(v) for k, v in batch_sizes.items()})

    metadata_extra = {
        "covered_bases": assets["covered_bases"],
        "covered_query_suites": assets["covered_query_suites"],
        "covered_frozen_roots": assets["covered_frozen_roots"],
    }

    summary_encoders: dict[str, Any] = {}
    for slug in slugs:
        bs = default_bs.get(slug, 16)
        print(
            f"[PRECOMPUTE {slug}] unique_images={len(assets['image_assets'])} "
            f"unique_texts={len(assets['texts'])} batch_size={bs} text_batch_size={text_batch_size}",
            flush=True,
        )
        store = update_precomputed_encoder_store(
            slug,
            image_assets=assets["image_assets"],
            texts=assets["texts"],
            store_root=store_root,
            device=device,
            batch_size=bs,
            text_batch_size=text_batch_size,
            num_workers=num_workers,
            metadata_extra=metadata_extra,
        )
        summary_encoders[slug] = {
            "encoder_slug": slug,
            "encoder_label": store.manifest["encoder_label"],
            "weight_sha256": store.weight_sha256,
            "vector_dim": store.manifest["vector_dim"],
            "image_count": store.manifest["image_count"],
            "text_count": store.manifest["text_count"],
            "image_vectors_sha256": store.manifest["image_vectors_sha256"],
            "text_vectors_sha256": store.manifest["text_vectors_sha256"],
        }

    sbert_info = None
    if precompute_sbert:
        sbert_payload = precompute_sbert_ghr_targets(store_root)
        sbert_info = {
            "sbert_model": sbert_payload["sbert_model"],
            "target_count": len(sbert_payload["targets"]),
        }

    global_manifest = {
        "schema_version": PRECOMPUTED_STORE_SCHEMA,
        "store_root": str(store_root),
        "encoders": summary_encoders,
        "sbert_ghr_targets": sbert_info,
        **metadata_extra,
    }
    save_json(store_root / "store_manifest.json", global_manifest)
    return global_manifest
