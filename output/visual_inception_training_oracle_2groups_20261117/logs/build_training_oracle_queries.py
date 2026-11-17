"""Build and verify 50%/30%/20% paper-style pure-text training_oracle manifests for 07_shampoo and 09_toothbrush."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path("/content/AmbiLAGE")
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.query_contract import load_formal_queries
from src.paths import EVALUATION_BASE
from src.visual.audit import freeze_json, read_json

OUT_QUERIES = REPO_ROOT / "output/visual_inception_training_oracle_2groups_20261117/logs/queries"

ORACLE_QUERIES = {
    "07_shampoo": {
        "target_public_id": "B00BDJODWI",
        "category": "shampoo",
        "source_metadata": "resources/evaluation_base/public_metadata/product_metadata.json#B00BDJODWI",
        "breakdown": {
            "50pct_public_description_and_metadata": [
                "Aveda Scalp Benefits Balancing Shampoo 33.8 OZ",
                "Aveda Scalp Benefits Balancing Shampoo 33.8 OZ - High Performance Plant Based Haircare",
                "High performance plant based haircare shampoo by Aveda.",
                "Aveda shampoo researched and developed under a rigorous set of environmental sustainability values.",
                "Aveda products are widely known for their use of pure plant and flower essences.",
                "AVEDA by Aveda SCALP BENEFITS BALANCING SHAMPOO 33.8 OZ",
                "Aveda possesses a blend of High Performance Plant Based Haircare.",
                "Scalp Benefits Balancing Shampoo 33.8 OZ from Manufacturer Aveda, Model 18084994054.",
                "Design House Aveda balancing shampoo with pure plant and flower essences.",
                "Plant-based haircare balancing shampoo formulated under environmental sustainability values.",
            ],
            "30pct_paraphrased_feature_queries": [
                "Looking for a scalp-balancing hair wash formulated with botanical and plant-derived cleansers.",
                "Need a daily haircare cleanser featuring pure floral and plant extracts for healthy scalp balance.",
                "Recommend a gentle balancing shampoo made from plant-based ingredients and botanical essences.",
                "Searching for an eco-conscious scalp care shampoo infused with plant and flower oils.",
                "Which shampoo uses natural plant-based haircare ingredients to keep the scalp balanced?",
                "I want a botanical balancing hair cleanser crafted with pure flower and plant essences for scalp care.",
            ],
            "20pct_buyer_intent_queries": [
                "I want to buy shampoo with plant-based haircare ingredients and a balancing formula.",
                "I would like to purchase shampoo featuring plant-based haircare ingredients and plant and flower essences.",
                "Looking to buy shampoo with a balancing formula and scalp care.",
                "Please help me buy shampoo featuring plant-based haircare ingredients and plant and flower essences.",
            ],
        },
    },
    "09_toothbrush": {
        "target_public_id": "B09J81RMLD",
        "category": "toothbrush",
        "source_metadata": "resources/evaluation_base/public_metadata/product_metadata.json#B09J81RMLD",
        "breakdown": {
            "50pct_public_description_and_metadata": [
                "Baby U Shaped Toothbrush for Aged 2-12 with 360° Soft Silicone Brush Heads Handle Kids Cute Teeth Whitening Massage Tool",
                "U-Shaped Design 360 Cleansing Method with U-shape brush head achieving three-sided fitting of teeth and gums.",
                "Made of food grade silicone, safe material, no special smell, ergonomic shape and size for toddlers and kids.",
                "Multi-layer bristles design give teeth full protection and can easily clean tongue coating.",
                "Soft silicone brush head suitable for 2-12 toddlers with easy-to-grip handle for small hands.",
                "Unique U-shaped brush head design to fit teeth better and clean teeth from all angles.",
                "360 degree clean baby teeth with food grade silicone brush head resistant to high temperature up to 100 C.",
                "Easy-to-grip cute handle kids toothbrush with 360 degree cleaning and teeth whitening massage tool.",
                "Toddler manual toothbrush with U-shape silicone head, wash in warm water, applicable age 0-12 years old.",
                "Kids oral teeth cleaning tool made of silicone and PP with three-sided fitting for effective brushing.",
            ],
            "30pct_paraphrased_feature_queries": [
                "Looking for a toddler oral brush shaped like a U with gentle silicone bristles for full-mouth cleaning.",
                "Need a kid-friendly manual toothbrush that wraps around teeth 360 degrees using soft food-grade silicone.",
                "Recommend a children's U-shape teeth cleaner with an easy-grip small handle and soft silicone head.",
                "Searching for a safe silicone kids toothbrush offering 360-degree coverage and comfortable grip for little hands.",
                "Which children's toothbrush combines a U-shaped silicone head with an ergonomic handle for small hands?",
                "I want a gentle 360-degree cleaning toothbrush for toddlers featuring a U-shaped soft silicone brush head.",
            ],
            "20pct_buyer_intent_queries": [
                "I want to buy a toothbrush with a U-shaped design and soft silicone brush heads.",
                "I would like to purchase a toothbrush featuring a U-shaped design and 360-degree cleaning.",
                "Looking to buy a toothbrush with soft silicone brush heads and a handle designed for small hands.",
                "Please help me buy a toothbrush featuring a U-shaped design and 360-degree cleaning.",
            ],
        },
    },
}


def main() -> None:
    manifest = read_json(EVALUATION_BASE / "manifest.json")
    OUT_QUERIES.mkdir(parents=True, exist_ok=True)
    for group, spec in ORACLE_QUERIES.items():
        b = spec["breakdown"]
        part50 = b["50pct_public_description_and_metadata"]
        part30 = b["30pct_paraphrased_feature_queries"]
        part20 = b["20pct_buyer_intent_queries"]
        assert len(part50) == 10 and len(part30) == 6 and len(part20) == 4
        instructions = part50 + part30 + part20
        assert len(instructions) == 20
        assert len({" ".join(t.split()).casefold() for t in instructions}) == 20
        instructions_sha = hashlib.sha256(
            json.dumps(instructions, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        doc = {
            "schema_version": 2,
            "status": "ready",
            "query_mode": "text",
            "target_public_id": spec["target_public_id"],
            "instruction_policy": "visual_inception_training_oracle_50_30_20_v1",
            "category": spec["category"],
            "source_metadata": spec["source_metadata"],
            "sampling_ratios": {
                "public_description_and_metadata": "10/20 (50%)",
                "paraphrased_feature_queries": "6/20 (30%)",
                "buyer_intent_queries": "4/20 (20%)",
            },
            "paper_exact_log_source": False,
            "breakdown": b,
            "instructions": instructions,
            "instructions_sha256": instructions_sha,
            "query_image_required": False,
            "query_images_generated": False,
        }
        out_path = OUT_QUERIES / f"{group}_training_oracle.json"
        if not out_path.exists():
            freeze_json(out_path, doc)
        loaded = load_formal_queries(out_path, {"id": manifest["groups"][group]["target_id"]}, query_mode="text")
        assert len(loaded) == 20
        print(f"Verified {group}: {out_path} (20 text queries, sha256={instructions_sha[:12]})")


if __name__ == "__main__":
    main()
