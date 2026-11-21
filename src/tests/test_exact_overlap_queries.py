"""Check the published text suites without retrieval or model credentials."""
import json
import shutil

import pytest

from src.experiments import exact_overlap_queries as overlap
from src.experiments.visual_inception_category_evaluation import validate_category_query_suite


@pytest.mark.parametrize('suite, repeated, fallback_count',
                         [('training_oracle_v4', 20, 0), ('overlap_30_v4', 6, 3), ('zero_overlap_v4', 0, 4)])
def test_frozen_fourth_version_has_exact_overlap(suite, repeated, fallback_count):
    result = validate_category_query_suite(query_suite=suite)
    assert result['training_overlap_checked']
    assert result['group_count'] == 9
    assert result['total_queries'] == 180
    assert result['training_normalized_exact_overlap_count'] == repeated * 9
    assert all(group['training_normalized_exact_overlap_count'] == repeated
               for group in result['groups'].values())
    assert result['synthetic_search_count'] == fallback_count
    assert result['groups']['02_chair']['synthetic_search_count'] == fallback_count


def test_validator_rejects_a_new_text_even_when_the_file_digest_is_updated(tmp_path):
    root = tmp_path / overlap.ROOT.name
    shutil.copytree(overlap.ROOT, root)
    suite = root / 'overlap_30_v4'
    manifest_path = suite / 'manifest.json'
    manifest = overlap.read_json(manifest_path)
    group = next(iter(manifest['groups']))
    query_path = suite / group / 'queries.json'
    doc = overlap.read_json(query_path)
    doc['instructions'][0] = 'Synthetic fixture text that is unrelated to the bound query.'
    doc['query_records'][0]['instruction'] = doc['instructions'][0]
    doc['instructions_sha256'] = overlap.instructions_sha(doc['instructions'])
    query_path.write_text(json.dumps(doc), encoding='utf-8')
    manifest['groups'][group] = overlap.file_sha(query_path)
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='配对位置不一致'):
        overlap.validate(suite)


def test_genuine_search_shortfall_never_creates_an_eight_group_suite(tmp_path):
    frozen = overlap.read_json(overlap.ROOT / 'pairs.json')
    frozen['status'] = 'pending'
    frozen['groups'].pop('02_chair')
    frozen['missing_groups']['02_chair'] = {
        'counts': dict(overlap.FAMILIES, heldout_search_query=0),
        'missing': {'heldout_search_query': 4},
    }
    (tmp_path / 'prepared_pairs.json').write_text(json.dumps(frozen), encoding='utf-8')
    assert overlap.validate_preparation(tmp_path)['status'] == 'pending'
    with pytest.raises(ValueError, match='待补齐九组.*02_chair'):
        overlap.validate(tmp_path / 'zero_overlap_v4')


def test_duplicate_checks_catch_reordering_and_training_field_fragments():
    index = overlap.DuplicateIndex(['Deep blue cotton denim', 'A longer field contains unconsumed-looking text.'])
    assert 'Cotton denim, deep blue!' in index
    assert index.contains_fragment('contains unconsumed-looking text')
    assert 'Deep blue stretch denim' not in index


def test_acceptable_rejects_noise_heads_and_off_category_queries():
    from tools.build_visual_inception_eval_v4_suites import acceptable

    assert not acceptable('06_jeans', 'Find a light-weight soft breathable regular-fit casual top in a small Asian size.')
    assert not acceptable('06_jeans', '✤✤【SIZE CHART】✤✤Size:S/US:4Waist:58cm/22.83 Hip:84cm/33.07 Length:105cm')
    assert not acceptable('06_jeans', '100% Cotton Machine Wash. Comfortable breathable everyday wear for all seasons.')
    assert not acceptable('05_hair_extension', 'Suitable for daily use, work, school, dating, party, engagement, marriage, travel.')
    assert acceptable('06_jeans', 'High-waisted stretch denim jeans with a slim flared leg and classic five-pocket styling.')
    assert acceptable('05_hair_extension', 'Silky straight tape-in human hair extensions in 18 inch length for natural volume.')
