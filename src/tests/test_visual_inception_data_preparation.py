"""改写服务接口只使用注入响应与显式假凭据，不联网、不读取真实密钥。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.visual._visual_inception import data_preparation as preparation


def descriptions():
    return [dict(id=f'public-{index:03d}', category='shampoo',
        text=f'Public synthetic shampoo description number {index:03d} with gentle formula.')
        for index in range(50)]


def model_response(*, model='MiniMax-M3', finish_reason='stop'):
    rows = [dict(input_row_id=f'public-{index:03d}',
        text=f'Synthetic shopper wording {index:03d} about shampoo with gentle formula.')
        for index in range(30)]
    return dict(id='explicit-fixture-response', model=model, choices=[dict(finish_reason=finish_reason,
        message=dict(content=json.dumps(rows), reasoning_content='fixture_reasoning_never_saved'))])


def test_default_preparation_only_plans_without_reading_environment_or_calling_network(monkeypatch):
    class EnvironmentThatMustNotBeRead(dict):
        def get(self, *args, **kwargs):
            pytest.fail('仅规划时不能读取凭据环境')

    monkeypatch.setattr(preparation.os, 'environ', EnvironmentThatMustNotBeRead())
    monkeypatch.setattr(preparation, '_post_json', lambda *args: pytest.fail('仅规划时不得发请求'))
    result = preparation.generate_paraphrases(descriptions(), model='MiniMax-M3')
    assert result['status'] == 'planned' and result['network_request_performed'] is False
    assert result['requested_model'] == 'MiniMax-M3'
    assert 'temperature' not in result and 'requested_temperature' not in result


def test_missing_key_stops_before_request_and_does_not_emit_secret_value(monkeypatch):
    environment_name = 'VISUAL_INCEPTION_EXPLICIT_TEST_KEY'
    monkeypatch.delenv(environment_name, raising=False)
    called = []
    with pytest.raises(ValueError, match='凭据|环境变量'):
        preparation.generate_paraphrases(descriptions(), model='MiniMax-M3',
            api_key_env=environment_name, generate=True,
            request_fn=lambda *args: called.append(args))
    assert called == []


def test_minimax_generation_omits_unsupported_temperature_and_records_actual_identity(monkeypatch):
    environment_name, synthetic_key = 'VISUAL_INCEPTION_EXPLICIT_TEST_KEY', 'explicit-test-fixture-not-real-key'
    monkeypatch.setenv(environment_name, synthetic_key)
    captured = {}

    def request(url, payload, key, timeout):
        captured.update(url=url, payload=payload, key=key, timeout=timeout)
        return model_response()

    result = preparation.generate_paraphrases(descriptions(), model='MiniMax-M3',
        api_key_env=environment_name, generate=True, request_fn=request)
    assert result['network_request_performed'] is True and len(result['rows']) == 30
    assert captured['key'] == synthetic_key
    assert captured['payload']['model'] == 'MiniMax-M3'
    assert 'temperature' not in captured['payload']
    sent = json.loads(captured['payload']['messages'][1]['content'])
    assert len(sent) == 30
    assert [row['input_row_id'] for row in sent] == [row['id'] for row in descriptions()[:30]]
    identity = result['generation']
    assert identity['provider'] == 'minimax' and identity['model'] == 'MiniMax-M3'
    assert identity['effective_temperature'] == 1.0
    assert identity['temperature_supported'] is False and identity['temperature_parameter_sent'] is False
    assert 'temperature' not in identity and 'requested_temperature' not in identity
    assert all(row['temperature_parameter_sent'] is False and 'temperature' not in row
               and 'requested_temperature' not in row for row in result['rows'])
    serialized = json.dumps(result)
    assert synthetic_key not in serialized and 'fixture_reasoning_never_saved' not in serialized


def test_supported_model_keeps_explicit_paper_temperature_in_request_and_evidence(monkeypatch):
    environment_name = 'VISUAL_INCEPTION_EXPLICIT_TEST_KEY'
    monkeypatch.setenv(environment_name, 'explicit-test-fixture-not-real-key')
    captured = {}

    def request(url, payload, key, timeout):
        captured.update(payload=payload)
        return model_response(model='gpt-4-0613')

    result = preparation.generate_paraphrases(descriptions(), provider='openai', model='gpt-4-0613',
        base_url='https://example.invalid/v1', api_key_env=environment_name,
        generate=True, request_fn=request)
    assert captured['payload']['temperature'] == .8
    identity = result['generation']
    assert identity['temperature'] == identity['effective_temperature'] == .8
    assert identity['temperature_supported'] is identity['temperature_parameter_sent'] is True


@pytest.mark.parametrize('change', ['truncated', 'wrong_provider', 'duplicate_text', 'invalid_input_reference'])
def test_incomplete_or_misattributed_responses_are_not_frozen(monkeypatch, change):
    environment_name = 'VISUAL_INCEPTION_EXPLICIT_TEST_KEY'
    monkeypatch.setenv(environment_name, 'explicit-test-fixture-not-real-key')
    response = model_response()
    if change == 'truncated':
        response['choices'][0]['finish_reason'] = 'length'
    elif change == 'wrong_provider':
        response['model'] = 'some-unrelated-model'
    else:
        rows = json.loads(response['choices'][0]['message']['content'])
        if change == 'duplicate_text':
            rows[-1]['text'] = rows[0]['text']
        else:
            rows[-1]['input_row_id'] = 'not-an-input-description'
        response['choices'][0]['message']['content'] = json.dumps(rows)
    with pytest.raises(ValueError):
        preparation.generate_paraphrases(descriptions(), model='MiniMax-M3',
            api_key_env=environment_name, generate=True, request_fn=lambda *args: response)


def test_command_line_plan_validates_descriptions_without_creating_output_or_calling_service(tmp_path, monkeypatch, capsys):
    from tools.prepare_visual_inception_data import main
    from src.visual._visual_inception import training_data
    source = tmp_path / 'public_descriptions.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in descriptions()), encoding='utf-8')
    monkeypatch.setattr(training_data, 'DATA', tmp_path / 'data')
    output = tmp_path / 'data' / 'public' / 'must_not_be_created_by_plan'
    monkeypatch.setattr(preparation, '_post_json', lambda *args: pytest.fail('默认命令不得调用外部服务'))
    assert main(['paraphrases', '--descriptions', str(source), '--model', 'MiniMax-M3',
                 '--output', str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'planned' and result['network_request_performed'] is False
    assert not output.exists()


@pytest.mark.parametrize('entry,relative', [
    ('source', 'resources/evaluation_base/evaluation_inputs/queries.json'),
    ('metadata', 'src/agent/config/llm.local.json'),
])
def test_forbidden_source_and_metadata_paths_are_rejected_before_open(monkeypatch, entry, relative):
    from src.paths import PROJECT
    from src.visual._visual_inception.training_data import read_source_rows
    from tools.prepare_visual_inception_data import _json
    forbidden = (PROJECT / relative).resolve()
    original_open = Path.open
    opened = []

    def tracked_open(path, *args, **kwargs):
        if path.resolve() == forbidden:
            opened.append(str(path))
            pytest.fail('私有输入必须在打开之前拒绝')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', tracked_open)
    with pytest.raises(ValueError):
        if entry == 'source':
            read_source_rows(forbidden)
        else:
            _json(forbidden)
    assert opened == []


def test_invalid_production_output_stops_before_reading_or_creating_anything(tmp_path, monkeypatch):
    invalid_output = tmp_path / 'outside_public_training_roots'
    monkeypatch.setattr(preparation, 'read_source_rows',
        lambda *args: pytest.fail('非法生产输出应在读取来源之前拒绝'))
    with pytest.raises(ValueError, match='公开训练目录|生产|正式'):
        preparation.prepare_training_bundle(source=tmp_path / 'unused_source.png',
            target=tmp_path / 'unused_target.png', public_semantic={},
            descriptions=tmp_path / 'unused_descriptions.jsonl',
            paraphrases=tmp_path / 'unused_paraphrases.jsonl',
            heldout_queries=tmp_path / 'unused_heldout.jsonl',
            candidates=tmp_path / 'unused_candidates.jsonl', source_metadata={},
            output_dir=invalid_output, test_fixture=False)
    assert not invalid_output.exists()

