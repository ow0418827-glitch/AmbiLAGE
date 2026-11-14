"""用户确认的公开或私有同步范围；仅临时测试仓库，不调用网络。"""
import json
import gzip
import hashlib
from pathlib import Path
import subprocess
import pytest
from src.paths import PROJECT
from tools import preflight
from tools.check_github_sync import inspect_github_sync, read_sync_policy, credential_kinds, catalog_bundle_status


def _project(tmp_path, visibility='public'):
    (tmp_path / ".gitignore").write_text((PROJECT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8")
    policy = json.loads((PROJECT / "resources/github_sync_policy.json").read_text(encoding="utf-8"))
    policy.update(repository_visibility=visibility, private_repo_required=visibility == 'private',
                  visibility_confirmation='user_confirmed')
    files = {
        "output/imported_lucid_baseline_20261113/IMPORT_MANIFEST.json": '{}',
        "resources/model_lock.json": '{"schema_version":1,"models":{}}',
        "resources/github_sync_policy.json": json.dumps(policy),
        "src/module.py": "value = 1\n",
        "resources/evaluation_base/shared_memory/shared_memory.json": "{}",
        "resources/evaluation_base/shared_memory/images/shared.jpg": "fixture",
        "data/private/input.json": "{}",
        "models/editor/model.safetensors": "excluded model",
        "weights/encoder/model.pt": "excluded model",
        "output/formal_experiment/results.json": "{}",
        "output/formal_experiment/resume.pt": "resume state",
        "output/formal_experiment/optimizer.pth": "optimizer state",
        "output/formal_experiment/retrieval_index_cache/vector.json": "{}",
        "output/local_validation/pytest.log": "test report",
        "output/local_validation/work/test.json": "excluded temporary fixture",
        "output/package_check/copy.txt": "excluded copy",
        "src/config/llm.local.json": "excluded other local config",
        "src/agent/config/llm.local.json": json.dumps({"model": {"api_key": "sk-" + "disposableauthorized123456789"}}),
        "src/agent/config/llm.example.json": json.dumps({'model':{'provider':'minimax','model':'fixture',
            'base_url':'https://example.invalid/v1','api_key':'','api_key_env':'MINIMAX_API_KEY'}}),
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    bundle=tmp_path/'resources/webshop/catalog_bundle'
    parts=[]
    for index,name in enumerate(['catalog.sqlite','data_manifest.json','index_manifest.json','goals.json','index/segments_1']):
        relative=f'chunks/f{index:04d}.00000.part'
        path=bundle/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        raw=('explicit tiny fixture: '+name).encode('utf-8')
        packed=gzip.compress(raw,mtime=0)
        path.write_bytes(packed)
        parts.append(dict(path=name,codec='gzip',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            compressed_bytes=len(packed),chunks=[dict(path=relative,bytes=len(packed),sha256=hashlib.sha256(packed).hexdigest())]))
    (bundle/'manifest.json').write_text(json.dumps(dict(schema_version=1,codec='gzip',completed_at='fixture',
        search_backend='lucene',raw_bytes=sum(p['bytes'] for p in parts),
        stored_bytes=sum(p['compressed_bytes'] for p in parts),files=parts)),encoding='utf-8')
    return tmp_path


@pytest.mark.parametrize('visibility', ['public', 'private'])
def test_confirmed_sync_includes_inputs_results_and_resume_but_excludes_key(tmp_path, visibility):
    root = _project(tmp_path, visibility)
    result = inspect_github_sync(root)
    assert result["passed"] is True
    assert result['repository_visibility'] == visibility
    assert result['private_repo_required'] is (visibility == 'private')
    required = {
        "resources/evaluation_base/shared_memory/shared_memory.json",
        "resources/evaluation_base/shared_memory/images/shared.jpg",
        "data/private/input.json", "src/agent/config/llm.example.json",
        "output/formal_experiment/results.json", "output/formal_experiment/resume.pt",
        "output/formal_experiment/optimizer.pth",
        "output/formal_experiment/retrieval_index_cache/vector.json",
        "output/local_validation/pytest.log",
    }
    assert required.issubset(result["candidate_paths"])
    assert "models/editor/model.safetensors" not in result["candidate_paths"]
    assert "output/local_validation/work/test.json" not in result["candidate_paths"]
    assert "output/package_check/copy.txt" not in result["candidate_paths"]
    assert all(p["ignored"] == p["expected_ignored"] for p in result["ignore_probes"])
    assert "src/agent/config/llm.local.json" not in result['candidate_paths']
    assert result["authorized_credential_candidates"] == []
    assert result['user_authorized_credentials'] == []
    assert result['known_local_credentials_checked'] is True
    assert not result["credential_candidates"] and not result["missing_required_candidates"]
    assert not (root / ".git").exists()
    assert result["staged_committed_or_pushed"] is False
    assert result["remote_visibility_verified"] is False
    assert result['catalog_bundle']['ready'] is True
    assert result['catalog_bundle']['chunk_hashes_checked'] is False


def test_git_inspection_detects_force_tracked_excluded_weight(tmp_path):
    root = _project(tmp_path)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    relative = "models/editor/model.safetensors"
    subprocess.run(["git", "-C", str(root), "add", "--force", relative], check=True)
    result = inspect_github_sync(root)
    assert result["passed"] is False and relative in result["forbidden_candidates"]
    assert result["repository_mode"] == "existing_project_repository_read_only"


def test_unexpected_credential_is_reported_without_value(tmp_path):
    root = _project(tmp_path)
    token = "sk-" + "disposableunexpected123456789"
    (root / "src/unexpected.json").write_text(json.dumps({"api_key": token}), encoding="utf-8")
    result = inspect_github_sync(root)
    assert result["passed"] is False
    assert {"path": "src/unexpected.json", "kind": "api_token_shape"} in result["credential_candidates"]
    assert token not in json.dumps(result)


def test_force_tracked_local_key_is_rejected_without_value(tmp_path):
    root = _project(tmp_path)
    subprocess.run(['git','init','--quiet',str(root)],check=True)
    subprocess.run(['git','-C',str(root),'add','--force','src/agent/config/llm.local.json'],check=True)
    result = inspect_github_sync(root)
    assert result["passed"] is False
    assert 'src/agent/config/llm.local.json' in result['forbidden_candidates']
    assert 'disposableauthorized123456789' not in json.dumps(result)


def test_backup_does_not_require_real_key_config(tmp_path):
    root = _project(tmp_path)
    (root / "src/agent/config/llm.local.json").unlink()
    result = inspect_github_sync(root)
    assert result["passed"] is True
    assert result['known_local_credentials_checked'] is False


def test_exact_local_key_copied_to_result_log_is_detected(tmp_path):
    root = _project(tmp_path)
    token='provider-'+'opaquecredentialvalue0123456789'
    path=root/'src/agent/config/llm.local.json'
    path.write_text(json.dumps({'model':{'api_key':' '+token+' '}}),encoding='utf-8')
    log=root/'output/formal_experiment/provider.log'
    log.write_text('failed response: '+token,encoding='utf-8')
    result=inspect_github_sync(root)
    assert result['passed'] is False
    assert {'path':'output/formal_experiment/provider.log','kind':'known_local_credential'} in result['credential_candidates']
    assert token not in json.dumps(result)


def test_credentials_crossing_stream_chunk_boundary_are_found(tmp_path):
    token='sk-'+'streamcredential0123456789'
    path=tmp_path/'response.jsonl'
    path.write_text('x'*30+token,encoding='utf-8')
    assert 'api_token_shape' in credential_kinds(path,chunk_bytes=32)


def test_credential_exceptions_cannot_be_reenabled_in_policy(tmp_path):
    root=_project(tmp_path)
    path=root/'resources/github_sync_policy.json'
    policy=json.loads(path.read_text(encoding='utf-8'))
    policy['user_authorized_credentials']=[{'path':'src/agent/config/llm.local.json'}]
    path.write_text(json.dumps(policy),encoding='utf-8')
    with pytest.raises(ValueError,match='no credential exceptions'):
        read_sync_policy(root)


def test_partial_bundle_is_not_reported_ready(tmp_path):
    root=_project(tmp_path)
    path=root/'resources/webshop/catalog_bundle/manifest.partial.json'
    path.write_text('{}',encoding='utf-8')
    assert catalog_bundle_status(root)['status']=='packing_incomplete'
    result=inspect_github_sync(root)
    assert result['passed'] is False
    assert str(path.relative_to(root)).replace('\\','/') not in result['candidate_paths']


def test_orphan_bundle_part_is_not_reported_ready(tmp_path):
    root=_project(tmp_path)
    (root/'resources/webshop/catalog_bundle/chunks/orphan.part').write_bytes(b'x')
    result=catalog_bundle_status(root)
    assert result['ready'] is False
    assert result['status']=='unregistered_or_missing_parts'


def test_missing_or_oversized_bundle_part_is_not_reported_ready(tmp_path):
    root=_project(tmp_path)
    part=root/'resources/webshop/catalog_bundle/chunks/f0000.00000.part'
    part.write_bytes(b'changed-size')
    assert catalog_bundle_status(root)['ready'] is False


@pytest.mark.parametrize('visibility', ['unknown', '', None])
def test_sync_policy_rejects_unknown_visibility(tmp_path, visibility):
    root = _project(tmp_path)
    path = root / "resources/github_sync_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["repository_visibility"] = visibility
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly user_confirmed"):
        read_sync_policy(root)


@pytest.mark.parametrize('confirmation', ['assumed', '', None])
def test_sync_policy_requires_user_visibility_confirmation(tmp_path, confirmation):
    root = _project(tmp_path)
    path = root / 'resources/github_sync_policy.json'
    policy = json.loads(path.read_text(encoding='utf-8'))
    policy['visibility_confirmation'] = confirmation
    path.write_text(json.dumps(policy), encoding='utf-8')
    with pytest.raises(ValueError, match='explicitly user_confirmed'):
        read_sync_policy(root)


def test_sync_policy_rejects_inconsistent_private_requirement(tmp_path):
    root = _project(tmp_path)
    path = root / 'resources/github_sync_policy.json'
    policy = json.loads(path.read_text(encoding='utf-8'))
    policy['private_repo_required'] = True
    path.write_text(json.dumps(policy), encoding='utf-8')
    with pytest.raises(ValueError, match='must agree'):
        read_sync_policy(root)


def test_code_only_preflight_skips_resource_checker(tmp_path, monkeypatch, capsys):
    def forbid(*args, **kwargs):
        raise AssertionError("Resource checker must not be called in code-only mode")
    monkeypatch.setattr(preflight, "check_resources", forbid)
    output = tmp_path / "report.json"
    assert preflight.main(["--code-only", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["resources"] == {"status": "skipped_code_only", "private_resources_read": False}
    assert report["code"]["python_files_parsed"] > 0
    assert report["remote_model_execution_verified"] is False
    capsys.readouterr()


def test_complete_preflight_still_requires_resource_validation(tmp_path, monkeypatch):
    def missing():
        raise FileNotFoundError("Shared pool has not been synchronized")
    monkeypatch.setattr(preflight, "check_resources", missing)
    with pytest.raises(FileNotFoundError, match="not been synchronized"):
        preflight.main(["--output", str(tmp_path / "report.json")])
