"""轻量公开选图：公开索引→粗类型候选→灰度门槛取中位数/黄金先验带→仅选中图对标注→发布。"""
import argparse
from collections import Counter
from dataclasses import asdict
import inspect
import math
import os
import re
from types import SimpleNamespace

import numpy as np

from src.paths import PUBLIC, PRIVATE, VISUAL_CONFIGS, EXPERIMENT_CONFIGS, public_root_for, ProjectPath as Path
from src.paths import DATA,OUTPUT,CONFIGS,PRIVATE,SHARED_MEMORY,PROJECT
from src.visual.audit import digest,freeze_json,read_json,save_json
from src.visual.foreground_shape_features import create_shape_backend,VALID_SHAPE_POLICIES
from src.visual.image_utils import sha256_file
from src.visual.pair_selection import (
    GOLDEN_BAND_MEDIAN_POLICY,
    MEDIAN_SELECTION_POLICY,
    SELECTION_POLICY,
    VALID_SELECTION_STRATEGIES,
    SelectionConfig,
    LocalSelectionBackend,
    choose_pair,
    unit_rows,
    validate_product,
    validate_public_rank_band,
    verify_selection,
    validate_subtype_filter,
)
from src.visual.shape_calibration import calibrate,load_calibration
from src.visual.catalog_training_guard import catalog_training_guard
from src.visual.selection.webshop_public import (
    MAX_CANDIDATE_LIMIT,
    OWNERSHIP_POLICY,
    download_shortlist,
    extract_public_candidates,
    extract_public_rank_peers,
    random_shortlist,
)
from src.visual.pair_assets import publish_pair


def _public_root(value):
    path=Path(value).resolve()
    try:
        return public_root_for(path, extra_roots=(PUBLIC,))
    except ValueError:
        out=Path(OUTPUT).resolve()
        if path.is_relative_to(out):
            parts=path.relative_to(out).parts
            if 'data' in parts:
                return Path(out / Path(*parts[:parts.index('data') + 1]))
        raise


def public_path(value):
    path=Path(value);path=(path if path.is_absolute() else PROJECT/path).resolve()
    _public_root(path)
    return path


def validate_config(config):
    required={'schema_version','seed','candidate_limit','calibration_path','groups'}
    optional={
        'models','type_reviews','subtype_filters','strategy',
        'expand_candidate_limits','candidate_limits','preprocessing','public_rank_band',
        'excluded_source_ids',
    }
    if not isinstance(config,dict) or not required<=set(config) or set(config)-required-optional or config['schema_version']!=2:
        raise ValueError('轻量选图须使用第二版配置；旧多指标配置只保留历史读取')
    if type(config['seed']) is not int or type(config['candidate_limit']) is not int or not 1<=config['candidate_limit']<=MAX_CANDIDATE_LIMIT:
        raise ValueError(f'种子须为整数，公开图片候选预算须在一至 {MAX_CANDIDATE_LIMIT} 之间')
    if not isinstance(config['calibration_path'],str) or not config['calibration_path'].strip():raise ValueError('须配置公开开发集校准证书路径')
    if not config['groups']:raise ValueError('至少指定一个公开目标')
    for group in config['groups']:
        if (set(group)!={'key','target_id'} or not re.fullmatch(r'[A-Za-z0-9_]+',group['key'])
                or not re.fullmatch(r'[A-Za-z0-9]{10}',group['target_id'])):raise ValueError('组名或固定公开目标编号无效')
    if len({g['key'] for g in config['groups']})!=len(config['groups']) or len({g['target_id'] for g in config['groups']})!=len(config['groups']):
        raise ValueError('组名及公开目标编号须唯一')
    if set(config.get('models',{}))-{'dino_model','batch_size'}:raise ValueError('新选图阶段只使用单一冻结视觉模型')
    strategy=config.get('strategy','median')
    if strategy not in VALID_SELECTION_STRATEGIES:raise ValueError('选图策略只支持 median 或 uniform')
    if config.get('preprocessing') is not None and config['preprocessing'] not in VALID_SHAPE_POLICIES:
        raise ValueError('不受支持的选图外形预处理策略：'+str(config['preprocessing']))
    if 'public_rank_band' in config:
        validate_public_rank_band(config['public_rank_band'])
    if 'excluded_source_ids' in config:
        ex_ids=config['excluded_source_ids']
        if (not isinstance(ex_ids,list)
                or any(not isinstance(pid,str) or not re.fullmatch(r'[A-Za-z0-9]{10}',pid) for pid in ex_ids)
                or len(set(ex_ids))!=len(ex_ids)):
            raise ValueError('显式排除的原商品编号列表须为互异的十位字母数字编号')
    keys={g['key'] for g in config['groups']}
    if 'candidate_limits' in config:
        limits=config['candidate_limits']
        if not isinstance(limits,dict) or set(limits)-keys:raise ValueError('按组候选预算只支持配置中的公开组名')
        for value in limits.values():
            if type(value) is not int or not 1<=value<=MAX_CANDIDATE_LIMIT:
                raise ValueError(f'按组候选预算须为一至 {MAX_CANDIDATE_LIMIT} 之间的整数')
    if 'expand_candidate_limits' in config:
        ladders=config['expand_candidate_limits']
        if not isinstance(ladders,list) or not ladders or any(type(v) is not int or not config['candidate_limit']<=v<=MAX_CANDIDATE_LIMIT for v in ladders):
            raise ValueError(f'扩展候选预算阶梯须为不低于基础预算且不超过 {MAX_CANDIDATE_LIMIT} 的整数列表')
        if any(b<=a for a,b in zip(ladders,ladders[1:])):raise ValueError('扩展候选预算阶梯须严格递增')
    if 'subtype_filters' in config:
        filters=config['subtype_filters']
        if not isinstance(filters,dict) or set(filters)-keys:raise ValueError('细分品类过滤只支持配置中的公开组名')
        for rule in filters.values():
            if rule is None:raise ValueError('细分品类过滤规则不能为空')
            validate_subtype_filter(rule,allow_empty=False)
    return config


def selection_settings(config):
    return SelectionConfig(
        seed=config['seed'],
        calibration_path=str(public_path(config['calibration_path'])),
        strategy=config.get('strategy','median'),
    )


def freeze_plan(config,root):
    from src.visual.product_types import TYPE_RULES_SHA256
    path=public_path(config['calibration_path']);proof=read_json(path)
    load_calibration(path,proof['models'],public_root=(_public_root(path), PUBLIC))
    impl_files=['src/visual/pair_selection.py','src/visual/shape_calibration.py',
        'src/visual/selection/webshop_public.py','src/experiments/public_pair_selection.py']
    if config.get('preprocessing'):
        impl_files.append('src/visual/foreground_shape_features.py')
    rank_band=config.get('public_rank_band') or {}
    empty_policy=('rank_band_shape_median_fallback_with_explicit_audit'
                  if rank_band.get('allow_shape_threshold_fallback') else 'stop_without_relaxation')
    freeze_json(Path(root)/'PUBLIC_PLAN.json',dict(schema_version=2,config=config,
        selection=asdict(selection_settings(config)),calibration_sha256=sha256_file(path),
        type_rules_sha256=TYPE_RULES_SHA256,
        implementation_sha256={name:sha256_file(PROJECT/name) for name in impl_files},
        ownership_policy=OWNERSHIP_POLICY,target_ids_fixed=[g['target_id'] for g in config['groups']],
        formal_queries_used=False,victim_memory_used=False,shopping_allowed=False,
        annotation_scope='drawn_source_and_target_only',empty_set_policy=empty_policy))


def pipeline_config(key,catalog,root,*,selection=None,models=None):
    root=Path(root)
    return dict(schema_version=1,method='ambilage',public_catalog=str(Path(catalog).resolve()),
        run_dir=str((root/'groups'/key).resolve()),selection=selection or {},selection_models=models or {},
        generation=dict(lucid=dict(steps=1000,epsilon_levels=8),ambilage={}),
        evaluation=dict(shared_pool=str(SHARED_MEMORY),query_manifest=str(PRIVATE/'query_images'/root.name/(key+'.json'))))


def downloaded_product(product,downloads,*,source=False):
    record=dict(id=product['id'],title=product['title'],source_url=product['source_url'],image=downloads['records'][product['id']]['image'])
    if 'type_review' in product:record['type_review']=product['type_review']
    if source:record.update(owned=True,source_note='固定散列规则预先登记的实验模拟可控商品；不声明真实商家所有权')
    return validate_product(record,Path(record['image']).parent,owned=source,generation_ready=False,
                            public_root=_public_root(record['image']))


def _instantiate_backend(backend_factory,models_cfg,cache_dir,device,preprocessing=None):
    if backend_factory is create_shape_backend:
        return create_shape_backend(models_cfg,cache_dir,device=device,preprocessing=preprocessing)
    sig=inspect.signature(backend_factory)
    if 'preprocessing' in sig.parameters or any(p.kind==inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return backend_factory(models_cfg,cache_dir,device=device,preprocessing=preprocessing)
    return backend_factory(models_cfg,cache_dir,device)


_RANK_TITLE_STOPWORDS=frozenset({'the','and','for','with','of','in','to','by','on','at','from'})


def _lexical_cosine(title_a,title_b):
    tok_a={t for t in re.findall(r'[a-z0-9]{2,}',str(title_a or '').lower()) if t not in _RANK_TITLE_STOPWORDS}
    tok_b={t for t in re.findall(r'[a-z0-9]{2,}',str(title_b or '').lower()) if t not in _RANK_TITLE_STOPWORDS}
    if not tok_a or not tok_b:
        return 0.0
    return float(len(tok_a&tok_b)/math.sqrt(len(tok_a)*len(tok_b)))


def _resolve_local_clip_weight(model_key):
    from src.model_registry import resolve_model
    root=Path(resolve_model(model_key))
    for name in ('open_clip_model.safetensors','open_clip_pytorch_model.safetensors','model.safetensors','pytorch_model.bin'):
        cand=root/name
        if cand.is_file():
            return str(cand)
    for ext in ('*.safetensors','*.bin'):
        matches=sorted(root.rglob(ext))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f'公开先验编码器缺少本地权重文件：{root}')


def _close_prior_scorer_cache(scorer_cache):
    if not scorer_cache:
        return
    scorer_cache.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _prior_item_cache_key(scorer,item):
    img_sha=item.get('image_sha256') or sha256_file(item['image'])
    return digest({'scorer':scorer,'image_sha256':img_sha,'title':str(item.get('title') or '')})


def compute_public_prior_ranks(target,candidates,peers,backend,*,scorer='dino_lexical_hybrid',device='cpu',scorer_cache=None,cache_dir=None,batch_size=32):
    """基于零泄漏公开对照池估算每个原商品候选在公开先验下的基准名次（1 为最似目标）。"""
    validate_public_rank_band({'scorer':scorer})
    candidates=list(candidates);peers=list(peers)
    if not candidates or not peers:
        return {}
    all_items=[target]+candidates+peers
    scores={}
    if scorer=='dino_lexical_hybrid':
        if hasattr(backend,'features_many'):
            backend.features_many(all_items)
        t_vec=unit_rows(backend.features(target)['shape'])
        for p in candidates+peers:
            p_vec=unit_rows(backend.features(p)['shape'])
            vis=float(np.clip(p_vec@t_vec,-1.0,1.0))
            lex=_lexical_cosine(p['title'],target['title'])
            scores[p['id']]=0.55*vis+0.45*lex
    else:
        import torch
        import torch.nn.functional as F
        from src.visual.encoders.backbone import OpenCLIPSurrogate
        from src.visual.encoders.joint import ChunkedClipEncoder
        from src.visual.encoders.spec import BackboneConfig
        from src.visual.image_utils import read_tensor
        owns_cache=scorer_cache is None
        cache_store={} if owns_cache else scorer_cache
        cache_key=(scorer,str(device))
        disk_dir=Path(cache_dir).resolve() if cache_dir is not None else None
        if disk_dir is not None:
            disk_dir.mkdir(parents=True,exist_ok=True)
        try:
            entry=cache_store.get(cache_key)
            if entry is None:
                clip_name={'clip_b32':'ViT-B-32','clip_b16':'ViT-B-16','clip_l14':'ViT-L-14'}[scorer]
                spec=BackboneConfig(name=clip_name,pretrained='openai',weight_path=_resolve_local_clip_weight(scorer))
                surrogate=OpenCLIPSurrogate(spec,device=device,precision='fp32',allow_downloads=False,
                    allow_device_fallback=True,require_true_local_tokens=False)
                text_encoder=ChunkedClipEncoder(SimpleNamespace(surrogate=surrogate,device=surrogate.device,snapshot=surrogate.snapshot()))
                entry={'surrogate':surrogate,'text_encoder':text_encoder,'vectors':{},'text_vectors':{}}
                cache_store[cache_key]=entry
            surrogate=entry['surrogate']
            text_encoder=entry['text_encoder']
            vec_cache=entry['vectors']
            txt_cache=entry['text_vectors']
            vectors_by_id={}
            missing=[]
            seen_missing_keys=set()
            for item in all_items:
                ikey=_prior_item_cache_key(scorer,item)
                vec=vec_cache.get(ikey)
                if vec is None and disk_dir is not None:
                    fpath=disk_dir/f'{ikey}.npy'
                    if fpath.is_file():
                        with fpath.open('rb') as f:
                            loaded=np.asarray(np.load(f,allow_pickle=False),dtype=np.float64)
                        if loaded.ndim==1 and np.isfinite(loaded).all():
                            vec=unit_rows(loaded)
                            vec_cache[ikey]=vec
                if vec is not None:
                    vectors_by_id[item['id']]=vec
                elif ikey not in seen_missing_keys:
                    seen_missing_keys.add(ikey)
                    missing.append((ikey,item))
            if missing:
                step=max(1,int(batch_size))
                target_hw=(surrogate.image_size,surrogate.image_size)
                with torch.inference_mode():
                    for start in range(0,len(missing),step):
                        chunk=missing[start:start+step]
                        tensors=[]
                        for _,item in chunk:
                            img_t=read_tensor(item['image'],surrogate.device)
                            if img_t.shape[-2:]!=target_hw:
                                img_t=F.interpolate(img_t,size=target_hw,mode='bicubic',align_corners=False,antialias=True)
                            tensors.append(img_t)
                        batch_t=torch.cat(tensors,dim=0)
                        img_embs=surrogate.encode_image(batch_t).global_embedding.detach().cpu().numpy().astype(np.float64)
                        for (ikey,item),img_emb in zip(chunk,img_embs):
                            title_str=str(item.get('title') or '')
                            txt_emb=txt_cache.get(title_str)
                            if txt_emb is None:
                                txt_emb=np.asarray(text_encoder.encode_text(title_str),dtype=np.float64)
                                txt_cache[title_str]=txt_emb
                            vec=unit_rows(0.5*unit_rows(img_emb)+0.5*unit_rows(txt_emb))
                            vec_cache[ikey]=vec
                            if disk_dir is not None:
                                tmp_p=disk_dir/f'{ikey}.tmp.{os.getpid()}.npy'
                                with tmp_p.open('wb') as f:
                                    np.save(f,vec,allow_pickle=False)
                                tmp_p.replace(disk_dir/f'{ikey}.npy')
                for item in all_items:
                    if item['id'] not in vectors_by_id:
                        vectors_by_id[item['id']]=vec_cache[_prior_item_cache_key(scorer,item)]
            t_vec=vectors_by_id[target['id']]
            for p in candidates+peers:
                scores[p['id']]=float(np.clip(vectors_by_id[p['id']]@t_vec,-1.0,1.0))
        finally:
            if owns_cache:
                _close_prior_scorer_cache(cache_store)
    peer_keys=[(-scores[p['id']],p['id']) for p in peers]
    return {c['id']:1+sum(1 for pk in peer_keys if pk<(-scores[c['id']],c['id'])) for c in candidates}


def _prepare_and_download_rank_peers(products_path,groups,extracted,peer_dir,asset_output,*,seed,peer_count,index_path,excluded_ids,type_reviews,verify_source,subtype_filters):
    candidate_ids_by_group={g['key']:[c['id'] for c in g['candidates']] for g in extracted['groups']}
    peers_doc=extract_public_rank_peers(
        products_path,groups,peer_dir,seed=seed,peer_count=peer_count,index_path=index_path,
        excluded_ids=excluded_ids,candidate_ids_by_group=candidate_ids_by_group,
        type_reviews=type_reviews,verify_source=verify_source,subtype_filters=subtype_filters,
    )
    shortlist=dict(schema_version=2,groups=[dict(key=g['key'],target=g['target'],candidates=g['peers']) for g in peers_doc['groups']])
    downloads=download_shortlist(shortlist,peer_dir,asset_output=asset_output)
    return peers_doc,downloads


def _load_group_rank_peers(peers_dir,group_key):
    peers_path=Path(peers_dir)/'public_rank_peers.json'
    dl_path=Path(peers_dir)/'download_manifest.json'
    if not peers_path.is_file() or not dl_path.is_file():
        raise ValueError('已启用公开先验名次带但缺少公开对照池或下载清单，请先运行 prepare 阶段')
    peers_doc=read_json(peers_path);downloads=read_json(dl_path)
    group_row=next((g for g in peers_doc['groups'] if g['key']==group_key),None)
    if group_row is None:
        raise ValueError('公开对照池缺少分组：'+group_key)
    return [downloaded_product(p,downloads,source=False) for p in group_row['peers'] if p['id'] in downloads['records']]


def select_groups(config,root,public,*,device='cuda',backend_factory=create_shape_backend):
    root,public=Path(root),Path(public)
    short=read_json(public/'candidate_shortlist.json');downloads=read_json(public/'download_manifest.json')
    if downloads['shortlist_sha256']!=digest(short):raise ValueError('图片下载与公开候选集合不一致')
    settings=selection_settings(config)
    rank_band=validate_public_rank_band(config.get('public_rank_band'))
    backend=_instantiate_backend(backend_factory,config.get('models',{}),public/'selection_features',device,preprocessing=config.get('preprocessing'))
    prior_cache={};prior_cache_dir=(public/'prior_rank_features'/rank_band['scorer']) if (rank_band and rank_band['enabled']) else None
    rows=[];used=set(config.get('excluded_source_ids') or ());targets={g['target_id'] for g in config['groups']}
    try:
        for group in short['groups']:
            target=downloaded_product(group['target'],downloads)
            candidates=[downloaded_product(p,downloads,source=True) for p in group['candidates'] if p['id'] in downloads['records']]
            catalog=public/'candidate_catalogs'/(group['key']+'.json')
            freeze_json(catalog,dict(schema_version=2,target=target,candidates=candidates))
            if not candidates:raise ValueError('本组所有公开候选下载失败：'+group['key'])
            prior_ranks=None
            if rank_band and rank_band['enabled']:
                peers=_load_group_rank_peers(public/'rank_peers',group['key'])
                prior_ranks=compute_public_prior_ranks(target,candidates,peers,backend,
                    scorer=rank_band['scorer'],device=device,scorer_cache=prior_cache,cache_dir=prior_cache_dir)
            result=choose_pair(target,candidates,backend,settings,excluded_source_ids=used|targets,
                               public_root=_public_root(public),
                               calibration_public_root=(_public_root(settings.calibration_path), PUBLIC),
                               subtype_filter=config.get('subtype_filters',{}).get(group['key']),
                               public_rank_band=rank_band,public_prior_ranks=prior_ranks)
            result.update(target=target,binding=dict(public_catalog_sha256=sha256_file(catalog),models_configuration=config.get('models',{}),
                calibration_sha256=sha256_file(settings.calibration_path),public_plan_sha256=sha256_file(root/'PUBLIC_PLAN.json')),
                download_failures={p['id']:downloads['failures'][p['id']] for p in group['candidates'] if p['id'] in downloads['failures']})
            if result['selected_source_id']:
                result['source']=next(p for p in candidates if p['id']==result['selected_source_id']);used.add(result['selected_source_id'])
            path=root/'groups'/group['key']/'selection/DRAWN_PAIR.json';freeze_json(path,result)
            rows.append(dict(key=group['key'],status=result['status'],selection_path=str(path.resolve()),selection_sha256=sha256_file(path),
                target=target,source=result.get('source'),candidate_count=len(candidates),
                eligible_count=sum(r['eligible'] for r in result['candidates']),
                fallback_used=bool(result.get('shape_threshold_fallback_used',False)),
                passes_calibrated_threshold=bool(result.get('passes_calibrated_threshold',True))))
    finally:
        backend.close()
        _close_prior_scorer_cache(prior_cache)
    if rank_band and rank_band['enabled'] and settings.strategy=='median':
        scope='golden_band_median_shape_v1'
    elif settings.strategy=='median':
        scope='median_shape_after_public_shape_and_type_gates_v3'
    else:
        scope='uniform_draw_after_public_shape_and_type_gates_v2'
    bundle=dict(schema_version=2,status='selected' if all(r['status']=='selected' for r in rows) else 'no_eligible_pair',
        selection_scope=scope,strategy=settings.strategy,groups=rows,
        public_plan_sha256=sha256_file(root/'PUBLIC_PLAN.json'),shared_memory_read=False,formal_queries_read=False,generation_runs=0)
    freeze_json(root/'PAIR_SELECTION_BUNDLE.json',bundle);return bundle


def resolve_candidate_limits(config,products_path,index_path,root,public,*,device='cuda',verify_source=False,backend_factory=create_shape_backend):
    """按严格递增预算阶梯扩展合格数为零的组；不放宽外形门槛（仅允许在最终预算且配置开启时启用带内中位数回退），并保留完整尝试记录。"""
    root,public=Path(root),Path(public)
    base_limits=dict(config.get('candidate_limits',{}))
    ladders=list(config.get('expand_candidate_limits') or [])
    if not ladders:
        return dict(config),None
    proof=read_json(public_path(config['calibration_path']))
    reviews=read_json(public_path(config['type_reviews'])) if config.get('type_reviews') else {}
    settings=selection_settings(config)
    rank_band=validate_public_rank_band(config.get('public_rank_band'))
    backend=_instantiate_backend(backend_factory,config.get('models',{}),public/'selection_features',device,preprocessing=config.get('preprocessing'))
    prior_cache={};prior_cache_dir=(public/'prior_rank_features'/rank_band['scorer']) if (rank_band and rank_band['enabled']) else None
    config_excluded=set(config.get('excluded_source_ids') or ())
    extract_excluded=list(proof['development_product_ids'])+sorted(config_excluded)
    resolved={};used=set(config_excluded);targets={g['target_id'] for g in config['groups']};history={}
    try:
        for group in config['groups']:
            key=group['key']
            start=base_limits.get(key,config['candidate_limit'])
            steps=[start]+[v for v in ladders if v>start]
            attempts=[];chosen_limit=steps[-1]
            sub_dict={key:config['subtype_filters'][key]} if key in config.get('subtype_filters',{}) else None
            for limit in steps:
                trial_dir=root/'expansion_rounds'/key/f'limit_{limit}'
                extracted=extract_public_candidates(
                    products_path,[group],trial_dir,seed=config['seed'],lexical_limit=limit,
                    index_path=index_path,excluded_ids=extract_excluded,
                    type_reviews=reviews,verify_source=verify_source,
                    subtype_filters=sub_dict,
                )
                short=random_shortlist(extracted,trial_dir)
                downloads=download_shortlist(short,trial_dir,asset_output=public)
                row_group=short['groups'][0]
                target=downloaded_product(row_group['target'],downloads)
                candidates=[downloaded_product(p,downloads,source=True) for p in row_group['candidates'] if p['id'] in downloads['records']]
                reasons=Counter()
                eligible_count=0;status='no_downloads';selected_id=None;fallback_used=False
                if candidates:
                    prior_ranks=None
                    if rank_band and rank_band['enabled']:
                        peers_doc,peer_dl=_prepare_and_download_rank_peers(
                            products_path,[group],extracted,trial_dir/'rank_peers',public,
                            seed=config['seed'],peer_count=rank_band['peer_count'],index_path=index_path,
                            excluded_ids=extract_excluded,type_reviews=reviews,
                            verify_source=verify_source,subtype_filters=sub_dict,
                        )
                        peers=[downloaded_product(p,peer_dl,source=False) for p in peers_doc['groups'][0]['peers'] if p['id'] in peer_dl['records']]
                        prior_ranks=compute_public_prior_ranks(target,candidates,peers,backend,
                            scorer=rank_band['scorer'],device=device,scorer_cache=prior_cache,cache_dir=prior_cache_dir)
                    is_final_step=(limit==steps[-1])
                    result=choose_pair(
                        target,candidates,backend,settings,excluded_source_ids=used|targets,
                        public_root=_public_root(public),
                        calibration_public_root=(_public_root(settings.calibration_path),PUBLIC),
                        subtype_filter=config.get('subtype_filters',{}).get(key),
                        public_rank_band=rank_band,public_prior_ranks=prior_ranks,
                        allow_shape_threshold_fallback=(None if is_final_step else False),
                    )
                    status=result['status']
                    selected_id=result.get('selected_source_id')
                    fallback_used=bool(result.get('shape_threshold_fallback_used',False))
                    for c in result['candidates']:
                        if c['eligible']:eligible_count+=1
                        else:reasons.update(c['rejected_reasons'])
                attempts.append(dict(candidate_limit=limit,extracted_candidates=len(row_group['candidates']),
                    downloaded_candidates=len(candidates),eligible_count=eligible_count,status=status,
                    selected_source_id=selected_id,fallback_used=fallback_used,
                    rejection_counts=dict(sorted(reasons.items()))))
                chosen_limit=limit
                if status=='selected':
                    used.add(selected_id)
                    break
            resolved[key]=chosen_limit;history[key]=attempts
    finally:
        backend.close()
        _close_prior_scorer_cache(prior_cache)
    audit_log=dict(schema_version=1,strategy=settings.strategy,threshold_relaxed=False,
        any_shape_threshold_fallback=any(a.get('fallback_used') for atts in history.values() for a in atts),
        resolved_limits=resolved,attempts_by_group=history)
    save_json(root/'EXPANSION_LOG.json',audit_log)
    updated=dict(config);updated['candidate_limits']=resolved
    return updated,audit_log


def selected_inputs(root,public):
    bundle=read_json(Path(root)/'PAIR_SELECTION_BUNDLE.json')
    if bundle['status']!='selected':raise ValueError('存在空合格集合；停止本轮，不放宽门槛或重新抽样')
    short=read_json(Path(public)/'candidate_shortlist.json');downloads=read_json(Path(public)/'download_manifest.json')
    groups=[];ids=set()
    for row in bundle['groups']:
        if sha256_file(row['selection_path'])!=row['selection_sha256']:raise ValueError('抽样证据改变')
        selection=read_json(row['selection_path']);verify_selection(selection)
        original=next(g for g in short['groups'] if g['key']==row['key'])
        source=next(p for p in original['candidates'] if p['id']==selection['selected_source_id'])
        groups.append(dict(key=row['key'],target=original['target'],candidates=[source]));ids.update((source['id'],original['target']['id']))
    selected_short=dict(schema_version=2,groups=groups,original_shortlist_sha256=digest(short))
    subset=dict(records={pid:downloads['records'][pid] for pid in sorted(ids)},original_downloads_sha256=digest(downloads))
    return selected_short,subset


def annotate_selected(root,public,*,device='cuda'):
    from src.visual.public_annotations import annotate_downloads,contact_sheets
    root,public=Path(root),Path(public);short,downloads=selected_inputs(root,public)
    freeze_json(public/'selected_shortlist.json',short)
    annotations=annotate_downloads(downloads,public/'selected_annotations',device=device)
    contact_sheets(short,annotations,root/'public_review')
    return annotations


def publish_groups(root,public):
    from src.visual.public_annotations import apply_reviews
    root,public=Path(root),Path(public);short,downloads=selected_inputs(root,public)
    annotations=read_json(public/'selected_annotations/annotation_manifest.json')
    reviews=read_json(public/'selected_visual_reviews.json')
    catalogs=apply_reviews(short,downloads,annotations,reviews,public/'selected_annotations')
    bundle=read_json(root/'PAIR_SELECTION_BUNDLE.json');records=[]
    for row in bundle['groups']:
        original=read_json(row['selection_path']);document=read_json(catalogs['catalogs'][row['key']])
        selected=dict(original)
        for role,product in [('target',document['target']),('source',document['candidates'][0])]:
            for field in ('id','title','image'):
                if product[field]!=original[role][field]:raise ValueError('选中图对标注改变了公开身份或原图')
            if 'type_review' in original[role]:product['type_review']=original[role]['type_review']
            selected[role]=validate_product(product,Path(product['image']).parent,owned=role=='source',
                                           public_root=_public_root(public))
        selected['annotation_binding']=dict(drawn_pair_sha256=row['selection_sha256'],annotations_sha256=digest(annotations),reviews_sha256=digest(reviews))
        # 选图阶段同步提取目标商品特点文字与训练指令
        from src.visual.target_features import extract_product_features
        target_feat = extract_product_features(selected['target']['id'])
        selected['target_feature_text'] = target_feat['canonical_sentence']
        selected['target']['brand'] = target_feat['brand']
        selected['target']['attributes'] = target_feat['attributes']
        selected['target']['feature_text'] = target_feat['canonical_sentence']
        selected['target']['training_queries'] = target_feat['training_queries']
        selected_path=root/'groups'/row['key']/'selection/SELECTED_PAIR.json';freeze_json(selected_path,selected)
        # 协议版本目录保护此前已冻结的分组 source/target 及其实验结果。
        directory=root/'data/groups'/row['key']
        published=publish_pair(selected_path,directory,root/'data',
                               input_public_root=_public_root(public))
        cfg=pipeline_config(row['key'],catalogs['catalogs'][row['key']],root,
            selection=published['configuration'],models=published['binding']['models_configuration'])
        cfg.pop('public_catalog');cfg['public_pair']=published['pair_manifest']
        cfg['run_dir']=str((OUTPUT/'runs'/root.name/row['key']).resolve())
        # 感知方法从已绑定的公开图对读取同一正文；配置结构须可被完整入口直接加载。
        cfg['generation']['lucid']['target_text'] = target_feat['canonical_sentence']
        cfg['generation']['lucid']['text_weight'] = 1.0
        freeze_json(directory / 'features.json', target_feat)
        config_path=directory/'pipeline.json';freeze_json(config_path,cfg)
        records.append(dict(key=row['key'],directory=str(directory.resolve()),source=published['source']['image'],target=published['target']['image'],
            manifest=published['pair_manifest'],pipeline_config=str(config_path.resolve())))
    result=dict(schema_version=2,status='published',groups=records,bundle_sha256=sha256_file(root/'PAIR_SELECTION_BUNDLE.json'),
        generation_runs=0,shared_memory_read=False)
    freeze_json(root/'NAMED_PAIRS.json',result);return result


def report(root,public):
    root=Path(root);bundle=read_json(root/'PAIR_SELECTION_BUNDLE.json')
    if bundle['public_plan_sha256']!=sha256_file(root/'PUBLIC_PLAN.json'):raise ValueError('公开报告与冻结计划不一致')
    strategy=bundle.get('strategy','uniform')
    scope=bundle.get('selection_scope','')
    if scope=='golden_band_median_shape_v1':
        title='# 轻量公开图对选取（黄金先验区间 + 中位数外形）'
        summary='以粗类型与目标差异属性排除、前景外形校准门槛、公开零泄漏对照池先验名次带及去重判断合格，再按外形分取中位数候选。未读取受害记忆或正式查询。'
    elif strategy=='median':
        title='# 轻量公开图对选取（中位数外形）'
        summary='仅以类型兼容、灰度全局特征门槛及去重判断合格，并在合格集合中按灰度外形分数取中位数候选。没有读取正式记忆或查询。'
    else:
        title='# 轻量公开图对抽样'
        summary='仅以类型兼容、灰度全局特征门槛及去重判断合格，使用冻结种子在合格集合均匀抽样。没有按最高相似分选图，也没有读取正式记忆或查询。'
    lines=[title,'',summary,'',
        '|组别|目标编号|选中原商品|图片候选数|合格数|选中外形分|合格外形中位数|',
        '|---|---|---|---:|---:|---:|---:|']
    for row in bundle['groups']:
        if sha256_file(row['selection_path'])!=row['selection_sha256']:raise ValueError('公开抽样证据改变')
        selection=read_json(row['selection_path'])
        if row['status']=='selected':verify_selection(selection)
        source=row['source']['id'] if row['source'] else '未选出';n=row['eligible_count']
        draw=selection.get('draw') or {}
        elig_scores=[c['shape_similarity'] for c in selection['candidates'] if c['eligible'] and c.get('shape_similarity') is not None]
        if elig_scores:
            sel_score=next((c['shape_similarity'] for c in selection['candidates'] if c['source_id']==source),None)
            sorted_scores=sorted(elig_scores)
            mid_score=sorted_scores[(len(sorted_scores)-1)//2]
            sel_col=f'{sel_score:.4f}' if sel_score is not None else '-'
            mid_col=f'{mid_score:.4f} (n={n})'
        elif selection.get('shape_threshold_fallback_used') and draw:
            fb_n=draw.get('eligible_count',0)
            sel_s=draw.get('selected_shape_similarity')
            mid_s=draw.get('median_shape_similarity')
            sel_col=f'{sel_s:.4f} (未过门槛回退)' if sel_s is not None else '-'
            mid_col=f'{mid_s:.4f} (带内回退 n={fb_n})' if mid_s is not None else '带内回退'
        else:
            sel_col='-';mid_col='无合格集合'
        lines.append(f"|{row['key']}|{row['target']['id']}|{source}|{row['candidate_count']}|{n}|{sel_col}|{mid_col}|")
    lines+=['','选中图对才进行主体与印刷区标注；逐图视觉核对后发布到协议版本目录，每组继续使用 source、target 文件名。',
        '外形分数是灰度代理，不等于正式图文检索排名或攻击成功概率。开发集商品与正式候选、目标不重叠。',
        '主体、品牌文字等公开核对不合格时停止发布，不根据正式结果重选。正式生成及购物不属于本次选图命令。','']
    path=root/'REPORT.md';path.write_text('\n'.join(lines),encoding='utf-8');return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('calibrate','prepare','select','annotate','publish','report','all'),nargs='?',default='all')
    parser.add_argument('--config',type=Path,default=EXPERIMENT_CONFIGS/'selection_groups29.example.json')
    from src.webshop.config import CATALOG
    parser.add_argument('--products',type=Path,default=CATALOG/'catalog.sqlite',
                        help='默认本项目恢复的完整商品库；也可显式指定公开原始商品数组')
    parser.add_argument('--development',type=Path)
    parser.add_argument('--index',type=Path,default=PUBLIC/'public_products_v2.sqlite')
    parser.add_argument('--verify-source',action='store_true')
    parser.add_argument('--root',type=Path,default=OUTPUT/'public_pair_selection/shape_v2')
    parser.add_argument('--public-root',type=Path,help='默认保存到本次结果目录的数据子目录')
    parser.add_argument('--strategy',choices=VALID_SELECTION_STRATEGIES,help='覆盖配置中的合格集合选取方式（median 或 uniform）')
    parser.add_argument('--expand-limits',nargs='+',type=int,help=f'当基础预算下某组合格数为零时逐步扩展到指定预算（不超过 {MAX_CANDIDATE_LIMIT}，不降门槛）')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(argv);raw_config=read_json(args.config)
    if args.strategy is not None:raw_config=dict(raw_config,strategy=args.strategy)
    if args.expand_limits is not None:raw_config=dict(raw_config,expand_candidate_limits=list(args.expand_limits))
    config=validate_config(raw_config);root=args.root.resolve();public=public_path(args.public_root or root/'data')
    if not root.is_relative_to(OUTPUT.resolve()):parser.error('运行输出须位于当前项目自己的输出目录')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    try:
        with catalog_training_guard(allowed_output_roots=[root,public,PUBLIC]) as audit:
            if args.stage=='calibrate':
                if args.development is None:parser.error('校准需要 --development 指定实际公开视觉标签清单')
                dev_path=public_path(args.development)
                cal_path=public_path(config['calibration_path'])
                backend=create_shape_backend(config.get('models',{}),public/'calibration_features',device=args.device,preprocessing=config.get('preprocessing'))
                try:calibrate(dev_path,cal_path,backend,public_root=(_public_root(dev_path),public,PUBLIC))
                finally:backend.close()
                save_json(root/'ACCESS_AUDIT_calibrate.json',{k:v for k,v in audit.items() if k!='active'})
                print('公开开发集门槛已拟合并冻结。');return 0
            if args.stage in ('prepare','all'):
                if not args.products.is_file():
                    parser.error('未找到本项目已恢复商品库；请先恢复商城数据，或用 --products 指定公开商品文件')
                if config.get('expand_candidate_limits'):
                    config,_=resolve_candidate_limits(config,args.products,public_path(args.index),root,public,
                        device=args.device,verify_source=args.verify_source)
                freeze_plan(config,root)
                proof=read_json(public_path(config['calibration_path']))
                reviews=read_json(public_path(config['type_reviews'])) if config.get('type_reviews') else {}
                extract_excluded=list(proof['development_product_ids'])+sorted(set(config.get('excluded_source_ids') or ()))
                extracted=extract_public_candidates(args.products,config['groups'],public,seed=config['seed'],
                    lexical_limit=config['candidate_limit'],candidate_limits=config.get('candidate_limits'),
                    index_path=public_path(args.index),excluded_ids=extract_excluded,
                    type_reviews=reviews,verify_source=args.verify_source,subtype_filters=config.get('subtype_filters'))
                download_shortlist(random_shortlist(extracted,public),public)
                rank_band=validate_public_rank_band(config.get('public_rank_band'))
                if rank_band and rank_band['enabled']:
                    _prepare_and_download_rank_peers(
                        args.products,config['groups'],extracted,public/'rank_peers',public,
                        seed=config['seed'],peer_count=rank_band['peer_count'],index_path=public_path(args.index),
                        excluded_ids=extract_excluded,type_reviews=reviews,
                        verify_source=args.verify_source,subtype_filters=config.get('subtype_filters'),
                    )
            elif args.stage!='report':
                if (root/'EXPANSION_LOG.json').is_file() and 'candidate_limits' not in config:
                    config=dict(config,candidate_limits=read_json(root/'EXPANSION_LOG.json')['resolved_limits'])
                freeze_plan(config,root)
            if args.stage in ('select','all'):
                selected=select_groups(config,root,public,device=args.device)
                if selected['status']!='selected':
                    report(root,public)
                    save_json(root/'RUN_STATUS.json',dict(status='no_eligible_pair',generation_runs=0,threshold_relaxed=False))
                    save_json(root/('ACCESS_AUDIT_'+args.stage+'.json'),{k:v for k,v in audit.items() if k!='active'})
                    return 3
            if args.stage in ('annotate','all'):annotate_selected(root,public,device=args.device)
            if args.stage in ('publish','all'):
                if not (public/'selected_visual_reviews.json').is_file():
                    save_json(root/'RUN_STATUS.json',dict(status='awaiting_selected_public_visual_review',selected_only=True,generation_runs=0))
                    print('仅选中图对已准备标注；等待实际逐图视觉核对后发布。');return 2
                publish_groups(root,public)
            if args.stage in ('select','publish','report','all'):report(root,public)
            save_json(root/('ACCESS_AUDIT_'+args.stage+'.json'),{k:v for k,v in audit.items() if k!='active'})
        if args.stage=='report':return 0
        state={'prepare':'prepared','select':'drawn_awaiting_selected_annotations','annotate':'awaiting_selected_public_visual_review',
            'publish':'published','all':'published','report':'reported'}[args.stage]
        save_json(root/'RUN_STATUS.json',dict(status=state,victim_memory_read=False,formal_queries_read=False,shopping_runs=0))
        return 0
    except Exception as error:
        save_json(root/'RUN_STATUS.json',dict(status='failed',stage=args.stage,type=type(error).__name__,reason=str(error)))
        raise


if __name__=='__main__':raise SystemExit(main())
