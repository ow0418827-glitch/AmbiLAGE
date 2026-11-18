"""公开字段索引、粗类型候选抽样及有限主图下载；不计算标题相似度。"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import closing
import hashlib
import heapq
import io
import json
from src.paths import ProjectPath as Path
import re
import sqlite3
from urllib.parse import urlparse
from PIL import Image
from src.visual.audit import digest,freeze_json,read_json
from src.visual.image_utils import sha256_file
from src.visual.product_types import classify_public_type,type_compatibility,TYPE_RULES_SHA256

IMAGE_HOSTS=frozenset(('m.media-amazon.com','images-na.ssl-images-amazon.com','images-eu.ssl-images-amazon.com','images.amazon.com','ecx.images-amazon.com'))
OWNERSHIP_POLICY='simulated_controlled_public_subset_sha256_even_v1'
INDEX_POLICY='public_projection_json_or_normalized_sqlite_coarse_type_index_v3'


def product_source_kind(path):
    with Path(path).open('rb') as stream:
        return 'normalized_catalog_sqlite' if stream.read(16) == b'SQLite format 3\x00' else 'raw_products_json'


def sqlite_public_products(path):
    """数据库内只投影公开三字段，不将购买指令或其他商品字段传入选图流程。"""
    source = Path(path).resolve()
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA cache_size=-8192')
        try:
            rows = db.execute("SELECT asin, json_extract(data, '$.name'), "
                              "json_extract(data, '$.images[0]') FROM products ORDER BY ordinal")
            for asin, name, image in rows:
                yield dict(asin=asin, name=name, images=[image])
        except sqlite3.DatabaseError as error:
            raise ValueError('商品库必须提供可只读投影的 products 表及公开名称、图片字段') from error


def public_projection(item):
    asin,name,images=item.get('asin'),item.get('name'),item.get('images')
    if (not isinstance(asin,str) or not re.fullmatch(r'[A-Za-z0-9]{10}',asin)
            or not isinstance(name,str) or not name.strip() or not isinstance(images,list)
            or not images or not isinstance(images[0],str)):return None
    parsed=urlparse(images[0])
    if parsed.scheme not in ('http','https') or parsed.hostname not in IMAGE_HOSTS or parsed.username or parsed.password:return None
    return dict(id=asin,title=' '.join(name.split()),image_url=images[0],source_url='https://www.amazon.com/dp/'+asin)


def simulated_controlled(product_id,seed,target_ids):
    if product_id in target_ids:return False
    value=hashlib.sha256(f'{OWNERSHIP_POLICY}|{seed}|{product_id}'.encode('utf-8')).digest()
    return int.from_bytes(value[:8],'big')%2==0


def build_public_index(products_path,index_path,*,verify_source=False):
    """首次流式扫描；复用时按源路径、字节数、修改时间和类型规则检查，显式选项可复核原文件摘要。"""
    source=Path(products_path).resolve();index=Path(index_path).resolve();stat=source.stat()
    source_kind=product_source_kind(source)
    identity=dict(policy=INDEX_POLICY,source_kind=source_kind,source_path=str(source),source_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns,type_rules_sha256=TYPE_RULES_SHA256)
    if index.exists():
        with closing(sqlite3.connect(index.as_uri()+'?mode=ro',uri=True)) as db:
            saved=json.loads(db.execute('SELECT value FROM meta WHERE key=?',('manifest',)).fetchone()[0])
        if saved['identity']!=identity:raise ValueError('公开源文件或类型规则改变，请重建独立索引')
        if verify_source and sha256_file(source)!=saved['source_sha256']:raise ValueError('原始商品文件内容改变')
        return saved
    index.parent.mkdir(parents=True,exist_ok=True);temporary=index.with_name(index.name+'.building')
    if temporary.exists():raise ValueError('发现未完成公开索引，请使用新索引路径或核查失败证据')
    class HashedReader:
        def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256()
        def read(self,n=-1):
            data=self.stream.read(n);self.hash.update(data);return data
    scanned=projected=0
    with closing(sqlite3.connect(temporary)) as db:
        db.execute('CREATE TABLE products(id TEXT PRIMARY KEY,title TEXT,image_url TEXT,source_url TEXT,category TEXT)')
        db.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)')
        def project_rows(raw_rows):
            nonlocal scanned,projected
            for raw in raw_rows:
                scanned+=1;p=public_projection(raw)
                if p is None:continue
                projected+=1;category=classify_public_type(p)['category']
                prior=db.execute('SELECT title,image_url,source_url FROM products WHERE id=?',(p['id'],)).fetchone()
                row=(p['title'],p['image_url'],p['source_url'])
                if prior is not None and prior!=row:raise ValueError('同一公开商品编号存在不同字段')
                db.execute('INSERT OR IGNORE INTO products VALUES(?,?,?,?,?)',(p['id'],*row,category))
                if scanned%200000==0:print(f'已只读建立 {scanned} 条公开商品索引',flush=True)
        if source_kind == 'normalized_catalog_sqlite':
            project_rows(sqlite_public_products(source))
            source_hash=sha256_file(source)
        else:
            import ijson
            with source.open('rb') as stream:
                reader=HashedReader(stream)
                project_rows(ijson.items(reader,'item',use_float=True))
                # 某些解析器在数组结束后没有继续读取尾部空白。
                while reader.read(1024*1024):pass
                source_hash=reader.hash.hexdigest()
        after=source.stat()
        if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):
            raise ValueError('建立公开索引期间原商品数据发生变化')
        db.execute('CREATE INDEX public_category ON products(category)')
        result=dict(identity=identity,source_sha256=source_hash,scanned_products=scanned,usable_public_products=projected,
            stored_fields=['id','title','image_url','source_url','category_derived_from_public_title'],private_fields_read=False,
            source_projection='public_sql_fields_from_normalized_catalog' if source_kind=='normalized_catalog_sqlite' else 'public_fields_from_raw_json',
            normalization_may_differ_from_original_raw_catalog=source_kind=='normalized_catalog_sqlite')
        db.execute('INSERT INTO meta VALUES(?,?)',('manifest',json.dumps(result,ensure_ascii=False)))
        db.commit()
    temporary.replace(index)
    return result


MAX_CANDIDATE_LIMIT=1024


def extract_public_candidates(products_path,groups,output,*,seed=42,lexical_limit=32,candidate_limits=None,index_path=None,excluded_ids=(),type_reviews=None,verify_source=False,subtype_filters=None):
    """从可复用公开索引均匀截取有限候选；支持按组指定候选上限，不计算语义相似度。"""
    from src.visual.pair_selection import matches_subtype_filter,validate_subtype_filter
    if type(lexical_limit) is not int or not 1<=lexical_limit<=MAX_CANDIDATE_LIMIT:
        raise ValueError(f'公开图片候选预算须在一至 {MAX_CANDIDATE_LIMIT} 之间')
    output=Path(output);source=Path(products_path).resolve();index=Path(index_path or output.parent/'public_products_v2.sqlite')
    target_ids={g['target_id'] for g in groups};group_keys={g['key'] for g in groups}
    if len(target_ids)!=len(groups) or len(group_keys)!=len(groups):raise ValueError('目标及组名须唯一')
    resolved_limits={}
    if candidate_limits is not None:
        if not isinstance(candidate_limits,dict) or set(candidate_limits)-group_keys:
            raise ValueError('按组候选预算只支持配置中的公开组名')
        for k,limit in candidate_limits.items():
            if type(limit) is not int or not 1<=limit<=MAX_CANDIDATE_LIMIT:
                raise ValueError(f'按组候选预算须为一至 {MAX_CANDIDATE_LIMIT} 之间的整数：'+k)
            resolved_limits[k]=limit
    if subtype_filters is not None:
        if not isinstance(subtype_filters,dict) or set(subtype_filters)-group_keys:
            raise ValueError('细分品类过滤只支持配置中的公开组名')
        validated_filters={}
        for k,rule in subtype_filters.items():
            if rule is None:raise ValueError('细分品类过滤规则不能为空')
            validated_filters[k]=validate_subtype_filter(rule,allow_empty=False)
        subtype_filters=validated_filters
    else:
        subtype_filters={}
    manifest=build_public_index(source,index,verify_source=verify_source)
    excluded=set(excluded_ids);type_reviews=type_reviews or {}
    if target_ids&excluded:raise ValueError('正式目标与开发校准商品重叠')
    identity=dict(index_manifest_sha256=digest(manifest),groups=groups,seed=seed,candidate_limit=lexical_limit,
        ownership_policy=OWNERSHIP_POLICY,excluded_ids=sorted(excluded),type_reviews=type_reviews)
    if resolved_limits:identity['candidate_limits']=resolved_limits
    if subtype_filters:identity['subtype_filters']=subtype_filters
    final=output/'public_extraction.json'
    if final.exists():
        saved=read_json(final)
        if saved['identity']!=identity:raise ValueError('公开候选规则改变，请使用新目录')
        return saved
    results=[]
    with closing(sqlite3.connect(index.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        def product(row):
            p=dict(zip(('id','title','image_url','source_url'),row[:4]))
            if p['id'] in type_reviews:p['type_review']=type_reviews[p['id']]
            return p
        for group in groups:
            row=db.execute('SELECT id,title,image_url,source_url FROM products WHERE id=?',(group['target_id'],)).fetchone()
            if row is None:raise ValueError('公开索引缺少固定目标：'+group['target_id'])
            target=product(row);category=classify_public_type(target)
            if category['category'] is None:raise ValueError('目标公开类型未解决，须提供逐商品类型核对：'+target['id'])
            sub_filter=subtype_filters.get(group['key'])
            if not matches_subtype_filter(target['title'],sub_filter,role='target'):
                raise ValueError('固定目标不满足公开细分品类规则：'+target['id'])
            # 类别索引缩小范围；公开人工核对可显式纠正原有未知/多义条目。
            from src.visual.product_types import RULES,COMPATIBLE
            cats=[c for c in RULES if c==category['category'] or frozenset((c,category['category'])) in COMPATIBLE]
            placeholders=','.join('?' for _ in cats)
            def matching():
                seen=set()
                for record in db.execute(f'SELECT id,title,image_url,source_url FROM products WHERE category IN ({placeholders})',cats):
                    p=product(record);seen.add(p['id']);yield p
                for pid in sorted(type_reviews):
                    if pid not in seen:
                        r=db.execute('SELECT id,title,image_url,source_url FROM products WHERE id=?',(pid,)).fetchone()
                        if r:yield product(r)
            counts=dict(type_compatible=0,simulated_controlled=0,development_excluded=0)
            if sub_filter:counts['subtype_filtered']=0
            def eligible():
                for p in matching():
                    if not type_compatibility(p,target)['compatible']:continue
                    if not matches_subtype_filter(p['title'],sub_filter,role='source'):
                        counts['subtype_filtered']+=1;continue
                    counts['type_compatible']+=1
                    if p['id'] in excluded:counts['development_excluded']+=1;continue
                    if not simulated_controlled(p['id'],seed,target_ids):continue
                    counts['simulated_controlled']+=1
                    yield p
            group_limit=resolved_limits.get(group['key'],lexical_limit)
            retained=heapq.nsmallest(group_limit,eligible(),key=lambda p:(digest(['public_candidate_uniform_v2',seed,target['id'],p['id']]),p['id']))
            results.append(dict(key=group['key'],target=target,candidates=retained,counts=dict(counts,candidate_limit=group_limit,retained_for_images=len(retained))))
    result=dict(schema_version=2,identity=identity,index_path=str(index.resolve()),index_manifest=manifest,
        scanned_products=manifest['scanned_products'],usable_public_products=manifest['usable_public_products'],
        groups=results,projected_fields=['asin','name','images[0]'],victim_memory_used=False,formal_queries_used=False,
        ownership_is_experimental_assumption=True)
    freeze_json(final,result);return result


def extract_public_rank_peers(products_path,groups,output,*,seed=42,peer_count=20,index_path=None,excluded_ids=(),candidate_ids_by_group=None,type_reviews=None,verify_source=False,subtype_filters=None):
    """从公开目录零泄漏抽样每组同品类对照池（与候选集、目标、开发集严格互斥）。"""
    from src.visual.pair_selection import matches_subtype_filter,validate_subtype_filter
    if type(peer_count) is not int or peer_count<2:
        raise ValueError('公开先验对照池规模 peer_count 须为不小于二的整数')
    output=Path(output);source=Path(products_path).resolve();index=Path(index_path or output.parent/'public_products_v2.sqlite')
    target_ids={g['target_id'] for g in groups};group_keys={g['key'] for g in groups}
    if len(target_ids)!=len(groups) or len(group_keys)!=len(groups):raise ValueError('目标及组名须唯一')
    if subtype_filters is not None:
        if not isinstance(subtype_filters,dict) or set(subtype_filters)-group_keys:
            raise ValueError('细分品类过滤只支持配置中的公开组名')
        subtype_filters={k:validate_subtype_filter(v,allow_empty=False) for k,v in subtype_filters.items()}
    else:
        subtype_filters={}
    candidate_ids_by_group={k:sorted(set(v)) for k,v in (candidate_ids_by_group or {}).items()}
    manifest=build_public_index(source,index,verify_source=verify_source)
    excluded=set(excluded_ids);type_reviews=type_reviews or {}
    identity=dict(index_manifest_sha256=digest(manifest),groups=groups,seed=seed,peer_count=peer_count,
        excluded_ids=sorted(excluded),candidate_ids_by_group=candidate_ids_by_group,
        type_reviews=type_reviews,subtype_filters=subtype_filters)
    final=output/'public_rank_peers.json'
    if final.exists():
        saved=read_json(final)
        if saved['identity']!=identity:raise ValueError('公开对照池规则改变，请使用新目录')
        return saved
    results=[]
    with closing(sqlite3.connect(index.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        def product(row):
            p=dict(zip(('id','title','image_url','source_url'),row[:4]))
            if p['id'] in type_reviews:p['type_review']=type_reviews[p['id']]
            return p
        for group in groups:
            row=db.execute('SELECT id,title,image_url,source_url FROM products WHERE id=?',(group['target_id'],)).fetchone()
            if row is None:raise ValueError('公开索引缺少固定目标：'+group['target_id'])
            target=product(row);category=classify_public_type(target)
            if category['category'] is None:raise ValueError('目标公开类型未解决：'+target['id'])
            sub_filter=subtype_filters.get(group['key'])
            from src.visual.product_types import RULES,COMPATIBLE
            cats=[c for c in RULES if c==category['category'] or frozenset((c,category['category'])) in COMPATIBLE]
            placeholders=','.join('?' for _ in cats)
            group_candidates=set(candidate_ids_by_group.get(group['key'],()))
            def eligible_peers():
                seen=set()
                for record in db.execute(f'SELECT id,title,image_url,source_url FROM products WHERE category IN ({placeholders})',cats):
                    p=product(record)
                    if p['id'] in seen:continue
                    seen.add(p['id'])
                    if p['id'] in excluded or p['id'] in target_ids or p['id'] in group_candidates:continue
                    if not type_compatibility(p,target)['compatible']:continue
                    # 对照池允许出现具有目标属性或中性同品类竞品，仅保留品类/exclude 一致性
                    if not matches_subtype_filter(p['title'],sub_filter,role='target'):continue
                    yield p
            retained=heapq.nsmallest(peer_count,eligible_peers(),key=lambda p:(digest(['public_rank_peer_v1',seed,target['id'],p['id']]),p['id']))
            if len(retained)<peer_count:
                # 若严格同子类对照不足，从同大类不互斥候选补足（仍与候选/目标/开发集严格互斥）
                already={p['id'] for p in retained}
                def broad_peers():
                    for record in db.execute(f'SELECT id,title,image_url,source_url FROM products WHERE category IN ({placeholders})',cats):
                        p=product(record)
                        if p['id'] in already or p['id'] in excluded or p['id'] in target_ids or p['id'] in group_candidates:continue
                        if not type_compatibility(p,target)['compatible']:continue
                        yield p
                extra=heapq.nsmallest(peer_count-len(retained),broad_peers(),key=lambda p:(digest(['public_rank_peer_v1',seed,target['id'],p['id']]),p['id']))
                retained.extend(extra)
            results.append(dict(key=group['key'],target=target,peers=retained,peer_count=len(retained)))
    result=dict(schema_version=1,identity=identity,index_path=str(index.resolve()),index_manifest=manifest,
        groups=results,victim_memory_used=False,formal_queries_used=False,disjoint_from_candidates=True)
    freeze_json(final,result);return result


def random_shortlist(extracted,output):
    """候选已按固定公开种子截取；保留全部，不加载文本模型或按语义排序。"""
    result=dict(schema_version=2,policy='coarse_type_and_public_hash_sample_v2',extraction_sha256=digest(extracted),
        groups=extracted['groups'],text_model_used=False,visual_score_used=False)
    freeze_json(Path(output)/'candidate_shortlist.json',result);return result


def download_public_image(product,output,*,fetcher=None):
    import requests
    output=Path(output);record_file=output/'image_records'/(product['id']+'.json')
    if record_file.exists():
        prior=read_json(record_file)
        if prior['image_url']!=product['image_url'] or sha256_file(prior['image'])!=prior['image_sha256']:raise ValueError('公开主图地址或缓存内容改变')
        return prior
    parsed=urlparse(product['image_url'])
    if parsed.scheme not in ('http','https') or parsed.hostname not in IMAGE_HOSTS:raise ValueError('只接受公开商品图像主机')
    for attempt in range(2):
        try:
            if fetcher is None:
                with requests.get(product['image_url'],timeout=(10,30),stream=True) as response:
                    response.raise_for_status()
                    if urlparse(response.url).hostname not in IMAGE_HOSTS:raise ValueError('公开图片重定向到未知主机')
                    body=b''
                    for part in response.iter_content(128*1024):
                        body+=part
                        if len(body)>12*1024*1024:raise ValueError('图片超过十二兆字节上限')
            else:body=fetcher(product['image_url'])
            with Image.open(io.BytesIO(body)) as image:
                extension={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}.get(image.format)
                if not extension or min(image.size)<64 or image.width*image.height>25000000:raise ValueError('图片格式或尺寸不合适')
                width,height=image.size;image.verify()
            image_hash=hashlib.sha256(body).hexdigest();path=output/'images'/(image_hash+extension);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and sha256_file(path)!=image_hash:raise ValueError('同名图片缓存内容不符')
            if not path.exists():path.write_bytes(body)
            record=dict(id=product['id'],title=product['title'],source_url=product['source_url'],image_url=product['image_url'],
                image=str(path.resolve()),image_sha256=image_hash,width=width,height=height,image_index=0,attempts=attempt+1)
            freeze_json(record_file,record);return record
        except Exception as error:last=error
    raise RuntimeError(f'公开主图下载失败 {product["id"]}：{last}')


def download_shortlist(shortlist,output,*,workers=6,asset_output=None):
    output=Path(output);assets=Path(asset_output or output)
    products={p['id']:p for g in shortlist['groups'] for p in [g['target']]+g['candidates']};records={};failures={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(download_public_image,p,assets):pid for pid,p in products.items()}
        for future in as_completed(jobs):
            pid=jobs[future]
            try:records[pid]=future.result()
            except Exception as error:failures[pid]=dict(type=type(error).__name__,reason=str(error))
    result=dict(shortlist_sha256=digest(shortlist),records=records,failures=failures,downloads=len(records),failures_count=len(failures))
    freeze_json(output/'download_manifest.json',result)
    for g in shortlist['groups']:
        if g['target']['id'] not in records:raise ValueError('固定目标主图下载失败：'+g['key'])
    return result
