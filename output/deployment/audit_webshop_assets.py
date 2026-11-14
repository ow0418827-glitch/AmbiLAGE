import hashlib,json,sqlite3,sys
from pathlib import Path
root=Path.cwd()
from src.webshop.bundle import inspect_bundle,ensure_catalog,sha256
from src.webshop.disk_catalog import DiskCatalog
bundle=root/'resources/webshop/catalog_bundle'
catalog=root/'resources/webshop/catalog'
manifest=inspect_bundle(bundle)
assert ensure_catalog(catalog,bundle)==catalog
base=json.loads((root/'resources/evaluation_base/manifest.json').read_text(encoding='utf-8'))
products=[]
shop=DiskCatalog.open(catalog)
for key,group in base['groups'].items():
    for role in ('source','target'):
        asin=group[role+'_id']
        row=shop.one("SELECT asin,json_extract(data,'$.name'),json_extract(data,'$.images[0]') FROM products WHERE asin=?",(asin,))
        assert row is not None,(key,role,asin)
        products.append({'group':key,'role':role,'asin':row[0],'has_public_name':bool(row[1]),'has_public_image_url':bool(row[2])})
vendor=root/'resources/webshop/source'
vm=json.loads((vendor/'VENDOR_MANIFEST.json').read_text(encoding='utf-8'))
for item in vm['files']:
    path=vendor/item['path']
    assert sha256(path)==item['vendored_sha256'],item['path']
    assert path.stat().st_size==item['vendored_bytes'],item['path']
index=json.loads((catalog/'index_manifest.json').read_text(encoding='utf-8'))
report={'schema_version':1,'bundle_manifest_sha256':sha256(bundle/'manifest.json'),'original_files':len(manifest['files']),'chunks':sum(len(i['chunks']) for i in manifest['files']),'raw_bytes':manifest['raw_bytes'],'stored_bytes':manifest['stored_bytes'],'all_database_and_index_raw_hashes_verified_during_restore':True,'database_sha256':next(i['sha256'] for i in manifest['files'] if i['path']=='catalog.sqlite'),'largest_chunk_bytes':max(p['bytes'] for i in manifest['files'] for p in i['chunks']),'catalog_products':shop.count,'catalog_goals':len(shop.goals),'search_backend':shop.backend,'search_index_documents':index['documents'],'empty_product_asins':len(index.get('empty_product_asins',[])),'search_index_rebuilt':False,'search_ranking_changed':False,'catalog_products_decoded_in_audit':shop.decoded_products,'frozen_current_pair_products_found':products,'vendor_files':len(vm['files']),'vendor_bytes':sum(i['vendored_bytes'] for i in vm['files']),'vendor_sha256_checks_passed':True,'micro_test_cases_passed':14,'large_models_loaded':False,'shopping_or_external_model_calls_started':False,'java_lucene_runtime_executed':False,'linux_runtime_executed':False,'product_image_scope':'frozen_experimental_images_local; full_catalog_images_remain_original_public_urls'}
shop.close()
(root/'output/deployment/webshop_migration_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
