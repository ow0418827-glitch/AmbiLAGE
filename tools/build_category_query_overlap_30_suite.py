"""根据冻结的三成重合清单与训练抽样清单生成 category_query_overlap_30_v1 测试查询集目录。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.visual_inception_category_evaluation import (
    OVERLAP_30_FAMILY_COUNTS,
    OVERLAP_30_FAMILY_LABELS_ZH,
    OVERLAP_30_PROTOCOL,
    OVERLAP_30_QUERY_ROOT,
    TRAINING_ORACLE_QUERY_ROOT,
    normalize_query_text,
)
from src.paths import EVALUATION_BASE, RESOURCES
from src.visual.audit import read_json
from src.visual.image_utils import sha256_file
from tools.validate_category_query_overlap import (
    MANIFEST as SOURCE_OVERLAP_MANIFEST,
    REUSED_BY_FAMILY,
    validate_query_overlap,
)


NEW_QUERIES_ZH = {
    '01_camera': [
        '哪款门廊安防摄像头在门口逆光时也能拍出清晰画面？',
        '找一款带手动隐私遮挡盖的小型室内安防摄像头。',
        '我想找一款无需付费订阅即可保存录像片段的车库安防摄像头。',
        '哪款户外摄像头能完整拍到窄车道的全宽视野？',
        '我需要一台可以从屋檐向下俯拍的家用安防摄像头。',
        '推荐几款适合租房族安装在有遮挡前门的免打孔安防摄像头。',
        '哪些家用监控摄像头支持与另一位家庭成员共享实时画面？',
        '我想要一款可用于侧院且支持自定义移动侦测区域的摄像头。',
        '推荐一款小巧的安防摄像头，用来记录后门快递投递。',
        '对比插电式和可充电安防摄像头在封闭式门廊中的表现。',
        '找一款能在回放时间轴上标记移动事件的家用安防摄像头。',
        '我需要一款带旋转支架的低调无线安防摄像头。',
        '哪款摄像头不用大泛光灯就能拍清昏暗的地下室楼梯？',
        '推荐几款能将视频存储在本地录像机上的有线家用安防摄像头。',
    ],
    '02_chair': [
        '找一款理发时能顺滑旋转并可锁定的理发椅。',
        '我需要一款适合窄过道店铺的紧凑型液压理发椅。',
        '哪些理发椅的脚踏板对身高较矮的顾客也很舒适？',
        '推荐几款带可拆卸软垫头枕的可躺式理发椅。',
        '哪款理发椅能够降到足够低以方便顾客轻松坐下？',
        '我想要一款易擦拭清洁的深色面料经典理发椅。',
        '对比方形底座和圆形底座理发椅在瓷砖地面上的稳定性。',
        '推荐一款适合长时间护理服务的宽扶手理发椅。',
        '找一款脚踏升降且底座周围空间宽敞的理发椅。',
        '哪些理发椅在顾客后仰剃须时也能保持稳固不晃动？',
        '我想买一款带坚固液压底座的矮背造型理发椅。',
        '推荐几款靠背后仰时腿托能同步抬升的理发椅。',
        '我需要一款座垫耐用、适合日常高频使用的商用理发椅。',
        '哪款可躺式理发椅既适合剪发也方便修面剃须？',
    ],
    '03_cookies': [
        '找一款可每次少烤几块的冷藏大块巧克力曲奇面团。',
        '我需要适合周末烘焙义卖的预先分份曲奇面团。',
        '哪款曲奇面团是冷冻独立球状而不是整块包装的？',
        '推荐几款可以擀平压模制作造型饼干的即烤糖霜曲奇面团。',
        '我想要一款能烤出一盘松软饼干的燕麦曲奇面团。',
        '对比冷藏曲奇面团卷和冷冻面团球在少量烘焙时的便利性。',
        '找一款适合招待客人的无乳制品巧克力豆曲奇面团。',
        '哪款无麸质曲奇面团已分成小块可直接放上烤盘烘烤？',
        '找一款可以自行添加糖针和配料的原味曲奇面团。',
        '我需要一份家庭装未烘焙巧克力碎曲奇面团。',
        '推荐几款能烤出外厚内软口感的即烤曲奇面团。',
        '哪款曲奇面团有适合一两人份的小袋冷冻包装？',
        '我想要一款已经分好造型小份的花生酱曲奇面团。',
        '找一款适合家庭电影夜的多种口味家庭烘焙曲奇面团组合装。',
    ],
    '04_curtain': [
        '找一款能柔和西晒房间午后强光的及地窗帘。',
        '哪些卧室窗帘带有遮挡清晨阳光的遮光内衬？',
        '我需要两条可机洗的窗帘布，用于客厅宽大窗户。',
        '推荐几款适合厨房水槽上方窗户的短款窗帘。',
        '对比亚麻质感窗帘与薄纱窗帘在白天隐私保护上的效果。',
        '什么窗帘挂在租住卧室的细窗帘杆上也能自然垂坠？',
        '我想要一款有肌理的中性色窗帘，可以遮住窗下的暖气片。',
        '找一款宽度足够完全拉合、适合透风露台门的隔热防风窗帘。',
        '推荐几款采用背带挂环、顶部造型利落简约的窗帘。',
        '哪些双层窗帘既能过滤阳光又不会让整个房间变得过暗？',
        '我需要一款易打理的遮光窗帘，用于临街婴儿房窗户。',
        '找一款适合家庭办公室的低饱和素色无内衬纯棉窗帘。',
        '哪款金属孔眼窗帘适合每天反复拉合的窗帘杆？',
        '找一款适合高大餐厅窗户、带低调暗纹的长款窗帘布。',
    ],
    '05_hair_extension': [
        '找一款轻盈的卡扣式接发片，为齐肩细软发质增加丰盈感。',
        '我需要深棕色微卷而非小卷的接发产品。',
        '推荐几款无需去美发沙龙即可自行佩戴的隐形鱼线接发。',
        '哪些贴片式接发的胶条足够轻薄，在发缝附近也能服帖隐形？',
        '我想要几片短款接发帘，用来填补层次剪裁头发的两侧发量。',
        '对比真发和合成纤维卡扣接发在临时正式盘发造型中的效果。',
        '找一款质地接近自然松散小波浪卷的卷发接发。',
        '哪款接发能够自然融合中棕色头发中的柔和挑染？',
        '我需要一款可拆卸、能扎成低马尾造型的接发。',
        '推荐几款冷调黑色的长直卡扣式接发片。',
        '哪些接发套装内同时包含宽幅和窄幅不同尺寸的发片？',
        '我想买一套轻便接发，用来增加麻花辫的长度。',
        '找一款发根深色、发尾逐渐过渡到暖色调的渐变接发。',
        '哪款卡扣式接发有足够的发量可以自然融入浓密头发中？',
    ],
    '06_jeans': [
        '找一款高腰直筒但大腿处仍留有舒适活动空间的牛仔裤。',
        '我需要一条版型利落、适合休闲办公室穿搭的深色牛仔裤。',
        '推荐几款适合小个子穿着的短裤长宽松版型牛仔裤。',
        '哪些微喇牛仔裤臀部略带弹性而腰部不会过紧？',
        '我想要一条浅水洗宽松牛仔裤，适合闷热天气替代厚重丹宁。',
        '对比纯棉硬挺牛仔裤和弹力牛仔裤在长途飞行时的舒适度。',
        '找一款前袋足够深、能稳妥装下手机的直筒牛仔裤。',
        '哪款阔腿牛仔裤拥有合身腰头与全长裤脚设计？',
        '我在找一条无破洞、水洗泛白效果的中腰牛仔裤。',
        '推荐几款裤脚能轻松塞进短靴里的锥形牛仔裤。',
        '哪些宽松牛仔裤提供加长裤长和纯黑水洗款式？',
        '我需要一条日常穿着、膝盖处空间更宽松的修身牛仔裤。',
        '找一款适合搭配平底鞋、裤脚利落的九分蓝色牛仔裤。',
        '哪款柔软丹宁牛仔裤适合腰围介于两个尺码之间的人？',
    ],
    '07_shampoo': [
        '找一款适合发根易出油细软发质、可频繁使用的温和洗发水。',
        '哪款无香精洗发水适合不喜欢洗护产品有香味的人？',
        '我需要一款针对洗后容易干涩粗硬卷发的保湿洗发水。',
        '推荐几款能保持染后红色头发鲜亮不褪色的护色洗发水。',
        '哪款洗发水适合经常游泳后清除泳池水残留物？',
        '我想要一款冲洗干净且不留蜡感残留的轻盈蓬松洗发水。',
        '对比深层清洁洗发水和日常温和洗发水对定型产品残留的清洁效果。',
        '找一款能装进小型旅行铁盒的无香固体洗发皂。',
        '我在找一款可中和漂染金发泛黄调的去黄紫色洗发水。',
        '推荐几款适合潮湿天气容易毛躁粗硬发质的顺滑洗发水。',
        '哪款针对波浪卷发的洗发水采用按压泵头瓶装以便淋浴使用？',
        '我需要一款能洗净干发喷雾残留又不让发尾粗糙的洗发水。',
        '找一款小瓶装温和洗发水，方便随身放在健身包里。',
        '哪些控油清洁洗发水既针对油性头皮又适合染后发质？',
    ],
    '08_speaker': [
        '找一款可以挂在双肩包肩带上的便携蓝牙音箱。',
        '我需要一款小音量播放播客时人声依然清晰的紧凑型无线音箱。',
        '推荐几款适合在小阳台使用的可充电蓝牙音箱。',
        '哪款便携蓝牙音箱支持两部手机轮流连接播放音乐？',
        '我想要一款带有无需低头看就能盲操物理按键的无线音箱。',
        '对比适合野餐连续播放数小时音乐的防水蓝牙音箱。',
        '找一对可以无线串联实现立体声播放的小型音箱。',
        '哪款蓝牙音箱体积小巧能放进拥挤厨房架子且支持接口充电？',
        '我想买一个园艺干活时可以夹在身上的便携蓝牙音箱。',
        '推荐几款同时保留有线音频输入接口的蓝牙音箱。',
        '哪些紧凑型蓝牙音箱在调高重低音时仍能保持人声清亮？',
        '我需要一款带提手挂环和直观电量指示灯的旅行音箱。',
        '找一款适合放在浴室听有声书的防泼溅蓝牙音箱。',
        '哪款小型无线音箱带有防滑橡胶底座，放在桌上不易滑动？',
    ],
    '09_toothbrush': [
        '找一款带轻柔挡位和小巧刷头的电动牙刷。',
        '我需要一支刷头颈部纤细、方便清洁大牙后侧的超软毛手动牙刷。',
        '推荐几款附带旅行收纳盒与基础计时功能的可充电牙刷。',
        '哪款电动牙刷在刷牙用力过猛时会给出明显提示？',
        '我想要一支刷毛柔软、手柄沾水后依然防滑的日常基础牙刷。',
        '对比可替换刷头手动牙刷与入门级可充电电动牙刷的性价比。',
        '找一款电池仓开启轻松方便的干电池驱动牙刷。',
        '哪款紧凑型电动牙刷适合外出旅行一周随身携带？',
        '我在找一支小刷头、尖细磨尖刷毛的手动牙刷。',
        '推荐几款配有两个不同颜色区分刷头的电动牙刷入门套装。',
        '哪款牙刷手柄较宽大，不用用力紧握也能稳稳拿住？',
        '我需要一支适合已换掉幼儿牙刷的儿童使用的软毛手动牙刷。',
        '找几款在每次换区刷牙时会发出停顿提醒的电动牙刷。',
        '哪款家庭装软毛牙刷配有多种不同颜色方便家人区分？',
    ],
}


def _instructions_sha256(instructions: list[str]) -> str:
    serialized = json.dumps(instructions, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def build_overlap_30_suite(
    *,
    source_manifest: Path = SOURCE_OVERLAP_MANIFEST,
    oracle_root: Path = TRAINING_ORACLE_QUERY_ROOT,
    output_root: Path = OVERLAP_30_QUERY_ROOT,
    base: Path = EVALUATION_BASE,
) -> dict:
    overlap_audit = validate_query_overlap(source_manifest)
    if overlap_audit['status'] != 'passed' or overlap_audit['overlap_ratio'] != 0.3:
        raise ValueError('三成重合查询清单未通过基础校验')

    source_doc = read_json(source_manifest)
    oracle_manifest = read_json(oracle_root / 'manifest.json')
    base_manifest_path = base / 'manifest.json'
    base_manifest = read_json(base_manifest_path)

    groups_meta = {}
    for group, oracle_entry in oracle_manifest['groups'].items():
        oracle_q_doc = read_json(oracle_root / oracle_entry['path'])
        oracle_records = oracle_q_doc['query_records']
        reused_records = []
        for family, count in REUSED_BY_FAMILY.items():
            matched = [dict(row) for row in oracle_records if row['query_family'] == family]
            if len(matched) < count:
                raise ValueError(f'{group} 可复用训练查询不足：{family}')
            reused_records.extend(matched[:count])

        new_en = source_doc['groups'][group]['new_queries']
        new_zh = NEW_QUERIES_ZH[group]
        if len(new_en) != 14 or len(new_zh) != 14:
            raise ValueError(f'{group} 新查询中英文条数必须均为 14')

        records = []
        zh_records = []
        instructions = []
        idx = 0
        for row in reused_records:
            idx += 1
            qid = f'{group}_overlap30_{idx:02d}'
            instructions.append(row['instruction'])
            records.append(dict(
                id=qid,
                query_family=row['query_family'],
                instruction=row['instruction'],
                instruction_zh=row['instruction_zh'],
                source_row_id=row['source_row_id'],
                source_file=row['source_file'],
                reused_oracle_id=row['id'],
            ))
            zh_records.append(dict(
                id=qid,
                query_family=row['query_family'],
                query_family_zh=OVERLAP_30_FAMILY_LABELS_ZH[row['query_family']],
                instruction=row['instruction_zh'],
            ))

        for en_text, zh_text in zip(new_en, new_zh):
            idx += 1
            qid = f'{group}_overlap30_{idx:02d}'
            instructions.append(en_text)
            records.append(dict(
                id=qid,
                query_family='synthetic_non_overlap',
                instruction=en_text,
                instruction_zh=zh_text,
            ))
            zh_records.append(dict(
                id=qid,
                query_family='synthetic_non_overlap',
                query_family_zh=OVERLAP_30_FAMILY_LABELS_ZH['synthetic_non_overlap'],
                instruction=zh_text,
            ))

        if len(instructions) != 20 or len({normalize_query_text(t) for t in instructions}) != 20:
            raise ValueError(f'{group} 组装后查询总数或去重数量异常')

        instr_sha = _instructions_sha256(instructions)
        q_rel = f'{group}/queries.json'
        zh_rel = f'{group}/queries.zh.json'
        category = oracle_entry['category']
        target_id = oracle_entry['target_public_id']

        q_doc = dict(
            schema_version=2,
            status='ready',
            method='visual_inception',
            evaluation_protocol=OVERLAP_30_PROTOCOL,
            group=group,
            query_mode='text',
            target_public_id=target_id,
            category=category,
            source_type='mixed_30pct_training_oracle_70pct_non_overlap',
            language='en',
            instruction_policy='category_query_overlap_30_v1',
            instructions=instructions,
            instructions_sha256=instr_sha,
            query_records=records,
            query_family_counts=dict(OVERLAP_30_FAMILY_COUNTS),
            query_image_required=False,
            query_images_generated=False,
            is_real_search_log=False,
            training_use_allowed=False,
        )
        zh_doc = dict(
            schema_version=1,
            language='zh',
            purpose='仅供人工阅读，不传给检索编码器',
            group=group,
            category=category,
            query_records=zh_records,
        )
        _write_json(output_root / q_rel, q_doc)
        _write_json(output_root / zh_rel, zh_doc)

        groups_meta[group] = dict(
            path=q_rel,
            sha256=sha256_file(output_root / q_rel),
            chinese_review_path=zh_rel,
            chinese_review_sha256=sha256_file(output_root / zh_rel),
            target_public_id=target_id,
            category=category,
            query_count=20,
            instructions_sha256=instr_sha,
            query_family_counts=dict(OVERLAP_30_FAMILY_COUNTS),
            public_features_sha256=base_manifest['groups'][group]['features_sha256'],
        )

    manifest = dict(
        schema_version=1,
        protocol=OVERLAP_30_PROTOCOL,
        method='visual_inception',
        query_mode='text',
        source_type='mixed_30pct_training_oracle_70pct_non_overlap',
        language='en',
        instruction_policy='category_query_overlap_30_v1',
        group_count=len(groups_meta),
        queries_per_group=20,
        total_queries=len(groups_meta) * 20,
        query_families=list(OVERLAP_30_FAMILY_COUNTS),
        queries_per_family=dict(OVERLAP_30_FAMILY_COUNTS),
        is_real_search_log=False,
        optimization_may_read=False,
        formal_memory_contents_used_for_authoring=False,
        existing_formal_queries_used_for_authoring=False,
        outcome_selection_used=False,
        base_manifest_sha256=sha256_file(base_manifest_path),
        source_overlap_manifest=str(source_manifest.relative_to(RESOURCES.parent).as_posix()),
        source_overlap_manifest_sha256=sha256_file(source_manifest),
        training_oracle_manifest_sha256=sha256_file(oracle_root / 'manifest.json'),
        training_overlap_audit=dict(
            normalization='unicode_nfkc_whitespace_casefold',
            normalized_exact_overlap_count=54,
            expected_overlap_ratio=0.3,
            semantic_independence_claimed=False,
            unique_training_text_count=oracle_manifest['training_overlap_audit']['unique_training_text_count'],
            training_source_snapshots=oracle_manifest['training_overlap_audit']['training_source_snapshots'],
        ),
        groups=groups_meta,
    )
    _write_json(output_root / 'manifest.json', manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OVERLAP_30_QUERY_ROOT)
    args = parser.parse_args(argv)
    build_overlap_30_suite(output_root=args.output)
    print('BUILT:', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
