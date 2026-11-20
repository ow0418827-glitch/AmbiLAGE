#!/usr/bin/env python3
"""构建 Visual Inception v2 评测查询集（不修改任何 v1 或公开训练集文件）。

生成三个 v2 查询集目录：
1. resources/visual_inception_evaluation/training_queries_oracle_v2
   - 保留 v1 中的 6 条 model_paraphrase 与 4 条 heldout_search_query；
   - 将全部 90 条 category_description 按对应商品 title + descriptions.jsonl 原文提炼为 <=35 词的纯视觉/功能摘要，
     剔除型号编码、URL、HTML 标签、操作手册及退换/质保模板噪声，并记录 distilled_from_raw_description 与 raw_source_text_sha256。
2. resources/visual_inception_evaluation/category_queries_v2 (zero_overlap_v2)
   - 每组 20 条（5 类问法各 4 条：2 条显式非重合 + 2 条无核心品类锚词的隐晦场景/功能查询，共 10/20 implicit_anchor_free）。
3. resources/visual_inception_evaluation/category_query_overlap_30_v2 (overlap_30_v2)
   - 每组 6 条复用 training_queries_oracle_v2（3 条清洗后描述 + 2 条改写 + 1 条搜索词，带 reused_oracle_id）+ 14 条非重合新查询
     （8 条显式非重合 + 6 条无核心品类锚词隐晦查询）。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.experiments.visual_inception_category_evaluation import (
    GROUP_CORE_CATEGORY_BANS,
    ORACLE_PROTOCOL_V2,
    OVERLAP_30_PROTOCOL_V2,
    PROTOCOL_V2,
    V2_DISTILLED_BOILERPLATE_RE,
    _instructions_sha256,
    normalize_query_text,
    validate_category_query_suite,
)
from src.paths import EVALUATION_BASE, RESOURCES
from src.visual.audit import freeze_json, read_json
from src.visual.image_utils import sha256_file

EVAL_RES = RESOURCES / 'visual_inception_evaluation'
ORACLE_V1_ROOT = EVAL_RES / 'training_queries_oracle_v1'
OVERLAP_V1_ROOT = EVAL_RES / 'category_query_overlap_30_v1'
ZERO_V1_ROOT = EVAL_RES / 'category_queries_v1'

ORACLE_V2_ROOT = EVAL_RES / 'training_queries_oracle_v2'
OVERLAP_V2_ROOT = EVAL_RES / 'category_query_overlap_30_v2'
ZERO_V2_ROOT = EVAL_RES / 'category_queries_v2'


DISTILLED_DESCRIPTIONS: dict[str, dict[str, tuple[str, str]]] = {
    '01_camera': {
        '01_camera_oracle_01': (
            '8MP 4K outdoor PoE IP bullet security camera featuring a 2.8mm wide-angle lens, built-in noise-filtering microphone, 98-foot infrared night vision, and weatherproof IP67 metal housing.',
            '提炼公开描述（B097445J2Y）：800万像素4K户外PoE枪机，2.8mm广角镜头、降噪麦克风、98英尺红外夜视及IP67防水金属机身。',
        ),
        '01_camera_oracle_02': (
            '1080p weatherproof IP67 aluminum bullet surveillance camera with a 3.6mm lens, 36 infrared LEDs for 100-foot night vision, and four-in-one hybrid analog video output.',
            '提炼公开描述（B010DC7TQW）：1080p IP67铝壳枪式监控，3.6mm镜头、36颗红外灯支持100英尺夜视及四合一模拟视频输出。',
        ),
        '01_camera_oracle_03': (
            'Wire-free 1080p battery-powered outdoor security camera with IP65 weatherproof casing, SmartSens color night vision, two-way audio, motion alerts, and wireless door entry sensors.',
            '提炼公开描述（B09JB9QKV7）：无线1080p电池供电户外安防套件，IP65防水、彩色夜视、双向语音、移动提醒及无线门窗感应器。',
        ),
        '01_camera_oracle_04': (
            'Smart indoor and outdoor 1080p Wi-Fi security camera featuring color night vision, person and vehicle motion alerts, two-way talk, warning spotlight, and local microSD storage.',
            '提炼公开描述（B07F749MDD）：室内外1080p Wi-Fi智能安防机，支持彩色夜视、人车移动检测提醒、双向通话、警示灯与本地存储卡。',
        ),
        '01_camera_oracle_05': (
            'Compact IP65 water-resistant 1080p indoor and outdoor video camera with a Starlight CMOS sensor for color night vision and two-way app audio.',
            '提炼公开描述（B0919CNXP2）：紧凑型IP65防水1080p室内外摄像机，搭载星光CMOS传感器实现低光彩色夜视与双向语音对讲。',
        ),
        '01_camera_oracle_06': (
            '3MP outdoor bullet IP security camera housed in a weather-resistant IP66 casing with a 2.8 to 12mm varifocal zoom lens and 30-foot infrared night vision.',
            '提炼公开描述（B0170B95U2）：300万像素IP66防水户外枪机，配备2.8至12mm手动变焦镜头与30英尺红外夜视功能。',
        ),
        '01_camera_oracle_07': (
            'Wired 1080p outdoor CCTV bullet security camera with IP66 weatherproof housing, 100-foot infrared night vision, three-axis mounting stand, and four-in-one DVR output.',
            '提炼公开描述（B07SV6H3K7）：有线1080p户外闭路枪式安防机，IP66防尘防水外壳、100英尺红外夜视、三轴支架及四合一硬盘录像输出。',
        ),
        '01_camera_oracle_08': (
            'Battery-powered 3MP wireless outdoor bullet security camera with dual 5dBi Wi-Fi antennas, rechargeable lithium cells, color spotlight night vision, PIR motion alerts, and two-way intercom.',
            '提炼公开描述（B09DPSXJBN）：300万像素无线电池供电户外枪机，双5dBi天线、锂电续航、补光彩色夜视、PIR人体感应及双向语音。',
        ),
        '01_camera_oracle_09': (
            'Compact plug-in 2K indoor Wi-Fi security camera kit featuring on-device human and pet detection, two-way audio communication, and voice assistant smart home compatibility.',
            '提炼公开描述（B08HPQKGZY）：2K插电式室内Wi-Fi双机套装，支持本地人宠识别、双向语音及智能语音助手联动。',
        ),
        '01_camera_oracle_10': (
            'Wire-free solar-powered outdoor pan-tilt 1080p Wi-Fi security camera with a 14400mAh rechargeable battery, IP65 waterproof shell, infrared night vision, PIR detection, and two-way audio.',
            '提炼公开描述（B09LR7RKMF）：太阳能电池供电户外云台1080p Wi-Fi摄像机，14400mAh内置电池、IP65防水、红外夜视及双向通话。',
        ),
    },
    '02_chair': {
        '02_chair_oracle_01': (
            'Heavy-duty vintage hydraulic reclining barber chair upholstered in stitched faux leather with an aluminum alloy frame, 27-inch chrome base, headrest, and 140-degree backrest recline.',
            '提炼公开描述（B084KKKSQK）：复古重型液压可躺理发椅，缝线仿皮软包、铝合金框架、27英寸镀铬底座、可拆卸头枕及140度后倾靠背。',
        ),
        '02_chair_oracle_02': (
            'Stainless steel and waterproof leather hydraulic barber chair featuring high-density rebound sponge cushions, smooth up-and-down lifting, and lockable 360-degree swivel for salon cutting.',
            '提炼公开描述（B09NKCPT92）：不锈钢与防水皮革液压美发椅，高密度回弹海绵坐垫、平稳升降及可锁定360度旋转。',
        ),
        '02_chair_oracle_03': (
            'PU leather reclining barbershop chair with waterproof thick sponge seat cushion, stainless steel and padded armrests, and a wide 135-degree tilting backrest for shaving.',
            '提炼公开描述（B09LLV2DRD）：PU皮革可倾倒剃须理发椅，加厚海绵防水坐垫、不锈钢包皮扶手及135度宽靠背调节。',
        ),
        '02_chair_oracle_04': (
            'Sleek commercial-grade soft leatherette salon styling chair built with dense cushioned seat and backrest, hydraulic pump base, T-shaped chrome footrest, and 360-degree rotation.',
            '提炼公开描述（B09MHPFGWK）：商用柔韧仿皮美发造型椅，厚实海绵座靠垫、液压泵升降底座、T形镀铬脚踏及360度旋转。',
        ),
        '02_chair_oracle_05': (
            'Retro salon styling chair covered in breathable waterproof PVC leather with high-density resilient foam padding and smooth 360-degree locking rotation for barbershops.',
            '提炼公开描述（B09JFL3FKZ）：复古透气防水PVC皮革沙龙理发椅，高弹海绵填充与任意角度锁定360度旋转设计。',
        ),
        '02_chair_oracle_06': (
            'PU leather heavy-duty barber and tattoo chair featuring brushed stainless steel trim, thickened wide backrest cushion, padded leather armrests, height adjustment, and 360-degree swivel.',
            '提炼公开描述（B09L7KJQXD）：重型PU皮革纹身理发椅，拉丝不锈钢饰件、加厚宽靠背、软包扶手及灵活高度升降与360度旋转。',
        ),
        '02_chair_oracle_07': (
            'Retro classic barbershop styling chair constructed with a heavy-duty steel frame, fully upholstered faux leather padding, high-density foam seat cushion, and hydraulic lift base.',
            '提炼公开描述（B09CP7J4C9）：经典复古风重型钢架理发店造型椅，全包覆仿皮高密度泡棉坐垫配液压升降底盘。',
        ),
        '02_chair_oracle_08': (
            'Classic hydraulic salon reclining chair crafted from easy-clean PVC leather with thick high-density foam padding, chrome footrest, curved armrests, and a sturdy round base.',
            '提炼公开描述（B09FY3HDPR）：经典液压可躺沙龙椅，易清洁PVC皮面、高密度加厚海绵、镀铬脚踏、弧形扶手及稳固圆盘底座。',
        ),
        '02_chair_oracle_09': (
            'Boutique black square-base hair salon chair with smooth PVC leather upholstery, high-density sponge padding, stainless steel construction, ergonomic contoured backrest, and 300-pound capacity.',
            '提炼公开描述（B09SD91X8S）：精品黑色方盘美发椅，柔顺PVC皮革面层、高密度海绵及人体工学贴合靠背，承重300磅。',
        ),
        '02_chair_oracle_10': (
            'Antique black reclining barber chair featuring dual-side left and right control levers, 5-inch memory foam cushion, waterproof saddle-stitched PVC leather, and 21-inch chrome base.',
            '提炼公开描述（B08BZ8KM9G）：古典黑双侧控制杆可躺理发椅，5英寸记忆海绵坐垫、防水双鞍线PVC皮及21英寸镀铬底座。',
        ),
    },
    '03_cookies': {
        '03_cookies_oracle_01': (
            'Gluten-free refrigerated chocolate chunk cookie dough made from wholesome baking ingredients, ready to bake into soft rich dessert cookies.',
            '提炼公开描述（B00B049CB8）：无麸质冷藏黑巧块曲奇面团，采用天然烘焙原料，烤后松软浓郁。',
        ),
        '03_cookies_oracle_02': (
            'Bite-sized non-GMO edible chocolate chip cookie dough chunks packed ready to eat straight from the snack pouch.',
            '提炼公开描述（B07SDN84R6）：非转基因可生食一口装巧克力豆曲奇面团块，开袋即食。',
        ),
        '03_cookies_oracle_03': (
            'Pre-portioned triple chocolate chip frozen cookie dough pucks formulated for bakery-style rich chocolate cookies straight from the oven.',
            '提炼公开描述（B00CHU1RGC）：三重巧克力豆冷冻预分装曲奇面团饼，适合烘焙坊级浓郁巧克力的现烤曲奇。',
        ),
        '03_cookies_oracle_04': (
            'Safe eggless raw edible frozen cookie dough in pint tubs made for scooping and eating straight with a spoon.',
            '提炼公开描述（B08Z6FD458）：品脱杯装无蛋安全可生食冷冻曲奇面团，用勺挖取即可直接享用。',
        ),
        '03_cookies_oracle_05': (
            'Rich peanut butter cookie dough loaded with chopped peanut butter cup pieces and creamy chips in pre-portioned baking pucks.',
            '提炼公开描述（B00CHU1RPI）：花生酱风味分装烘焙曲奇面团，融入切碎花生酱夹心杯与香浓花生粒。',
        ),
        '03_cookies_oracle_06': (
            'Handmade ready-to-bake butterscotch chip cookie dough crafted with real butter, fresh eggs, chocolate, and nuts in a scoopable tub.',
            '提炼公开描述（B01N05V944）：手工奶油糖果豆现烤曲奇面团桶，由真实黄油、鲜蛋、巧克力与坚果调制而成。',
        ),
        '03_cookies_oracle_07': (
            'Oversized peanut butter cookie dough mixed with broken peanut butter cup candies and rich peanut butter chips.',
            '提炼公开描述（B00CHU1RPI）：大尺寸花生酱曲奇面团，混合真实花生酱杯碎块与香醇花生酱豆。',
        ),
        '03_cookies_oracle_08': (
            'Gourmet semi-sweet chocolate chunk cookie dough blended with butter, molasses, sugar, and enriched flour for chewy bakery cookies.',
            '提炼公开描述（B00B046VTO）：美食级半甜大黑巧块曲奇面团，融合黄油、糖蜜与小麦粉制成有嚼劲的烘焙饼干。',
        ),
        '03_cookies_oracle_09': (
            'Bulk pre-portioned 1.33-ounce classic chocolate chip cookie dough drops ready to bake into crisp-edged golden cookies.',
            '提炼公开描述（B00CHU03DA）：批量分装1.33盎司经典巧克力豆曲奇面团球，烤制金黄酥脆曲奇。',
        ),
        '03_cookies_oracle_10': (
            'Classic chocolate chip cookie dough made with farm-fresh butter, eggs, and rich semi-sweet chocolate morsels for home baking.',
            '提炼公开描述（B00CHU03DA）：选用农场新鲜黄油、鸡蛋与半甜巧克力豆制成的经典家庭烘焙曲奇面团。',
        ),
    },
    '04_curtain': {
        '04_curtain_oracle_01': (
            'Two-panel semi-sheer polyester voile window curtains with a rod pocket top, printed wood-grain sports motif, and gentle light-filtering drape.',
            '提炼公开描述（B09KRNCBJJ）：双片穿杆式半透涤纶薄纱窗帘，复古木纹运动球类印花，柔和透光。',
        ),
        '04_curtain_oracle_02': (
            'Wrinkle-free 3D mermaid-scale printed polyester blackout grommet curtain panel designed to block harsh sunlight, dampen outdoor noise, and insulate bedroom windows.',
            '提炼公开描述（B086PX13GL）：抗皱3D美人鱼鳞印花涤纶打孔遮光窗帘，阻隔强光紫外线、降噪并平衡室温。',
        ),
        '04_curtain_oracle_03': (
            'Two-panel purple floral insulated blackout window curtains offering noise reduction, energy-saving thermal blocking, and dark room shading for cozy living rooms.',
            '提炼公开描述（B06VWM2BX7）：两片装紫色花卉隔热遮光窗帘，兼具隔音降噪、节能保温与客厅遮光保护隐私效果。',
        ),
        '04_curtain_oracle_04': (
            'Teal Moroccan-patterned microfiber blackout short window valance with a 3-inch rod pocket and triple-weave thermal insulation for kitchen or bathroom.',
            '提炼公开描述（B074FSM89X）：蓝绿色摩洛哥图案超细纤维三层梭织遮光短窗幔，3英寸穿杆袋，适合厨房或浴室。',
        ),
        '04_curtain_oracle_05': (
            'Holiday reindeer and snowflake printed polyester blackout window curtain panels with a rod pocket header, vibrant wash-resistant graphics, and easy care.',
            '提炼公开描述（B0819SK9M3）：木纹背景圣诞驯鹿雪花印花涤纶穿杆遮光窗帘套装，色彩鲜明耐水洗。',
        ),
        '04_curtain_oracle_06': (
            'Extra-wide light grey triple-weaved blackout grommet patio door curtain panel engineered to block up to 99% of light and divide room space.',
            '提炼公开描述（B08CRV4ZB5）：超宽浅灰三层梭织打孔露台推拉门遮光帘，高达85%-99%遮光率并可作空间隔断。',
        ),
        '04_curtain_oracle_07': (
            'Ultra-soft polyester printed window curtain panels featuring minimalist abstract art motifs, light privacy screening, and universal rod compatibility for kids bedrooms.',
            '提炼公开描述（B083GNLLGD）：极简抽象人物建筑印花超柔布艺窗帘两片装，兼顾室内隐私遮蔽与儿童房装饰。',
        ),
        '04_curtain_oracle_08': (
            'Two-panel lightweight chiffon semi-sheer window curtains printed in purple and black plaid Halloween motifs that softly filter daylight and allow airflow.',
            '提炼公开描述（B09CDF77ZB）：紫黑格纹万圣元素印花雪纺半透窗帘两片装，轻盈透气并柔化刺眼日光。',
        ),
        '04_curtain_oracle_09': (
            'Heavy 300gsm navy blue soft velvet blackout curtain pair tailored with rod-pocket headers for rich window drape, privacy, and thermal insulation.',
            '提炼公开描述（B08CKQPP7K）：厚实300gsm藏青色丝绒穿杆遮光窗帘对装，垂坠质感佳并阻挡强光及热量散失。',
        ),
        '04_curtain_oracle_10': (
            'Heavy-duty navy blue polyester blackout curtain pair featuring rod pockets and hanging ring compatibility to adjust height and block bright window light.',
            '提炼公开描述（B07WF9RJD3）：藏青色厚实涤纶穿杆遮光帘对装，可配挂环微调高度并有效隔热挡光。',
        ),
    },
    '05_hair_extension': {
        '05_hair_extension_oracle_01': (
            'Lightweight handmade 12-inch ombre synthetic Kanekalon box braid crochet strands featuring pre-looped roots for quick, soft, full-head protective styling.',
            '提炼公开描述（B07HCST9J1）：手工12英寸渐变色卡尼卡伦细脏辫钩针接发，预留发圈根部、轻盈易戴。',
        ),
        '05_hair_extension_oracle_02': (
            'Nine-piece dark brown 10-inch textured curly and kinky human hair blend clip-in hair extension set designed to match natural coiled curls.',
            '提炼公开描述（B08R5M9565）：九片装深棕10英寸蓬松自然卷混纺真人发卡扣接发套装，贴合自然卷发质。',
        ),
        '05_hair_extension_oracle_03': (
            'Seven packs of soft 12-inch black water-wave synthetic crochet hair strands ready for lightweight passion twist protective braiding styles.',
            '提炼公开描述（B09P8KHXYY）：七包12英寸黑色水波纹钩针扭辫接发，轻盈柔软适合日常及派对造型。',
        ),
        '05_hair_extension_oracle_04': (
            '18-inch 9A Brazilian Remy human hair double-wefted kinky curly 3C-4A clip-in extension set in natural black, heat-stylable and dyeable.',
            '提炼公开描述（B081VF1M6F）：18英寸9A巴西顺发双层缝制3C-4A天然黑紧卷发卡扣接发，可烫染造型。',
        ),
        '05_hair_extension_oracle_05': (
            'Thirty-piece 21.6-inch multicolored curly clip-in synthetic highlight strands crafted from heat-resistant fiber for instant vibrant party streaks without chemical dye.',
            '提炼公开描述（B07KVTSYRK）：30片装21.6英寸彩色卷曲耐高温化纤挑染接发片，免漂染即时增色。',
        ),
        '05_hair_extension_oracle_06': (
            'Four 100-gram bundles of silky straight 100% unprocessed human weave hair adding natural fullness, soft shine, and smooth length without chemical damage.',
            '提炼公开描述（B08D3FXQDD）：4束100克/束100%未处理原生直发发帘，无损自然增厚增长。',
        ),
        '05_hair_extension_oracle_07': (
            'Eight-piece golden long silky straight Remy real human hair double-wefted extensions fitted with gentle stainless clips for seamless everyday volume.',
            '提炼公开描述（B0957HJFYT）：8片装金色长直顺发真人发双层发帘，配不锈钢发夹自然融发增厚。',
        ),
        '05_hair_extension_oracle_08': (
            '14-inch natural-color 100% Brazilian virgin human hair straight weft bundle weighing 100 grams, soft enough to dye, curl, and blend smoothly.',
            '提炼公开描述（B00OV0XD40）：14英寸自然色8A巴西处女直发单束100克发帘，可烫染自然衔接。',
        ),
        '05_hair_extension_oracle_09': (
            '14-inch light brown 100% real human hair clip-in extension piece designed to thicken thinning crowns or build slicked-back top-knot volume.',
            '提炼公开描述（B098K864DJ）：14英寸浅棕色纯真人发卡扣补发片，增加发顶蓬松感或用于男士背头挽髻。',
        ),
        '05_hair_extension_oracle_10': (
            'Four-piece #18-613 blonde silky straight Remy human hair clip-in extension set attaching cleanly within five minutes and restylable with thermal tools.',
            '提炼公开描述（B078JR1WKN）：4片装#18-613浅金色顺滑直发真人发夹片，5分钟快戴且支持热工具造型。',
        ),
    },
    '06_jeans': {
        '06_jeans_oracle_01': (
            "Stretch cotton-lyocell pull-on women's straight-leg jeans designed with a slimming crisscross front panel, classic back pockets, and figure-flattering mid waist.",
            '提炼公开描述（B07JMZQT4B）：女式弹力棉莱赛尔套穿直筒牛仔裤，前片交叉塑形提臀配经典后袋设计。',
        ),
        '06_jeans_oracle_02': (
            "Plus-size mid-rise wide-leg flared bell-bottom women's denim jeans cut for soft casual comfort and easy pairing with everyday tees.",
            '提炼公开描述（B08PS1XYS9）：女式大码中腰阔腿微喇牛仔裤，剪裁舒适百搭日常上衣。',
        ),
        '06_jeans_oracle_03': (
            "100% cotton soft denim women's mid-waisted bell-bottom jeans styled with a tie belt waistband, dual side mini pockets, and zipper closure.",
            '提炼公开描述（B094FVGLXQ）：纯棉中腰系带蝴蝶结喇叭牛仔裤，双侧小口袋与拉链门襟，亲肤休闲。',
        ),
        '06_jeans_oracle_04': (
            "Light yet tough women's climbing denim jeans engineered with stretchy temperature-regulating fabric for unrestricted stem movement beneath climbing harnesses.",
            '提炼公开描述（B07N95CH33）：轻量强韧女式户外攀岩弹性牛仔裤，适配安全带且活动自如、透气调温。',
        ),
        '06_jeans_oracle_05': (
            'High-waisted 90s vintage wide-leg flared denim pants made from stretchy cotton-spandex blend fabric with subtle distressed patchwork street styling.',
            '提炼公开描述（B08TH8SFQW）：90年代复古风高腰宽腿微喇弹力牛仔裤，微破洞拼布街头版型。',
        ),
        '06_jeans_oracle_06': (
            'High-waisted baggy wide-straight boyfriend cargo denim jeans featuring spacious utility pockets, standard-weight soft jean fabric, and relaxed Y2K grunge silhouette.',
            '提炼公开描述（B09MZCG7JD）：复古Y2K高腰宽松直筒男友风工装牛仔裤，标准厚度柔软牛仔面料。',
        ),
        '06_jeans_oracle_07': (
            "Relaxed high-waisted women's Y2K baggy cargo jeans in cotton-poly denim featuring deep front flap utility pockets and full-length straight legs.",
            '提炼公开描述（B09MQTFP11）：高腰宽松Y2K风女式直筒工装牛仔裤，配大容量翻盖实用贴袋。',
        ),
        '06_jeans_oracle_08': (
            '100% cotton black low-rise straight-leg punk denim jeans accented with decorative chain straps, breathable woven fabric, and relaxed streetwear fit.',
            '提炼公开描述（B09Q2QW66Q）：纯棉黑色低腰宽松直筒暗黑朋克风牛仔裤，点缀金属链条配饰。',
        ),
        '06_jeans_oracle_09': (
            "Light-wash low-rise cotton-stretch bootcut women's jeans with classic five-pocket styling, whisker fading, and detailed Southwestern stitching across the back pockets.",
            '提炼公开描述（B07PQYS7SY）：浅水洗低腰弹力微喇五袋女式牛仔裤，后袋带有精致西南风刺绣与猫须做旧。',
        ),
        '06_jeans_oracle_10': (
            "High-waisted slim straight-leg streetwear women's jeans featuring Y2K tie-dye patchwork denim patterns and comfortable everyday stretch tailoring.",
            '提炼公开描述（B08VDLC82P）：高腰修身直筒Y2K扎染印花女式街头风牛仔长裤，日常舒适合身。',
        ),
    },
    '07_shampoo': {
        '07_shampoo_oracle_01': (
            'Sulfate-, silicone-, and paraben-free coconut and hibiscus moisturizing shampoo and conditioner set that gently cleanses scalps and restores dry brittle hair shine.',
            '提炼公开描述（B08GLB5W72）：无硫酸盐无硅油椰子木槿保湿洗发护发套装，温和洁净头皮并改善毛躁干枯发丝。',
        ),
        '07_shampoo_oracle_02': (
            'Pro-vitamin B5 five-in-one moisturizing shampoo formulated without parabens or mineral oils to gently cleanse, strengthen, smooth, and protect color-treated hair.',
            '提炼公开描述（B08HYBPD5Q）：含维B5五效合一保湿洗发水，不含矿物油与尼泊金酯，滋润顺滑并护色强韧。',
        ),
        '07_shampoo_oracle_03': (
            'Medicated moisturizing shampoo designed to cleanse hair gently, relieve dry itchy scalp flakiness, and reduce fiber friction on natural or treated hair.',
            '提炼公开描述（B004D9DKVA）：针对干痒头屑头皮研制的滋润洗发露，舒缓瘙痒同时温和清洁、减少发丝摩擦。',
        ),
        '07_shampoo_oracle_04': (
            'Sulfate-free violet argan oil brightening shampoo and conditioner infused with hydrolyzed proteins to neutralize brassy yellow tones in blonde or curly hair.',
            '提炼公开描述（B08Q77XBC7）：无硫酸盐阿甘油去黄提亮紫色洗护套装，含水解蛋白中和漂染泛黄并修护卷发。',
        ),
        '07_shampoo_oracle_05': (
            'Salon-inspired hair repair shampoo formulated to clean gently while reinforcing weakened inner hair fibers and smoothing rough cuticles.',
            '提炼公开描述（B0791Y9749）：沙龙级发丝纤维修护洗发水，洁净同时强韧受损发芯结构。',
        ),
        '07_shampoo_oracle_06': (
            'Salt-, sodium-, and sulfate-free gentle cleansing coconut-oil shampoo created specifically to protect keratin smoothing treatments and color-treated hair.',
            '提炼公开描述（B01LLT790C）：无盐无钠无硫酸盐温和椰子油洁净洗发水，专为延长角蛋白柔顺及染后护色设计。',
        ),
        '07_shampoo_oracle_07': (
            'Vegan paraben-free and sulfate-free nourishing repair shampoo blended with natural macadamia lipids to cleanse gently and restore soft healthy hair.',
            '提炼公开描述（B00VAIO8XG）：零残忍纯素无硫酸盐澳洲坚果修护滋养洗发水，令干燥发质恢复水润柔亮。',
        ),
        '07_shampoo_oracle_08': (
            'Large 1000ml color-protecting shampoo formulated to gently wash dyed hair while preserving vibrant pigment tone and natural luminous shine.',
            '提炼公开描述（B00G6JFV6M）：1000ml大容量亮彩护色洗发露，温和洁发并锁住染后鲜亮色泽。',
        ),
        '07_shampoo_oracle_09': (
            '32-ounce paraben-free nourishing shampoo enriched with coconut oil, bamboo, chamomile, green tea, and ginseng extracts for soft daily volume.',
            '提炼公开描述（B07KFPMP18）：32盎司植物精粹营养洗发水，蕴含椰子油、竹萃、洋甘菊、绿茶与人参成分。',
        ),
        '07_shampoo_oracle_10': (
            'Dark pomegranate and sandalwood scented 3-in-1 shampoo, conditioner, and body wash delivering rich odor-fighting shower lather for daily freshness.',
            '提炼公开描述（B086M9RW2X）：黑石榴檀香型男士三合一洗发护发沐浴露，泡沫丰盈带来持久清新。',
        ),
    },
    '08_speaker': {
        '08_speaker_oracle_01': (
            'Compact lightweight green macaron-style mini Bluetooth speaker featuring a carrying lanyard, one-click calling, FM radio, USB port, and wide device compatibility.',
            '提炼公开描述（B093GJCJQJ）：清新马卡龙绿迷你便携蓝牙音箱，附挂绳、一键通话、收音机及USB接口。',
        ),
        '08_speaker_oracle_02': (
            '15-inch portable indoor-outdoor PA Bluetooth speaker equipped with an 8-inch woofer, 1.5-inch tweeter, pulsing LED party lights, wired microphone, and FM/USB/SD playback.',
            '提炼公开描述（B09B48YFQV）：15英寸便携派对PA蓝牙音箱，内置8英寸低音与高音单元、律动彩灯及有线麦克风。',
        ),
        '08_speaker_oracle_03': (
            'Dual wireless Bluetooth speaker bundle combining a bass-rich SoundBox Pro+ unit with a rugged waterproof SoundBox Extreme outdoor speaker.',
            '提炼公开描述（B0967T22YR）：无线蓝牙双音箱套装，包含增强低音款桌面音箱与防水户外运动款音箱。',
        ),
        '08_speaker_oracle_04': (
            'Portable black wireless Bluetooth speaker featuring rubberized grip feet for vertical or horizontal placement, built-in speakerphone, and 12-hour battery runtime.',
            '提炼公开描述（B07KDYPR1D）：黑色便携无线蓝牙音箱，横竖可放防滑硅胶脚垫、内置免提通话麦及12小时续航。',
        ),
        '08_speaker_oracle_05': (
            'Portable blue Bluetooth 5.0 stereo wireless speaker offering high-definition sound, TF memory card playback, FM radio tuner, and auxiliary audio input.',
            '提炼公开描述（B07X8W4WN8）：蓝色便携蓝牙5.0立体声无线音箱，支持TF卡直读、FM调频及AUX音频线接入。',
        ),
        '08_speaker_oracle_06': (
            'Ultra-compact 2-inch cylindrical silver mini wireless Bluetooth speaker offering auto-pairing, USB recharging, and up to four hours of portable audio playback.',
            '提炼公开描述（B072MFY2T5）：超迷你2英寸银色圆柱蓝牙小音箱，支持快速自动回连与USB充电。',
        ),
        '08_speaker_oracle_07': (
            'IP55 weather-resistant 30-watt indoor-outdoor wireless Bluetooth speaker featuring a two-way bass radiator, warm flickering LED flame lighting, 150-foot range, and aux input.',
            '提炼公开描述（B086JD8V3Q）：IP55防泼水30瓦室内外无线氛围灯音箱，双分频低音、仿真摇曳火焰灯与150英尺传输。',
        ),
        '08_speaker_oracle_08': (
            'Pocket-sized portable wireless Bluetooth stereo speaker featuring high-definition audio output, built-in hands-free microphone for calls, and universal smartphone/laptop compatibility.',
            '提炼公开描述（B09FRB9PN2）：掌心大小便携无线立体声蓝牙音箱，高清晰音质并内置免提通话麦克风。',
        ),
        '08_speaker_oracle_09': (
            'Vintage natural beech-wood retro wireless Bluetooth speaker offering stereo bass, FM and AM radio dials, swappable faceplate, and 12-hour playtime.',
            '提炼公开描述（B08F1X8ZYY）：天然榉木壳复古收音机造型无线蓝牙音箱，支持FM/AM调频、可换面板及12小时长续航。',
        ),
        '08_speaker_oracle_10': (
            'Waterproof cylinder outdoor portable Bluetooth speaker featuring dual 45mm drivers, colorful RGB light rings, FM radio, TF slot, and 1200mAh battery.',
            '提炼公开描述（B098F1ZFV5）：防水圆柱形户外便携蓝牙音箱，双45mm扬声器、炫彩RGB灯效及FM/TF卡播放。',
        ),
    },
    '09_toothbrush': {
        '09_toothbrush_oracle_01': (
            "Cute cartoon donut-shaped children's manual toothbrush made from soft high-grade silicone bristles for gentle gum-safe mouth cleaning and easy toddler grip.",
            '提炼公开描述（B09QKHK4NP）：可爱卡通甜甜圈造型儿童硅胶软毛牙刷，轻柔护龈且便于宝宝小手握持。',
        ),
        '09_toothbrush_oracle_02': (
            "Ergonomic manual 360-degree U-shaped children's toothbrush molded with food-grade soft silica gel bristles to gently wrap teeth and protect sensitive toddler gums.",
            '提炼公开描述（B09Q5MBQK3）：人体工学360度U形儿童手动硅胶牙刷，食品级软胶毛温和包裹牙齿保护乳牙牙龈。',
        ),
        '09_toothbrush_oracle_03': (
            "Four-pack U-shaped manual silicone and PP children's toothbrushes sized for ages 2 to 12, cleaning front and back tooth surfaces gently.",
            '提炼公开描述（B09R1HJWSY）：四支装耐高温PP+硅胶U形儿童口腔清洁牙刷，分2-6岁与6-12岁大小号贴合齿列。',
        ),
        '09_toothbrush_oracle_04': (
            'Two-pack lightweight waterproof handheld kids U-shaped silicone training toothbrushes designed in small and large arches for ages 2 to 12.',
            '提炼公开描述（B09R9T6WYC）：两支装便携防水儿童U形硅胶训练牙刷，柔和弧形刷头适配幼童至少年刷牙练习。',
        ),
        '09_toothbrush_oracle_05': (
            "Pink two-pack U-shaped silicone children's whole-mouth toothbrushes featuring soft gum-massaging bristles and an easy-grip cartoon handle for toddlers and kids.",
            '提炼公开描述（B09JB4K18P）：粉色两支装全口包覆儿童U形牙刷，食品级硅胶细毛轻柔按摩牙龈并配趣味握柄。',
        ),
        '09_toothbrush_oracle_06': (
            'Blue octopus-themed waterproof electric kids toothbrush clad in soft food-grade silicone with dual gentle sonic modes up to 31,000 SPM for ages 3–12.',
            '提炼公开描述（B08RD8Z9QV）：蓝色小章鱼造型防水儿童电动牙刷，全包柔软硅胶机身带双轻柔声波清洁模式。',
        ),
        '09_toothbrush_oracle_07': (
            'Cartoon toddler training manual toothbrush crafted from eco-friendly bamboo fiber and wheat straw with ultra-soft bristles and a small anti-slip handle.',
            '提炼公开描述（B08TBDTHS1）：天然竹纤维小麦秸秆环保材质宝宝训练小牙刷，卡通防滑柄配超柔细刷毛。',
        ),
        '09_toothbrush_oracle_08': (
            "Bright yellow cow-patterned reusable manual children's U-shaped silicone toothbrush with heat-tolerant soft bristles sized compactly for baby and toddler mouths.",
            '提炼公开描述（B09QCT8FS6）：亮黄色奶牛造型儿童手动U形硅胶牙刷，耐温软刷毛小巧贴合低龄幼童口腔。',
        ),
        '09_toothbrush_oracle_09': (
            'Pink cartoon manual U-shaped kids toothbrush built with boil-safe food-grade silica gel bristles providing gentle 360-degree wrap-around dental cleaning.',
            '提炼公开描述（B09NRP8DSH）：粉色卡通儿童U形手动牙刷，耐高温消毒食品级硅胶头实现360度全方位清洁。',
        ),
        '09_toothbrush_oracle_10': (
            'Blue IPX7 waterproof kids ultrasonic electric toothbrush featuring three gentle cleaning modes, smart timer, cup cover, and three soft U-shaped silicone heads.',
            '提炼公开描述（B088TL63PW）：蓝色IPX7防水儿童超声波电动牙刷，三档轻柔洁净模式配三个U形软硅胶刷头与漱口杯盖。',
        ),
    },
}


# zero_overlap_v2: 5 families x 4 queries = 20 per group.
# Keep 2 clean explicit queries per family from zero_overlap_v1 + add 2 fresh implicit_anchor_free=True queries per family.
IMPLICIT_ANCHOR_FREE_ZERO_V2: dict[str, dict[str, list[tuple[str, str]]]] = {
    '01_camera': {
        'category_direct': [
            ('weatherproof outdoor motion detection video monitor with infrared night vision', '带红外夜视与移动侦测的户外防水视频监控器'),
            ('compact WiFi entryway night watcher with two-way talk and microSD slot', '支持双向通话与存储卡的紧凑型无线门厅夜视监控装置'),
        ],
        'purchase_intent': [
            ('I want to buy a solar-powered outdoor property watcher that alerts my phone when someone walks up the driveway.', '想买一款太阳能户外看护设备，有人走近车道时能向手机推送提醒。'),
            ('Help me order a discreet indoor living-room video monitor supporting local memory card recording.', '帮我挑选一款小巧的室内客厅视频看护仪，支持本地存储卡录像。'),
        ],
        'use_case': [
            ('I need a wire-free porch sentinel to check package deliveries at my front door after dark.', '需要一个免布线的门廊夜视监控装置，方便天黑后查看门口快递投递情况。'),
            ('Suggest an outdoor weather-resistant perimeter guardian for keeping watch over a dim backyard gate.', '推荐一款能应对日晒雨淋的户外院门看护监测仪，夜间也能看清昏暗后院。'),
        ],
        'functional_need': [
            ('Looking for a crisp 2K wireless security eye offering smart person recognition and bright low-light color imaging.', '想找一款画质清晰、带人形识别与微光全彩夜视功能的无线看护设备。'),
            ('Which indoor nursery video monitors deliver smooth real-time mobile viewing and clear voice intercom without monthly cloud fees?', '哪款室内看护录像装置不需要月费也能在手机端流畅看实时画面并清楚对话？'),
        ],
        'selection_comparison': [
            ('Compare battery-powered outdoor entry watchers against wired analog bullet monitors for wide-angle driveway coverage.', '对比电池供电无线门口看护设备与有线模拟枪机在覆盖宽阔车道上的表现。'),
            ('Which pan-tilt indoor home guardians maintain steadier Wi-Fi signal and sharper infrared footage in total darkness?', '哪类家用云台看护仪在全黑环境下夜视更清楚且无线连接更稳？'),
        ],
    },
    '02_chair': {
        'category_direct': [
            ('heavy duty hydraulic 360-degree swivel barbershop workstation seat with steel round base', '带圆形钢底座和360度液压旋转的重型理发工作座椅'),
            ('reclining faux leather hairstyling and shaving lounger with T-footrest and padded armrests', '配T形脚踏和软垫扶手的可倾倒仿皮美发修面躺座'),
        ],
        'purchase_intent': [
            ('I want to purchase a classic leatherette client seating unit with smooth foot-pump height adjustment for my grooming station.', '想买一款带流畅脚踏液压升降的复古仿皮美发工位客座。'),
            ('Help me order a sturdy hydraulic reclining grooming station seat supporting up to 400 pounds.', '帮我下单一款稳固耐用的液压可躺沙龙客座，承重可达400磅。'),
        ],
        'use_case': [
            ('I am setting up a single-station grooming shop and need an easy-wipe vinyl client lounger with 135-degree backrest tilt.', '正在布置单工位理容工作室，需要可135度后仰且易于擦拭清洁的皮质客座。'),
            ('Suggest a tall swivel hairstyling pedestal seat with a chrome footrest that stays rock-steady during beard trims.', '推荐一款配镀铬脚踏的旋转美发工作高座，修须剪发时稳固不晃动。'),
        ],
        'functional_need': [
            ('Looking for salon workstation seating padded with high-density rebound sponge and dual-side recline control levers.', '想找带有高密度回弹海绵坐垫和左右双侧调节拨杆的美发工作室客座。'),
            ('Which commercial tattoo and grooming seats combine stainless steel frames with seam-resistant PVC upholstery?', '哪款纹身与美发沙龙专用客座兼具不锈钢骨架与耐磨防水PVC皮面？'),
        ],
        'selection_comparison': [
            ('Compare round-chrome-base swivel client seats against square-plate salon workstations by floor stability.', '按地面稳定性对比圆盘镀铬旋转客座与方形底座沙龙工作座。'),
            ('Which reclining hairstyling loungers offer smoother hydraulic elevation and thicker memory-foam lower back support?', '哪类可倾倒造型工作座液压升降更顺滑、腰背记忆棉支撑更厚实？'),
        ],
    },
    '03_cookies': {
        'category_direct': [
            ('refrigerated semi-sweet chocolate chunk ready-to-bake pastry batter roll', '即烤型冷藏半甜大黑巧块甜点烘焙面团'),
            ('safe eggless edible chocolate chip dessert dough chunks in snack tubs', '可生食无蛋安全巧克力豆甜品面团杯装零食'),
        ],
        'purchase_intent': [
            ('I want to buy pre-portioned frozen butter and brown sugar baking pucks that crisp nicely on the edges.', '想买分装好的冷冻黄油红糖烘焙甜点面饼，烤出来边缘香脆。'),
            ('Help me order gluten-free plant-based dark chocolate baking dough for warm evening treats.', '帮我选购无麸质植物基黑巧克力碎烘焙面团，晚上烤着做点心。'),
        ],
        'use_case': [
            ('I need heat-treated flour edible batter safe to scoop straight from a pint jar onto vanilla ice cream.', '需要用热处理面粉制作的即食生面团，可以直接挖一勺拌香草冰淇淋吃。'),
            ('Suggest pre-cut slice-and-bake chocolate fudge rounds for making warm sheet-pan dessert snacks in twelve minutes.', '推荐提前切分好的免揉烘焙巧克力软心圆坯，十二分钟就能烤出一盘热甜点。'),
        ],
        'functional_need': [
            ('Looking for peanut-butter-cup stuffed gourmet baking dough made with real creamery butter and no artificial preservatives.', '寻找含真实奶油黄油和花生酱夹心碎、无人工防腐剂的精品即烤甜点面团。'),
            ('Which refrigerated sweet dessert doughs hold thick chewy bakery texture on parchment trays without spreading too flat?', '哪款冷藏甜点面团放烤盘上能保持厚实有嚼劲的质感而不会摊太扁？'),
        ],
        'selection_comparison': [
            ('Compare pre-formed frozen chocolate-chunk baking drops against refrigerated scoop-out butterscotch tubs by oven prep speed.', '从入炉备餐速度对比预成型冷冻黑巧碎烘焙面团与冷藏挖取奶油糖果面糊。'),
            ('Which dairy-free dessert batters bake softer golden centers with richer cocoa flavor?', '哪款无乳甜点烘培坯烤出的中心更松软金黄、可可香气更浓？'),
        ],
    },
    '04_curtain': {
        'category_direct': [
            ('triple-weave thermal room-darkening sliding glass door fabric panels', '三层梭织隔热防晒推拉落地窗遮光布艺片'),
            ('heavy 300gsm navy velvet rod-pocket window coverings for bedroom insulation', '厚重300gsm藏青丝绒穿杆卧室遮光保温窗布'),
        ],
        'purchase_intent': [
            ('I want to buy a wide grommet-top blackout fabric panel to block morning glare across bedroom patio glass.', '想买一面宽幅金属圈顶遮光布，用来挡住卧室玻璃门的刺眼晨光。'),
            ('Help me order lightweight semi-sheer floral printed chiffon window hangings for soft filtered living room sunlight.', '帮我买一对轻盈半透的印花雪纺挂饰布，柔和过滤客厅日光。'),
        ],
        'use_case': [
            ('I work night shifts and need dense 99%-light-blocking insulated bedroom window fabric to keep the room cool and dark.', '上夜班需要高密度全遮光隔热布料挂窗，让卧室大白天也能保持凉爽昏暗。'),
            ('Suggest machine-washable short kitchen cafe panels in warm plaid patterns that maintain privacy near the sink.', '推荐方便机洗的厨房小格纹短窗护布，既能保护隐私又耐油烟易打理。'),
        ],
        'functional_need': [
            ('Looking for wrinkle-free microfiber triple-interwoven window textiles that dampen street traffic sound and prevent sun fading.', '寻找抗皱超细纤维三层织物窗面隔音布，能隔绝马路噪音并减少家具晒褪色。'),
            ('Which extra-wide sliding patio door coverings hang smoothly on steel rods and work well as temporary room dividers?', '哪些加宽落地门遮蔽布拉动顺滑，还能顺便当卧室或客厅隔断使用？'),
        ],
        'selection_comparison': [
            ('Compare thick velvet rod-pocket light blockers against matte polyester grommet window panels by noise reduction.', '对比厚丝绒穿杆遮光布与哑光涤纶打孔挂帘的室内降噪表现。'),
            ('Which semi-sheer voile window textiles allow better summer breeze flow while softening afternoon harshness?', '哪款半透薄纱窗织物既能保证夏季通风，又能把下午强光柔化？'),
        ],
    },
    '05_hair_extension': {
        'category_direct': [
            ('100% Virgin Brazilian straight human hair bundles in 14 to 18 inches', '14到18英寸100%巴西原生直顺发束真发片'),
            ('pre-looped Kanekalon box-braiding crochet synthetic hair strands in ombre tones', '渐变色卡尼卡伦预打圈钩针麻花辫化纤发束'),
        ],
        'purchase_intent': [
            ('I want to buy soft double-wefted real human tresses with silicone-lined metal clips for instant crown fullness.', '想买带护发硅胶夹的双层真人发条，夹上即可瞬间增加头顶和两鬓发量。'),
            ('Help me order heat-resistant water-wave crochet twists gentle enough for a full protective scalp style.', '帮我选一套轻柔水波纹耐热合成螺旋辫束，做全头护发编发不压头皮。'),
        ],
        'use_case': [
            ('I need lightweight Remy straight snap-on strands in #18-613 blonde that attach in five minutes before work.', '需要轻便的#18-613浅金色顺滑真发暗夹发束，上班前五分钟即可快速戴稳。'),
            ('Suggest vibrant 30-piece multicolored curly snap-on highlight locks for festival outfits without bleaching natural roots.', '推荐30片装彩色卷曲即夹即用挑染假发束，去音乐节不漂染也有亮眼发色。'),
        ],
        'functional_need': [
            ('Looking for 3C-4A kinky-curly double-wefted virgin locks that match tight coils and hold curls after flat-iron or dye styling.', '寻找双层缝制3C-4A紧卷原生发条，卷度自然匹配且可安心烫染造型。'),
            ('Which invisible PU tape-in real hair pieces lie flattest against fine thin strands without bumpy weft ridges?', '哪类隐形胶条接发片能紧贴细软发根不显接缝凸起？'),
        ],
        'selection_comparison': [
            ('Compare pre-looped synthetic passion-twist packs against virgin human hair bundles by daily styling heat tolerance.', '按耐高温造型和寿命对比预打圈化纤扭辫包与原生真发束。'),
            ('Which stainless-clip real human hair strands keep fuller thickness at the ends without heavy root pulling?', '哪款不锈钢夹真人发片发尾够丰盈又不会把头皮扯痛？'),
        ],
    },
    '06_jeans': {
        'category_direct': [
            ('high-waisted baggy wide-leg streetwear cargo bottoms in washed cotton weave', '水洗纯棉高腰宽松大阔腿街头工装长裤'),
            ('stretch cotton-lyocell pull-on women straight-fit everyday bottoms with back pockets', '带经典后袋的女式棉莱赛尔弹力套穿直筒常服长裤'),
        ],
        'purchase_intent': [
            ('I want to buy comfortable slimming pull-on indigo bottoms featuring a crisscross tummy-smoothing front panel.', '想买前片带交叉收腹塑形结构、穿脱舒服的深蓝修身直筒休闲长裤。'),
            ('Help me order high-rise bell-bottom streetwear bottoms with deep cargo flap pockets and zipper fly.', '帮我选一条高腰喇叭脚街头风长裤，要带深口袋翻盖和稳固拉链门襟。'),
        ],
        'use_case': [
            ('I need tough stretchy climbing-grade indigo twill bottoms engineered to move freely under a sport harness.', '需要能贴合攀岩安全带又透气高弹的结实靛蓝工装户外长裤。'),
            ('Suggest relaxed Y2K grunge baggy straight bottoms easy to wear daily with chunky sneakers and cropped tees.', '推荐配厚底运动鞋和短上衣都好看的复古Y2K宽松做旧直筒男孩子风长裤。'),
        ],
        'functional_need': [
            ('Looking for curve-flattering bootcut cotton-spandex bottoms accented with Southwestern back-pocket embroidery.', '寻找带西部风后袋刺绣花纹、修身微喇显腿长的棉氨纶弹性长裤。'),
            ('Which high-waisted belted flare wide-leg bottoms stay wrinkle-soft after washing and offer real front key pockets?', '哪款高腰系带微喇阔腿裤洗后依然柔软有型，且配有实用侧兜？'),
        ],
        'selection_comparison': [
            ('Compare heavy 90s baggy cargo cotton bottoms against fitted stretch straight-leg pull-on styles for travel comfort.', '从旅行穿着舒适度对比90年代厚实工装阔腿款与套穿直筒弹力修身款。'),
            ('Which low-rise punk chain streetwear bottoms hold deep black dye cleanest without shrinking after laundry?', '哪类暗黑低腰链条街头长裤洗后更不易缩水掉色？'),
        ],
    },
    '07_shampoo': {
        'category_direct': [
            ('sulfate-free coconut hibiscus moisturizing scalp and hair cleanser bottle', '无硫酸盐椰子木槿滋润洁净护发洗头液'),
            ('pro-vitamin B5 color-protecting botanical hair wash free of parabens and silicones', '含维B5无硅油无尼泊金酯的植物护色洁发露'),
        ],
        'purchase_intent': [
            ('I want to buy a purple argan-oil hair cleanser that tones down brassy yellow warmth on bleached blonde strands.', '想买含阿甘油的紫色护色洁发液，用来中和漂染金发的泛黄杂色。'),
            ('Help me order a 32-ounce botanical moisturizing hair wash formulated with green tea, bamboo, and coconut extracts.', '帮我下单32盎司大瓶装椰子竹萃绿茶草本保湿洗发水替换装。'),
        ],
        'use_case': [
            ('I need a salt- and sodium-free gentle cleansing hair wash safe to protect salon keratin straightening longevity.', '需要无盐无钠温和洁净护发露，避免洗掉沙龙角蛋白拉直护发效果。'),
            ('Suggest a dark pomegranate and sandalwood 3-in-1 shower cleanser convenient for gym bags and quick post-workout washes.', '推荐一款黑石榴檀香三合一沐浴洗发全能洁净露，健身包随身带很省事。'),
        ],
        'functional_need': [
            ('Looking for a dry-and-itchy scalp calming cleanser that removes flaky buildup while reducing brittle strand friction.', '寻找能缓解头皮干痒头屑、清洁后又能顺滑减少打结断裂的舒缓洗发液。'),
            ('Which vegan cruelty-free macadamia hair-repair washes restore shine to split ends without weighing down fine roots?', '哪款无残忍澳洲坚果植物修护洁发露能抚平发梢分叉又不压扁头顶发根？'),
        ],
        'selection_comparison': [
            ('Compare 1000ml color-lock salon hair cleansers against five-in-one pro-vitamin B5 daily washes by lather richness.', '对比1000ml沙龙护色洁发露与五合一维B5日常洗护液的起泡绵密度与护色效果。'),
            ('Which sulfate-free keratin aftercare cleansers preserve glossy smoothness longer in humid weather?', '哪款无硫酸盐角蛋白护理洗发水在潮湿天气下抗毛躁顺滑维持更久？'),
        ],
    },
    '08_speaker': {
        'category_direct': [
            ('waterproof portable wireless Bluetooth audio cylinder with RGB light rings', '带炫彩RGB灯圈的防水便携圆柱无线蓝牙音频播放盒'),
            ('vintage natural beech-wood wireless stereo radio player with 12-hour battery', '天然榉木外壳配12小时长续航的复古无线调频立体声播放器'),
        ],
        'purchase_intent': [
            ('I want to buy a compact pocket wireless music box with a lanyard and one-touch call button for picnics.', '想买一只带挂绳、支持一键免提接听的掌心大小野餐无线音乐小盒子。'),
            ('Help me order an indoor-outdoor 30-watt patio acoustic unit featuring flickering warm lantern flame glow.', '帮我选购一台30瓦带仿真温暖火焰氛围灯的庭院防水无线音频播放器。'),
        ],
        'use_case': [
            ('I need a 15-inch portable karaoke PA system with wired mic, 8-inch woofer, and pulsing LEDs for weekend backyard parties.', '需要一台带8英寸低音单元、有线麦和律动灯效的15英寸户外派对移动功放音响。'),
            ('Suggest an IP55 weather-resistant wireless pool music player that stays steady horizontally or vertically on wet tiles.', '推荐一款IP55防泼水、在泳池边横放竖放都稳固防滑的无线音乐播放设备。'),
        ],
        'functional_need': [
            ('Looking for a rugged Bluetooth 5.0 wireless player offering TF memory-card MP3 playback, FM tuner, and clear calls.', '寻找支持蓝牙5.0、插卡直放MP3、FM调频以及高清免提通话的耐用无线播放器。'),
            ('Which ultra-small cylindrical silver audio players pair automatically on startup and recharge smoothly via USB?', '哪款迷你银色圆柱无线音频小钢炮开机自动回连且USB充电方便？'),
        ],
        'selection_comparison': [
            ('Compare retro wooden FM desktop audio boxes against dual-driver IPX7 waterproof cylindrical travel players by bass punch.', '从低音下潜表现对比木质复古桌面收音播放器与双单元防水圆柱旅行音响。'),
            ('Which compact wireless stereo players hold clearer hands-free call voice quality beyond thirty feet outdoors?', '哪款便携立体声无线音频设备在户外十米外通话拾音仍清晰稳定？'),
        ],
    },
    '09_toothbrush': {
        'category_direct': [
            ('360-degree U-shaped food-grade soft silicone kids oral cleaner for ages 2 to 6', '适合2至6岁幼童的360度U形食品级软硅胶口腔清洁器'),
            ('waterproof octopus cartoon kids ultrasonic oral care wand with gentle timer', '带轻柔洁齿定时器的章鱼卡通防水儿童超声波口腔洁具'),
        ],
        'purchase_intent': [
            ('I want to buy a multi-pack of cartoon U-shaped soft-silica mouth cleaners sized gently for toddlers learning daily hygiene.', '想买一套多支装卡通U形软硅胶口咬洁牙器，方便低龄宝宝自己动手学护齿。'),
            ('Help me order an IPX7 waterproof three-mode electric kid oral cleaner supplied with replaceable U-shaped heads and cup cover.', '帮我买一款IPX7防水三档模式、附带替换U形软硅胶头与杯盖的儿童电动护齿器。'),
        ],
        'use_case': [
            ('I need an adorable donut-handle manual baby cleaner soft enough to clean first teeth without irritating tender gums.', '需要一款防滑甜甜圈手柄、硅胶头超柔软、不伤宝宝娇嫩牙龈的手动洁牙器具。'),
            ('Suggest a natural wheat-straw eco toddler gum cleaner with an anti-slip grip sized nicely for one-to-four-year-old hands.', '推荐采用天然小麦秸秆环保材质、防滑小手柄专为1到4岁宝宝设计的温和乳牙护洁具。'),
        ],
        'functional_need': [
            ('Looking for boil-disinfectable food-grade silicone U-arch mouth cleaners that wrap front and back enamel simultaneously.', '寻找可用热水烫洗消毒、能一次贴合清洁牙面内外两侧的食品级U形硅胶护齿器。'),
            ('Which dual-mode cartoon kids ultrasonic oral wands run quietly at up to 31,000 gentle pulses per minute for sensitive teeth?', '哪款卡通双模式儿童声波洁牙棒震感轻柔静音、每分钟高达31000次保护敏感乳牙？'),
        ],
        'selection_comparison': [
            ('Compare cute cow-patterned U-shaped silicone manual toddler cleaners against ultrasonic three-head electric wands.', '对比可爱奶牛造型U形手动硅胶宝宝洁牙器与三替换头超声波电动护齿棒的实用性。'),
            ('Which small-arch silicone children oral cleaners fit 2-to-6-year-old mouths more comfortably than older-kid frames?', '哪款小号U形硅胶儿童洁齿器比大龄款更贴合2到6岁幼童牙弓轮廓？'),
        ],
    },
}


# For overlap_30_v2, keep 8 explicit non-overlap queries per group from overlap_30_v1 + add 6 fresh implicit_anchor_free=True queries per group.
IMPLICIT_ANCHOR_FREE_OVERLAP30_V2: dict[str, list[tuple[str, str]]] = {
    '01_camera': [
        ('I need a weatherproof front-door motion sentinel that saves clear night footage without monthly fees.', '需要一款无需月费即可高清夜视录制门口动静的户外防水监控装置。'),
        ('Which wireless entryway video guardians send phone snapshots when a person reaches the porch?', '哪款无线门厅视频看护设备能在有人靠近走廊时往手机发实时快照？'),
        ('Find a compact battery-powered perimeter watcher with two-way voice intercom and spotlight alerts.', '找一款带双向语音对讲与照明警示的电池续航户外看护器。'),
        ('Suggest a high-definition 3MP varifocal yard monitor able to zoom cleanly toward a parked car.', '推荐一款可手动变焦看清停车区域细节的300万像素高清院落监测装置。'),
        ('Compare solar-panel rooftop guardians against plug-in indoor pet video watchers for night clarity.', '对比太阳能屋檐户外监控机与插电室内看宠视频设备的夜视清晰度。'),
        ('What compact weather-resistant surveillance eye records 1080p clips locally onto a 128GB SD card?', '哪款防水紧凑型安防之眼能将1080p视频直接存在128GB本地存储卡中？'),
    ],
    '02_chair': [
        ('Find a heavy hydraulic 360-degree swivel barbershop seat built with leatherette armrests and chrome base.', '找一款带皮质扶手与镀铬底盘的360度液压理发工位座椅。'),
        ('Which salon reclining workstations provide dual left-and-right height levers and 140-degree backrest tilt?', '哪款沙龙后仰工位座配备左右双侧拨杆与140度靠背倾角调节？'),
        ('I want an easy-wipe PVC styling pedestal seat cushioned with thick high-density memory sponge.', '想要一款铺满厚实高密度记忆海绵且易擦洗PVC材质的理容工作客座。'),
        ('Compare round-chrome-base styling loungers against boutique square-base leatherette salon seats.', '对比圆盘镀铬款造型座与精品方底座仿皮沙龙座椅的稳定性。'),
        ('Suggest a stainless-steel frame tattoo and shave workstation with removable padded headrest.', '推荐带可拆卸软头枕的不锈钢骨架纹身修面工作室专用座。'),
        ('What retro commercial client lounger handles up to 400 pounds while rotating smoothly at any angle?', '哪款复古商用美发客座承重达400磅且能任意角度顺滑旋转锁定？'),
    ],
    '03_cookies': [
        ('Which safe eggless pint-tub dessert doughs can kids scoop and enjoy raw straight with a spoon?', '哪款无蛋安全品脱杯装甜品生面团可以直接让孩子用勺挖着吃？'),
        ('Compare frozen pre-portioned triple-chocolate baking pucks against scoopable butterscotch dessert tubs.', '对比冷冻预分装三重黑巧烘焙圆坯与可挖取奶油糖果甜点桶。'),
        ('Find gluten-free refrigerated chocolate-chunk baking dough made with wholesome natural ingredients.', '找一款采用天然无麸质原料制成的冷藏黑巧碎烘焙面团。'),
        ('I want peanut-butter-cup stuffed sweet dough rounds that bake soft and chewy inside in twelve minutes.', '想要一款含花生酱杯碎、十二分钟烤出外酥内糯口感的甜点面团坯。'),
        ('Suggest farm-fresh butter gourmet chocolate chunk baking drops convenient for quick family treats.', '推荐用农场鲜黄油配大块黑巧调制、适合家庭快手烘烤的曲奇坯料。'),
        ('What non-GMO bite-sized chocolate chip dough snacks come ready to eat right out of the pouch?', '哪款非转基因一口装巧克力碎甜面块开袋即可当作零食享用？'),
    ],
    '04_curtain': [
        ('I need triple-weave light-blocking sliding patio glass panels to keep afternoon heat out of a living room.', '需要三层梭织遮光防晒推拉门布艺片，隔绝下午客厅强光与热浪。'),
        ('Compare heavy 300gsm navy velvet rod-pocket window textiles against matte grommet bedroom light blockers.', '对比厚重300gsm藏青丝绒穿杆遮光窗布与哑光金属圈卧室遮光挂布。'),
        ('Find cute printed wrinkle-free bedroom window hangings that protect wooden floors from UV fading.', '找一款抗皱印花卧室遮光窗挂布，保护木地板免受紫外线暴晒褪色。'),
        ('Which semi-sheer voile chiffon window panels soften glare while letting gentle summer air circulate?', '哪些半透雪纺薄纱窗户挂布既柔化刺眼日光又透气不闷热？'),
        ('Suggest a teal Moroccan microfiber short kitchen window topper sliding easily onto standard tension rods.', '推荐一条易穿普通伸缩杆、阻隔灶台强光的蓝绿摩洛哥超纤厨房短窗护布。'),
        ('What extra-wide grey insulated room-divider fabric blocks 99% sunlight and muffles traffic noise?', '哪款超宽浅灰隔热隔断布拥有99%高遮光率且能削弱户外车流声？'),
    ],
    '05_hair_extension': [
        ('Which pre-looped Kanekalon box-braiding crochet packs feel lightest on tender scalps for protective styles?', '哪款卡尼卡伦预打圈钩针脏辫束戴起来最轻盈不拉扯敏感头皮？'),
        ('Find 18-inch 9A Brazilian Remy 3C-4A curly real hair clip-on strands soft enough to flat-iron or dye.', '找一套18英寸9A巴西顺发3C-4A自然黑紧卷卡扣真发片，可安心拉直烫染。'),
        ('I need four 100-gram straight unprocessed human locks to blend seamlessly for natural everyday length.', '需要四束百克装未处理原生直顺真人发束，轻松接出自然垂顺的长发量。'),
        ('Compare multicolored heat-resistant synthetic highlight clip-on streaks against double-wefted #18-613 blonde Remy sets.', '对比彩色卷曲耐热化纤挑染夹片与双层#18-613金色真发套片的日常造型感。'),
        ('Suggest lightweight 12-inch water-wave crochet twists ready for sleek summer vacation styling.', '推荐轻盈软弹的12英寸水波纹合成扭辫束，适合夏日出游做防护编发。'),
        ('What stainless-clip silky straight real hair strands attach invisibly in five minutes for fuller volume?', '哪款配不锈钢暗夹的顺直真发片五分钟就能隐形增厚发量？'),
    ],
    '06_jeans': [
        ('Which pull-on straight-fit indigo bottoms use a slimming front crisscross panel for smoother waist lines?', '哪款靛蓝套穿直筒常服裤带有前片交叉收腹结构、穿起来腰腹更显瘦？'),
        ('Compare Y2K high-waisted baggy cargo streetwear bottoms against vintage bell-bottom belted flares.', '对比Y2K复古高腰大阔腿工装街拍裤与中腰系带喇叭款长裤的垂坠版型。'),
        ('Find durable stretchy climbing twill bottoms built to flex smoothly without pinching under sport harnesses.', '找一条弹性极佳且耐磨的户外攀岩靛蓝斜纹长裤，穿戴安全带仍可自由伸腿。'),
        ('I want 100% cotton black low-rise straight punk bottoms accented with subtle metallic side chains.', '想要一条点缀金属侧链的纯棉黑色低腰直筒暗黑街潮长裤。'),
        ('Suggest light-wash stretchy bootcut five-pocket bottoms featuring detailed Southwestern rear-pocket embroidery.', '推荐后袋带有精致西南民族刺绣的浅水洗弹力五袋微喇休闲长裤。'),
        ('What deep-pocketed relaxed baggy wide-leg streetwear bottoms pair best with chunky sneakers all year?', '哪款带大容量工装侧袋的宽松阔腿街头长裤一年四季配老爹鞋最帅气？'),
    ],
    '07_shampoo': [
        ('Which salt- and sulfate-free coconut cleansers keep salon keratin smoothing treatments glossy longer?', '哪款无盐无硫酸盐椰子油洁发露能让沙龙角蛋白拉直护发效果维持得更久？'),
        ('Find a violet argan-oil sulfate-free hair wash formulated to remove brassy yellow tones from blonde strands.', '找一款含阿甘油的紫色无硫酸盐洁发乳，专为漂染金发去黄亮泽。'),
        ('I need a gentle dry-itchy-scalp calming hair cleanser that stops flakiness without stripping natural oils.', '需要一款舒缓干痒头屑头皮且不拔干天然皮脂的温和护发洗液。'),
        ('Compare five-in-one pro-vitamin B5 daily moisturizing washes against 1000ml color-protecting salon cleansers.', '对比五合一维B5日常润发露与1000ml大容量沙龙护色洁发液的修护力。'),
        ('Suggest a 32-ounce paraben-free botanical coconut, bamboo, and green-tea hair wash for bouncy softness.', '推荐32盎司大瓶无防腐剂椰油竹萃绿茶草本洗头水，洗后清爽蓬松。'),
        ('What dark pomegranate and sandalwood 3-in-1 shower cleanser cleans hair and body cleanly after sports?', '哪款黑石榴檀香三合一洁发沐浴露适合运动后一步搞定洗发和淋浴？'),
    ],
    '08_speaker': [
        ('I want a mini pastel-green macaron portable audio box with carrying strap, FM radio, and USB port.', '想要一只清新马卡龙绿带挂绳、支持收音和USB的便携迷你音乐小盒子。'),
        ('Compare IP55 weatherproof 30-watt patio flame-glow music players against rugged wireless beach cylinders.', '对比30瓦防泼水带火焰氛围灯的庭院音乐播放器与户外防水圆柱蓝牙音箱。'),
        ('Find an indoor-outdoor 15-inch karaoke PA audio tower fitted with 8-inch woofer, party lights, and wired mic.', '找一台配8英寸低音炮、霓虹灯和有线麦克风的15英寸便携派对唱跳移动音箱。'),
        ('Which compact beech-wood retro wireless desktop players feature classic FM/AM dials and 12-hour battery?', '哪款天然榉木复古桌面无线音频播放器带经典AM/FM调频旋钮与12小时续航？'),
        ('Suggest a rubberized-grip portable black stereo player built for safe horizontal or vertical shelf placement.', '推荐底座防滑、横竖摆放皆稳的黑色无线便携立体声家庭播放器。'),
        ('What ultra-tiny 2-inch cylindrical silver wireless audio unit pairs automatically and lasts four hours?', '哪款迷你2英寸银色圆柱蓝牙无线小钢炮支持自动回连并可连续播放四小时？'),
    ],
    '09_toothbrush': [
        ('Which 360-degree U-shaped food-grade soft-silica kids oral cleaners fit ages 2 to 12 comfortably?', '哪款360度U形食品级软硅胶儿童护齿清洁器在2至12岁不同年龄段都好咬合？'),
        ('Compare blue octopus dual-mode waterproof ultrasonic kids oral wands against cute cow U-shaped manual cleaners.', '对比蓝色小章鱼双模式防水儿童声波洁牙棒与奶牛造型手动U形软胶护齿器。'),
        ('Find natural bamboo-fiber wheat-straw toddler dental trainers equipped with extra-soft anti-gum-hurt bristles.', '找一款竹纤维小麦秸秆环保柄、不伤宝宝娇嫩牙龈的超柔毛乳牙训练器。'),
        ('I need pink cartoon whole-mouth silicone baby cleaners safe for warm-water boil disinfection after brushing.', '需要耐热水烫洗消毒、温和包裹宝宝全口乳牙的粉色卡通硅胶清洁器。'),
        ('Suggest an IPX7 waterproof three-mode children ultrasonic oral wand complete with cup lid and three heads.', '推荐配漱口杯盖与三枚替换头、IPX7全身防水的三档儿童超声波护齿器。'),
        ('What ergonomic donut-shaped silicone toddler gum cleaner stays easiest for little baby hands to hold?', '哪款甜甜圈防滑造型的软硅胶婴儿口腔洁具最方便宝宝小手握住自己清洁？'),
    ],
}


def verify_non_boilerplate(text: str, label: str) -> None:
    wc = len(text.split())
    if not (1 <= wc <= 35):
        raise ValueError(f'{label}: word count {wc} outside 1..35: {text}')
    m = V2_DISTILLED_BOILERPLATE_RE.search(text)
    if m:
        raise ValueError(f'{label}: matched boilerplate regex {m.group(0)!r}: {text}')


def write_pair(out_dir: Path, group: str, en_doc: dict, zh_doc: dict) -> tuple[Path, Path]:
    g_dir = out_dir / group
    g_dir.mkdir(parents=True, exist_ok=True)
    q_path = g_dir / 'queries.json'
    zh_path = g_dir / 'queries.zh.json'
    freeze_json(q_path, en_doc)
    freeze_json(zh_path, zh_doc)
    return q_path, zh_path


def main() -> None:
    base_manifest = read_json(EVALUATION_BASE / 'manifest.json')
    base_groups = list(base_manifest['groups'])

    # Load all training text norms cleanly to guard zero_overlap / synthetic_non_overlap
    o1_manifest = read_json(ORACLE_V1_ROOT / 'manifest.json')
    z1_manifest = read_json(ZERO_V1_ROOT / 'manifest.json')
    ov1_manifest = read_json(OVERLAP_V1_ROOT / 'manifest.json')

    snapshots = o1_manifest['training_overlap_audit']['training_source_snapshots']
    all_training_norms: set[str] = set()
    for snap in snapshots:
        sp = REPO / snap['path']
        for line in sp.read_text(encoding='utf-8').splitlines():
            if line.strip():
                all_training_norms.add(normalize_query_text(json.loads(line)['text']))

    # 1. Build training_queries_oracle_v2
    o2_group_entries = {}
    for group in base_groups:
        src_q = read_json(ORACLE_V1_ROOT / group / 'queries.json')
        src_zh = read_json(ORACLE_V1_ROOT / group / 'queries.zh.json')
        desc_path = REPO / f'data/public/visual_inception_inputs/catalog_v5/{group}/descriptions.jsonl'
        raw_rows = {
            str(r['id']): r['text']
            for line in desc_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
            for r in [json.loads(line)]
        }

        new_records = []
        new_zh_records = []
        new_instructions = []
        for rec, zrec in zip(src_q['query_records'], src_zh['query_records']):
            rec2 = copy.deepcopy(rec)
            zrec2 = copy.deepcopy(zrec)
            if rec['query_family'] == 'category_description':
                rid = rec['id']
                srow_id = str(rec['source_row_id'])
                raw_text = raw_rows[srow_id]
                raw_sha = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()
                en_sum, zh_sum = DISTILLED_DESCRIPTIONS[group][rid]
                verify_non_boilerplate(en_sum, f'oracle_v2/{group}/{rid}')
                rec2['instruction'] = en_sum
                rec2['instruction_zh'] = zh_sum
                rec2['distilled_from_raw_description'] = True
                rec2['raw_source_text_sha256'] = raw_sha
                zrec2['instruction'] = zh_sum
            new_records.append(rec2)
            new_zh_records.append(zrec2)
            new_instructions.append(rec2['instruction'])

        assert len({normalize_query_text(t) for t in new_instructions}) == 20, f'Duplicate in oracle_v2/{group}'
        instr_sha = _instructions_sha256(new_instructions)
        en_doc = copy.deepcopy(src_q)
        en_doc['evaluation_protocol'] = ORACLE_PROTOCOL_V2
        en_doc['instruction_policy'] = 'training_oracle_v2_distilled_descriptions_20'
        en_doc['instructions'] = new_instructions
        en_doc['instructions_sha256'] = instr_sha
        en_doc['query_records'] = new_records

        zh_doc = copy.deepcopy(src_zh)
        zh_doc['query_records'] = new_zh_records
        q_path, zh_path = write_pair(ORACLE_V2_ROOT, group, en_doc, zh_doc)

        entry = copy.deepcopy(o1_manifest['groups'][group])
        entry['sha256'] = sha256_file(q_path)
        entry['chinese_review_sha256'] = sha256_file(zh_path)
        entry['instructions_sha256'] = instr_sha
        o2_group_entries[group] = entry

    o2_manifest = copy.deepcopy(o1_manifest)
    o2_manifest['protocol'] = ORACLE_PROTOCOL_V2
    o2_manifest['instruction_policy'] = 'training_oracle_v2_distilled_descriptions_20'
    o2_manifest['v1_source_manifest_sha256'] = sha256_file(ORACLE_V1_ROOT / 'manifest.json')
    o2_manifest['groups'] = o2_group_entries
    freeze_json(ORACLE_V2_ROOT / 'manifest.json', o2_manifest)

    # 2. Build category_queries_v2 (zero_overlap_v2)
    z2_group_entries = {}
    for group in base_groups:
        src_q = read_json(ZERO_V1_ROOT / group / 'queries.json')
        src_zh = read_json(ZERO_V1_ROOT / group / 'queries.zh.json')
        ban_re = GROUP_CORE_CATEGORY_BANS[group]

        # Group src records by family, keep first 2 explicit + replace 3rd/4th with 2 implicit_anchor_free=True
        fam_seen: dict[str, int] = {}
        new_records = []
        new_zh_records = []
        new_instructions = []
        for idx, (rec, zrec) in enumerate(zip(src_q['query_records'], src_zh['query_records'])):
            fam = rec['query_family']
            k = fam_seen.get(fam, 0)
            fam_seen[fam] = k + 1
            rec2 = copy.deepcopy(rec)
            zrec2 = copy.deepcopy(zrec)
            rec2['id'] = rec['id'].replace('_zero_', '_zerov2_') if '_zero_' in rec['id'] else f'{group}_zerov2_{idx+1:02d}'
            zrec2['id'] = rec2['id']
            if k < 2:
                rec2['implicit_anchor_free'] = False
            else:
                en_imp, zh_imp = IMPLICIT_ANCHOR_FREE_ZERO_V2[group][fam][k - 2]
                assert not ban_re.search(en_imp), f'Banned word leaked in zero_v2/{group}/{fam}: {en_imp}'
                assert normalize_query_text(en_imp) not in all_training_norms, f'Overlap leak in zero_v2/{group}: {en_imp}'
                rec2['instruction'] = en_imp
                rec2['instruction_zh'] = zh_imp
                rec2['implicit_anchor_free'] = True
                zrec2['instruction'] = zh_imp
            new_records.append(rec2)
            new_zh_records.append(zrec2)
            new_instructions.append(rec2['instruction'])

        assert len({normalize_query_text(t) for t in new_instructions}) == 20, f'Duplicate in zero_v2/{group}'
        instr_sha = _instructions_sha256(new_instructions)
        en_doc = copy.deepcopy(src_q)
        en_doc['evaluation_protocol'] = PROTOCOL_V2
        en_doc['instruction_policy'] = 'zero_overlap_v2_10explicit_10implicit_20'
        en_doc['instructions'] = new_instructions
        en_doc['instructions_sha256'] = instr_sha
        en_doc['query_records'] = new_records

        zh_doc = copy.deepcopy(src_zh)
        zh_doc['query_records'] = new_zh_records
        q_path, zh_path = write_pair(ZERO_V2_ROOT, group, en_doc, zh_doc)

        entry = copy.deepcopy(z1_manifest['groups'][group])
        entry['sha256'] = sha256_file(q_path)
        entry['chinese_review_sha256'] = sha256_file(zh_path)
        entry['instructions_sha256'] = instr_sha
        z2_group_entries[group] = entry

    z2_manifest = copy.deepcopy(z1_manifest)
    z2_manifest['protocol'] = PROTOCOL_V2
    z2_manifest['instruction_policy'] = 'zero_overlap_v2_10explicit_10implicit_20'
    z2_manifest['v1_source_manifest_sha256'] = sha256_file(ZERO_V1_ROOT / 'manifest.json')
    z2_manifest['groups'] = z2_group_entries
    freeze_json(ZERO_V2_ROOT / 'manifest.json', z2_manifest)

    # 3. Build category_query_overlap_30_v2 (overlap_30_v2)
    ov2_group_entries = {}
    for group in base_groups:
        src_q = read_json(OVERLAP_V1_ROOT / group / 'queries.json')
        src_zh = read_json(OVERLAP_V1_ROOT / group / 'queries.zh.json')
        o2_q = read_json(ORACLE_V2_ROOT / group / 'queries.json')
        o2_by_id = {r['id']: r for r in o2_q['query_records']}
        ban_re = GROUP_CORE_CATEGORY_BANS[group]

        syn_idx = 0
        new_records = []
        new_zh_records = []
        new_instructions = []
        for idx, (rec, zrec) in enumerate(zip(src_q['query_records'], src_zh['query_records'])):
            rec2 = copy.deepcopy(rec)
            zrec2 = copy.deepcopy(zrec)
            rec2['id'] = f'{group}_overlap30v2_{idx+1:02d}'
            zrec2['id'] = rec2['id']
            if rec['query_family'] in ('category_description', 'model_paraphrase', 'heldout_search_query'):
                oid = rec['reused_oracle_id']
                orec = o2_by_id[oid]
                rec2['instruction'] = orec['instruction']
                rec2['instruction_zh'] = orec['instruction_zh']
                zrec2['instruction'] = orec['instruction_zh']
                if rec['query_family'] == 'category_description':
                    rec2['distilled_from_raw_description'] = True
                    rec2['raw_source_text_sha256'] = orec['raw_source_text_sha256']
            else:
                if syn_idx < 8:
                    rec2['implicit_anchor_free'] = False
                else:
                    en_imp, zh_imp = IMPLICIT_ANCHOR_FREE_OVERLAP30_V2[group][syn_idx - 8]
                    assert not ban_re.search(en_imp), f'Banned word leaked in overlap30_v2/{group}: {en_imp}'
                    assert normalize_query_text(en_imp) not in all_training_norms, f'Overlap leak in overlap30_v2/{group}: {en_imp}'
                    rec2['instruction'] = en_imp
                    rec2['instruction_zh'] = zh_imp
                    rec2['implicit_anchor_free'] = True
                    zrec2['instruction'] = zh_imp
                syn_idx += 1
            new_records.append(rec2)
            new_zh_records.append(zrec2)
            new_instructions.append(rec2['instruction'])

        assert len({normalize_query_text(t) for t in new_instructions}) == 20, f'Duplicate in overlap30_v2/{group}'
        instr_sha = _instructions_sha256(new_instructions)
        en_doc = copy.deepcopy(src_q)
        en_doc['evaluation_protocol'] = OVERLAP_30_PROTOCOL_V2
        en_doc['instruction_policy'] = 'category_query_overlap_30_v2'
        en_doc['instructions'] = new_instructions
        en_doc['instructions_sha256'] = instr_sha
        en_doc['query_records'] = new_records

        zh_doc = copy.deepcopy(src_zh)
        zh_doc['query_records'] = new_zh_records
        q_path, zh_path = write_pair(OVERLAP_V2_ROOT, group, en_doc, zh_doc)

        entry = copy.deepcopy(ov1_manifest['groups'][group])
        entry['sha256'] = sha256_file(q_path)
        entry['chinese_review_sha256'] = sha256_file(zh_path)
        entry['instructions_sha256'] = instr_sha
        ov2_group_entries[group] = entry

    ov2_manifest = copy.deepcopy(ov1_manifest)
    ov2_manifest['protocol'] = OVERLAP_30_PROTOCOL_V2
    ov2_manifest['instruction_policy'] = 'category_query_overlap_30_v2'
    ov2_manifest['training_oracle_manifest_sha256'] = sha256_file(ORACLE_V2_ROOT / 'manifest.json')
    ov2_manifest['v1_source_manifest_sha256'] = sha256_file(OVERLAP_V1_ROOT / 'manifest.json')
    ov2_manifest.pop('source_overlap_manifest', None)
    ov2_manifest.pop('source_overlap_manifest_sha256', None)
    ov2_manifest['groups'] = ov2_group_entries
    freeze_json(OVERLAP_V2_ROOT / 'manifest.json', ov2_manifest)

    # Validate all 6 suites end-to-end
    for suite in ('training_oracle', 'overlap_30', 'zero_overlap', 'training_oracle_v2', 'overlap_30_v2', 'zero_overlap_v2'):
        res = validate_category_query_suite(query_suite=suite)
        print(f'[VALIDATED] {suite}: status={res["status"]}, protocol={res["protocol"]}, overlap={res["training_normalized_exact_overlap_count"]}')


if __name__ == '__main__':
    main()
