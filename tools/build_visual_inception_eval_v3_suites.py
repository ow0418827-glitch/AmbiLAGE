#!/usr/bin/env python3
"""构建 Visual Inception v3 严格嵌套对齐评测集（不修改 v1 / v2 / 公开训练数据）。

三套 v3 评测集严格共用同构 10 / 6 / 4 子池结构：
- training_queries_oracle_v3 (100%): 10 去单品噪声公开描述 + 6 长度/噪声对齐改写问句 + 4 干净真搜短词（均绑定训练包）。
- category_queries_v3 (zero_overlap_v3, 0%): 10 品类功能摘要（6 显式 + 4 隐晦无锚词）+ 6 自然问句（2 显式 + 4 隐晦）+ 4 非重合短搜词；
  与 training_bundle 全部训练句及 target_concept_text 保持 0 精确重合与 0 3-gram 重合。
- category_query_overlap_30_v3 (overlap_30_v3, 30%): 严格由 oracle_v3 (3+2+1=6) 与 zero_overlap_v3 (7+4+3=14，含 3+3=6 条隐晦) 嵌套拼装。
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
    ORACLE_FAMILIES,
    ORACLE_FAMILY_COUNTS,
    ORACLE_FAMILY_LABELS_ZH,
    ORACLE_PROTOCOL_V3,
    OVERLAP_30_PROTOCOL_V3,
    OVERLAP_30_V3_FAMILY_LABELS_ZH,
    PROTOCOL_V3,
    V2_DISTILLED_BOILERPLATE_RE,
    V3_SINGLE_ITEM_NOISE_RE,
    ZERO_V3_FAMILY_LABELS_ZH,
    _instructions_sha256,
    _token_3grams,
    normalize_query_text,
    validate_category_query_suite,
)
from src.paths import EVALUATION_BASE, RESOURCES
from src.visual.audit import freeze_json, read_json
from src.visual.image_utils import sha256_file

EVAL_RES = RESOURCES / 'visual_inception_evaluation'
ORACLE_V1_ROOT = EVAL_RES / 'training_queries_oracle_v1'
ORACLE_V3_ROOT = EVAL_RES / 'training_queries_oracle_v3'
OVERLAP_V3_ROOT = EVAL_RES / 'category_query_overlap_30_v3'
ZERO_V3_ROOT = EVAL_RES / 'category_queries_v3'

# 90 generic category_description items distilled from oracle_v1/descriptions.jsonl (10-18 words, mean 12-15)
ORACLE_V3_DESC: dict[str, list[tuple[str, str]]] = {
    '01_camera': [
        ('Weatherproof outdoor wired security camera offering clear night vision, wide viewing angle, and voice recording.', '防水有线户外安防摄像头，具备清晰夜视、广角视野与语音录制。'),
        ('Weatherproof metal bullet surveillance camera delivering clear daytime monitoring and long-range infrared night vision.', '耐候金属枪式监控摄像头，提供清晰日间监测与远距离红外夜视。'),
        ('Wire-free battery-powered outdoor security camera with weatherproof casing, color night vision, audio, and alerts.', '免布线电池供电户外安防摄像头，具备防水外壳、夜视、语音和警报功能。'),
        ('Smart indoor and outdoor Wi-Fi security camera featuring night vision, motion notifications, and local recording.', '智能室内外无线安防摄像头，支持夜视、移动侦测提醒和本地录像存储。'),
        ('Compact water-resistant indoor and outdoor video camera built with low-light color night vision and audio.', '紧凑型防水室内外视频摄像头，内置微光夜视成像与双向语音。'),
        ('Weather-resistant outdoor bullet IP security camera equipped with adjustable zoom focus and infrared night illumination.', '防雨防尘户外枪式网络安防摄像头，支持可调变焦视野与红外夜视照明。'),
        ('Wired outdoor weatherproof surveillance camera featuring adjustable mounting angle and reliable night monitoring.', '有线户外防水监控摄像头，支架角度可自由调节且支持夜间清晰监视。'),
        ('Rechargeable wire-free outdoor security camera featuring smart motion detection alerts and real-time voice intercom.', '可充电真无线户外安防摄像头，具备智能移动提醒与实时语音对讲。'),
        ('Compact plug-in indoor security camera featuring smart human detection, two-way voice audio, and mobile viewing.', '插电式室内安防摄像头，具备智能人体识别、双向语音及手机实时查看。'),
        ('Wire-free solar outdoor pan-tilt security camera featuring weatherproof housing, night vision, and motion alerts.', '太阳能无线户外云台安防摄像头，兼具防雨机身、夜视与移动侦测提醒。'),
    ],
    '02_chair': [
        ('Heavy-duty reclining hydraulic barber chair featuring swivel rotation, padded seating, headrest, and sturdy base.', '重型可仰卧液压理发椅，配备灵活旋转、软包座垫、头枕和稳固底盘。'),
        ('Hydraulic hair salon barber chair with waterproof seat cushion, height adjustment, and smooth swivel rotation.', '液压美发沙龙理发椅，配防水坐垫、高度升降调节与顺滑全向旋转。'),
        ('Reclining barbershop chair with thick padded cushion, supportive armrests, and tilting backrest for shaving comfort.', '可后倾理发店专座椅，采用厚实软垫、支撑扶手和舒适剃须靠背角度。'),
        ('Commercial salon styling chair featuring cushioned backrest, hydraulic foot pump, footrest, and full swivel rotation.', '商用美发造型椅，具备软包靠背、脚踏液压泵、脚踏支撑和旋转功能。'),
        ('Retro hydraulic barber chair with breathable waterproof upholstery, resilient foam cushion, and locking rotation.', '复古造型液压理发椅，采用透气防水表皮、高弹泡棉坐垫和锁定旋转。'),
        ('Heavy-duty barbershop and tattoo styling chair featuring thickened padded backrest, comfortable armrests, and height lift.', '重型纹身与美发造型椅，配加厚软包靠背、舒适扶手及高度调节底座。'),
        ('Classic steel-frame barber styling chair offering full cushioned comfort, hydraulic lift, and timeless salon look.', '经典钢架理发店造型椅，全软包舒适座感并支持液压升降调节。'),
        ('Hydraulic salon reclining chair featuring easy-clean waterproof leather cushion, padded armrests, and footrest support.', '液压沙龙后仰躺椅，配有易清洁防水坐垫、软包扶手与舒适脚踏。'),
        ('Boutique hair salon styling chair offering waterproof upholstered cushions, ergonomic backrest support, and hydraulic lift.', '精品发廊造型座椅，带防水易打理坐垫、人体工学靠背与液压高度调节。'),
        ('Reclining barber styling chair featuring dual control levers, memory foam cushion, and sturdy floor base.', '双侧拨杆可仰卧理发造型椅，配记忆海绵软垫与稳固支撑底座。'),
    ],
    '03_cookies': [
        ('Gluten-free refrigerated chocolate chunk cookie dough made with wholesome ingredients for warm soft homemade desserts.', '无麸质冷藏黑巧块曲奇面团，采用天然烘焙原料制成松软美味点心。'),
        ('Edible ready-to-eat chocolate chip cookie dough chunks packed conveniently for quick sweet snacking anytime.', '可直接食用的巧克力豆曲奇面团块，开袋即享便捷甜品零食。'),
        ('Pre-portioned frozen triple chocolate chip cookie dough ready to bake into soft bakery-style chocolate treats.', '冷冻分装三重巧克力豆曲奇面团，轻松烤出烘焙坊般浓郁软心曲奇。'),
        ('Eggless edible frozen cookie dough packaged for safe scooping straight out of the snack container.', '无蛋可生食冷冻曲奇面团，支持开盖用勺直接挖取享用安全美味。'),
        ('Peanut butter cookie dough blended with rich chocolate peanut butter candy pieces in convenient portions.', '混入浓郁花生酱糖果碎块的分装花生酱曲奇烘焙面团。'),
        ('Handmade ready-to-bake butterscotch chip cookie dough blended with butter, eggs, chocolate chips, and crunchy nuts.', '手工现烤奶油糖果豆曲奇面团，融合黄油、鲜蛋、巧克力碎和坚果。'),
        ('Oversized peanut butter cookie dough loaded with chopped peanut butter candy pieces and sweet chips.', '大份量花生酱曲奇面团，拌入浓郁花生酱夹心碎与香甜烘焙豆。'),
        ('Gourmet semi-sweet chocolate chunk cookie dough blended with butter and molasses for soft chewy cookies.', '精选半甜黑巧块曲奇面团，揉合黄油与糖蜜烤出松软有嚼劲曲奇。'),
        ('Pre-portioned classic chocolate chip cookie dough rounds ready for fast oven baking into crisp desserts.', '预分装经典巧克力豆曲奇圆面团，省去揉面快速烤出香酥曲奇。'),
        ('Classic chocolate chip cookie dough prepared with fresh butter, eggs, and rich chocolate morsels for baking.', '使用鲜黄油、鸡蛋与香浓巧克力豆调制的经典家庭烘焙曲奇面团。'),
    ],
    '04_curtain': [
        ('Light-filtering semi-sheer window curtains with a rod pocket top to brighten rooms and soften glare.', '穿杆式半透光窗帘，柔化室内刺眼强光并营造柔和优雅氛围。'),
        ('Wrinkle-free printed blackout grommet curtains designed to block sunlight, dampen outdoor noise, and insulate rooms.', '抗皱印花打孔遮光窗帘，有效遮挡强光、隔绝户外噪音并维持室温。'),
        ('Thermal insulated blackout window curtains designed for effective room darkening, noise reduction, and home comfort.', '隔热保温遮光窗帘，兼顾显著卧室暗化效果、降噪与居家隐私保护。'),
        ('Short microfiber blackout window covering built with thermal insulation and easy slip-on rod pocket design.', '超细纤维短款遮光窗布，穿杆即挂并具备隔热遮阳效果。'),
        ('Decorative printed blackout window curtains with wash-resistant fabric and simple rod-pocket installation for kitchens.', '装饰印花可机洗遮光窗帘，穿杆挂装方便并美化厨房餐厅窗口。'),
        ('Wide triple-weave blackout sliding patio door curtain engineered to block harsh sunlight and divide spaces.', '加宽三层梭织遮光推拉门窗帘，高遮光率且可用作室内软隔断。'),
        ('Soft decorative window curtains offering bedroom privacy protection, bright style, and universal rod compatibility.', '柔软装饰型窗帘，兼具日常室内隐私遮蔽与多房间通用杆适配。'),
        ('Lightweight semi-sheer printed window curtains allowing gentle daylight filtering and comfortable indoor airflow.', '轻盈半透印花窗帘，在温和滤光同时保证室内空气流畅交换。'),
        ('Thick soft blackout window curtains tailored with rod pockets for rich drape, privacy, and insulation.', '厚实柔顺穿杆式遮光窗帘，垂坠感佳并兼顾室内隐私与隔热。'),
        ('Heavy-duty thermal blackout window curtains compatible with hanging rings to reduce sunlight glare and heat.', '耐用加厚隔热遮光窗帘，适配挂环微调高度并有效隔绝日光热能。'),
    ],
    '05_hair_extension': [
        ('Lightweight handmade braided crochet synthetic hair strands featuring pre-looped ends for quick protective styling.', '轻盈手工钩针编织化纤麻花接发，预留发圈设计方便快速完成护发造型。'),
        ('Natural coiled curly clip-in hair extension set designed to blend seamlessly with textured curly hair.', '自然紧密卷曲卡扣接发套装，贴合天生卷发发质轻松增加发量。'),
        ('Soft water-wave synthetic crochet twist braiding hair suitable for lightweight daily and party hairstyles.', '柔顺水波纹合成钩针扭辫接发，轻便百搭适合日常及聚会造型。'),
        ('Double-wefted curly real human hair clip-in extensions ready for heat styling, dyeing, and added length.', '双层缝制卷曲真发卡扣接发，可自由烫染造型并瞬间增加长度发量。'),
        ('Colorful curly clip-in synthetic hair strands made from heat-resistant fiber for instant bright styling.', '耐高温化纤彩色卷曲挑染接发片，无需漂染即可轻松变换明艳发色。'),
        ('Natural unprocessed straight real human weave hair bundles adding thick fullness, shine, and smooth length.', '天然未处理直顺真发发束，大幅增添蓬松丰盈发量并保持光泽垂感。'),
        ('Double-wefted silky straight real human hair clip-in extensions engineered for seamless volume and natural blending.', '双层缝制顺直真人发卡扣接发，轻松隐形衔接并带来自然丰厚感。'),
        ('Natural straight real human hair weft bundle crafted with soft strands easy to dye and curl.', '顺滑自然直发真发发帘单束，发质柔亮可自由染烫造型。'),
        ('Real human hair clip-in piece designed to thicken thinning crowns and add versatile styling volume.', '真人发卡扣补发片，方便加厚头顶发量并打造多元蓬松发型。'),
        ('Silky straight real human hair clip-in extensions attaching quickly and tolerating everyday thermal heat styling.', '顺滑直发真发卡扣接发套片，快速佩挂且支持日常热工具卷烫造型。'),
    ],
    '06_jeans': [
        ('Stretch pull-on women straight-leg jeans designed with a tummy-shaping front panel and comfortable waist.', '女士高弹套穿直筒牛仔裤，前片塑形提臀配舒适腰部裁剪。'),
        ('Plus-size mid-rise wide-leg flared denim jeans cut for comfortable casual pairing with everyday tops.', '大码中腰阔腿微喇牛仔裤，穿着自在舒适并百搭日常上衣。'),
        ('Cotton mid-waisted bell-bottom denim jeans featuring comfortable relaxed mobility, practical pockets, and zipper closure.', '纯棉中腰宽松喇叭牛仔裤，活动无束缚且搭配实用口袋与拉链。'),
        ('Stretch performance climbing denim jeans engineered for flexible outdoor movement beneath athletic safety gear.', '高弹户外攀岩专用机能牛仔裤，兼顾耐磨防护与大跨步舒展自由度。'),
        ('High-waisted vintage loose bell-bottom denim pants crafted with stretchy breathable fabric for casual street outfits.', '复古高腰宽松微喇街头牛仔裤，弹性透气面料展现复古休闲风。'),
        ('High-waisted baggy straight-leg cargo denim jeans built with spacious side utility pockets and relaxed cut.', '高腰宽松直筒工装风牛仔裤，搭配实用大容量口袋与松弛剪裁。'),
        ('Relaxed high-waisted baggy cargo denim jeans featuring deep front flap pockets and casual straight legs.', '宽松高腰直筒工装牛仔长裤，前侧配有深大翻盖贴袋方便收纳随身小物。'),
        ('Cotton low-rise loose straight-leg street denim jeans designed for breathable year-round casual fashion outfits.', '纯棉低腰宽松直筒街头潮流牛仔裤，四季透气百搭各类休闲上衣。'),
        ('Low-rise stretch cotton bootcut denim jeans designed with five practical pockets and flattering leg cut.', '低腰棉弹五袋微喇牛仔裤，剪裁修长显瘦并兼顾经典细节。'),
        ('High-waisted slim straight-leg street denim jeans made with comfortable stretch tailoring for everyday casual style.', '高腰修身直筒潮流牛仔长裤，日常弹力舒适且展现利落街头版型。'),
    ],
    '07_shampoo': [
        ('Sulfate-free moisturizing botanical shampoo and conditioner set that gently cleanses scalps and restores hair shine.', '无硫酸盐植物保湿洗发护发套装，温和洁净头皮并重焕干枯发丝光泽。'),
        ('Gentle vitamin moisturizing shampoo formulated without harsh chemicals to protect smooth color-treated hair strands.', '不含刺激化学添加的维生素保湿洗发露，温和护色同时强韧顺滑发丝。'),
        ('Soothing moisturizing scalp shampoo formulated to relieve dry itchy flakiness while gently cleansing hair.', '针对干痒头屑研制的舒缓滋润洗发水，洁净同时减少头皮不适与发丝摩擦。'),
        ('Sulfate-free brightening toning shampoo enriched with botanical oils and proteins for curly and bleached hair.', '无硫酸盐亮泽护色植物油蛋白洗发水，滋养卷发并中和染后泛黄暗沉。'),
        ('Strengthening hair repair shampoo formulated to gently cleanse strands while reinforcing damaged inner hair fibers.', '强韧修护洗发露，在温和清洁同时深入修护受损发芯纤维改善毛躁。'),
        ('Salt- and sulfate-free gentle cleansing moisturizing shampoo engineered to preserve salon keratin hair smoothing treatments.', '无盐无硫酸盐温和保湿洁发水，有效保护沙龙角蛋白柔顺与染后发色。'),
        ('Vegan sulfate-free nourishing repair shampoo blended with natural plant oils to restore soft healthy hair.', '纯素无硫酸盐滋养修护洗发水，融合天然植物精粹恢复发丝柔润健康。'),
        ('Large salon color-protecting shampoo formulated to cleanse dyed hair gently while locking in luminous vibrancy.', '沙龙级大容量亮彩护色洗发露，温和洗护染后发质持久留住鲜活光泽。'),
        ('Paraben-free botanical nourishing shampoo formulated with coconut oil and herbal extracts for daily soft volume.', '无防腐剂椰油植物精粹营养洗发水，日常温和洗护令秀发轻盈丰盈。'),
        ('Fragrant all-in-one shower shampoo, conditioner, and body cleanser delivering refreshing lather for daily odor protection.', '香氛型洗护沐浴三合一全效洁净露，绵密泡沫洗去汗味保持全天清爽。'),
    ],
    '08_speaker': [
        ('Compact lightweight portable wireless speaker offering easy one-click music playback, hands-free calling, and FM radio.', '小巧轻便便携式无线蓝牙音箱，支持一键音乐播放、免提通话及调频收音。'),
        ('Portable indoor and outdoor wireless party speaker system featuring LED lighting and integrated microphone input.', '室内外便携无线派对音响系统，配炫彩灯光效果并支持接入麦克风。'),
        ('Wireless portable stereo speaker bundle combining rich indoor room audio with rugged waterproof outdoor sound.', '无线便携立体声音箱组合套装，兼顾室内澎湃音效与户外防水耐用体验。'),
        ('Portable wireless Bluetooth speaker designed with grippy rubber feet, speakerphone calling, and long battery life.', '防滑硅胶脚垫便携无线蓝牙音箱，支持横竖稳放、免提通话与长续航。'),
        ('Portable wireless stereo Bluetooth speaker supporting memory card music playback, FM tuner, and auxiliary input.', '便携立体声无线蓝牙音箱，全面兼容插卡直放、调频广播与有线输入。'),
        ('Ultra-compact cylindrical portable wireless speaker featuring quick automatic device pairing and convenient USB recharging.', '极简圆柱形便携迷你无线音箱，支持快速自动重连与方便USB充电。'),
        ('Weather-resistant indoor and outdoor wireless speaker combining warm ambient lighting, bass radiator, and wide range.', '防泼水室内外两用无线氛围音响，融合温暖灯光、低音振膜与广域无线连接。'),
        ('Pocket-sized wireless stereo portable speaker offering clear sound output and built-in microphone for calls.', '掌心尺寸便携立体声无线音箱，清晰音质表现并内置免提语音麦克风。'),
        ('Retro desktop wireless Bluetooth speaker offering rich stereo bass, classic radio tuner, and long battery playtime.', '复古桌面无线蓝牙音响，拥有饱满立体声低音、调频广播与持久电池续航。'),
        ('Waterproof outdoor portable Bluetooth speaker featuring colorful lights, radio tuner, memory card slot, and battery.', '防水户外便携蓝牙音箱，配动感彩灯、收音机调频与存储卡直插播放。'),
    ],
    '09_toothbrush': [
        ('Cute cartoon soft silicone children manual toothbrush designed for gentle gum cleaning and comfortable toddler grip.', '卡通趣味儿童手动软硅胶牙刷，呵护稚嫩牙龈并方便宝宝小手平稳握持。'),
        ('Ergonomic U-shaped manual children toothbrush made with soft silica bristles wrapping teeth safely without hurting gums.', '人体工学U形儿童手动牙刷，食品级软胶毛环绕洁齿并细心保护牙龈。'),
        ('U-shaped soft silicone kids manual toothbrush designed to clean front and back tooth surfaces gently.', 'U形全包覆儿童手动硅胶牙刷，温和清洁乳牙内外表面适合日常护齿。'),
        ('Lightweight waterproof handheld U-shaped kids silicone toothbrush encouraging healthy early oral hygiene habits.', '轻巧防水儿童手持U形硅胶训练牙刷，助力低龄宝宝养成自主刷牙好习惯。'),
        ('Whole-mouth U-shaped children silicone toothbrush featuring gum-massaging soft bristles and easy-grip cartoon handle.', '全口洁净儿童U形硅胶牙刷，柔软刷毛轻柔按摩牙龈并配有防滑卡通握柄。'),
        ('Waterproof cartoon electric kids toothbrush wrapped in soft silicone with dual gentle cleaning modes.', '防水卡通儿童声波电动牙刷，全硅胶柔软包裹机身且内置轻柔护龈模式。'),
        ('Eco-friendly cartoon baby manual toothbrush built with ultra-soft gentle bristles and small anti-slip grip.', '环保植物材质卡通宝宝手动小牙刷，配极细柔软刷毛与贴合小手的防滑柄。'),
        ('Reusable cartoon manual U-shaped children toothbrush built with compact heat-safe silicone bristles for toddler comfort.', '可爱卡通儿童手动U形牙刷，小巧耐高温硅胶刷头柔润贴合宝宝口腔。'),
        ('Manual cartoon U-shaped kids toothbrush with boil-safe soft silica head delivering gentle whole-mouth cleaning.', '卡通U形儿童手动牙刷，耐高温食品级硅胶刷头实现全方位轻柔清洁。'),
        ('Waterproof kids ultrasonic electric toothbrush featuring multiple gentle brushing modes, smart timer, and soft heads.', '全身防水儿童超声波电动牙刷，配备多档护龈模式、智能计时与柔软刷头。'),
    ],
}

# 54 cleaned model_paraphrase items aligned to oracle_v1_paraphrases (8-16 words, group mean 10-13)
ORACLE_V3_MP: dict[str, list[tuple[str, str]]] = {
    '01_camera': [
        ('Want a solar waterproof outdoor security camera offering motion alerts and talk audio.', '想要一款支持移动提醒与双向通话的太阳能户外防水安防摄像头。'),
        ('Need outdoor weatherproof security cameras featuring zoom focus and clear night vision.', '需要带变焦视野、防雨外壳和夜间清晰成像的户外安防摄像头。'),
        ('Want a smart outdoor floodlight security camera with panning, tracking, and clear audio.', '想买一款带云台追踪、移动警示灯和清晰语音的户外安防摄像头。'),
        ('Want a discreet portable security camera featuring loop night recording and steady battery power.', '想找一款支持循环录像与长续航夜视的小巧隐藏安防摄像头。'),
        ('Need an outdoor bullet security camera offering zoom focus and mobile motion notifications.', '需要带变焦镜头与手机远程移动侦测提醒的户外枪式安防摄像头。'),
        ('Want a wireless outdoor security camera with infrared vision, motion sensitivity, and weatherproofing.', '想要支持红外夜视、可调侦测灵敏度和耐候外壳的无线户外摄像头。'),
    ],
    '02_chair': [
        ('Want a compact salon barber chair offering locking swivel rotation and adjustable footrest.', '求购一款节省空间、可锁定旋转及脚踏高度可调的美发理发椅。'),
        ('Need a retro barbershop chair with waterproof cushions, height lift, and full rotation.', '想买带防水软垫、升降底座及旋转功能的复古风美发工作室理发椅。'),
        ('Want a salon barber chair offering tilting backrest, sponge cushion, and swivel base.', '需要带大角度后倾靠背、厚海绵垫与稳固旋转底座的发廊理发椅。'),
        ('Need a professional barbershop chair featuring hydraulic height adjustment, sturdy footrest, and swivel.', '寻找配备顺滑液压升降、耐用脚踏与旋转功能的专业理发工作室座椅。'),
        ('Want a salon barber chair built with ergonomic seat cushions and hydraulic elevation.', '想要带有舒适人体工学软包坐垫和稳固液压升降结构的沙龙理发椅。'),
        ('Need a classic waterproof barbershop styling chair featuring easy assembly and cushioned support.', '想选一款经典防水表皮、支撑舒适且组装省心的发廊造型理发椅。'),
    ],
    '03_cookies': [
        ('Want ready-to-bake chocolate chip cookie dough portioned cleanly for quick oven baking.', '想要预先分装好、随时直接放进烤箱烘烤的巧克力豆曲奇面团。'),
        ('Want gourmet chocolate chunk cookie dough made with butter, vanilla, molasses, and salt.', '想吃用黄油、香草、糖蜜与海盐调制的浓郁黑巧块曲奇烘焙面团。'),
        ('Seeking certified vegan frozen cookie dough chunks made for cold sweet dessert bowls.', '在找纯素认证、适合搭配冷饮甜点享用的冷冻曲奇生面团块。'),
        ('Need refrigerated chocolate chip cookie dough crafted without preservatives or artificial flavors.', '需要不含防腐剂与人工香精的天然配方冷藏巧克力豆曲奇面团。'),
        ('Where can shoppers buy effortless ready-to-bake sugar cookie dough without kitchen mixing?', '哪里能买到不用自己调配称重、省心干净即可开烤的香甜曲奇面团？'),
        ('Need pre-portioned frozen chocolate cookie dough baked easily into soft homemade desserts.', '需要分装规整的冷冻巧克力曲奇烘焙坯，方便在家烤制松软甜点。'),
    ],
    '04_curtain': [
        ('Want textured semi-sheer curtains that brighten rooms naturally while preserving daytime home privacy.', '寻找能提亮室内自然采光、透光又保持日常隐私的半透质感窗帘。'),
        ('Need soft draped curtains with rod pocket headers for cozy bedroom windows.', '想要悬垂柔美、穿杆式设计的优雅质感卧室遮挡装饰窗帘。'),
        ('Seeking lightweight semi-sheer window curtains filtering daylight softly while allowing room airflow.', '想找轻柔透光、保证空气流畅同时保护家居隐私的半透窗帘。'),
        ('Find decorative printed bedroom curtains suited cleanly for living rooms or dorms.', '推荐适合温馨卧室、客厅或宿舍布置的精致印花布艺装饰窗帘。'),
        ('Find compact thermal blackout window panels offering noise reduction and smooth woven texture.', '寻找具有良好隔热保温、减噪效果的紧密梭织遮光窗布。'),
        ('Find outdoor patio curtains made with weatherproof fabric and rust-resistant grommet tops.', '想买配备防锈金属圈、兼具防水与防晒涂层的现代户外露台窗帘。'),
    ],
    '05_hair_extension': [
        ('Want seamless straight real hair tape extensions providing fullness without shedding or tangling.', '想要无痕贴合、顺直不打结不掉发的真人发胶接片以丰盈发量。'),
        ('Want reusable heat-resistant synthetic hair extensions delivering soft bouncy human-like natural texture.', '想找质感接近真发、柔弹可耐热重复使用的手工安全化纤接发片。'),
        ('Need curly real hair clip extensions with gentle clips for comfortable daily wear.', '寻找佩戴不扯头皮、温和软夹固定的自然卷度真人发卡扣接发片。'),
        ('Want quick-attach straight real hair clip extensions blending naturally for daily outfits.', '在找日常通勤也能几分钟快速戴上且自然衔接的顺滑夹式接发。'),
        ('Seeking soft shiny weaving hair bundles offering elastic bounce and lasting wear.', '想选光泽弹润、少加工且耐用的优质真实发帘发束做造型编织。'),
        ('Want curly twist crochet braiding strands offering lightweight fullness for protective hairstyles.', '想要轻盈蓬松、易于做满头护发螺旋编织造型的柔卷钩针发束。'),
    ],
    '06_jeans': [
        ('Find stretchy flare women jeans crafted with comfortable pockets and breathable cotton weave.', '找一条透气棉弹、配经典口袋的全长修身微喇女士牛仔裤。'),
        ('Find casual comfortable low-waist pull-on street pants tailored for relaxed everyday outfits.', '寻一款日常穿脱自如的低腰宽松潮流风舒适街头休闲长裤。'),
        ('Find casual full-length wide-leg printed women street pants offering relaxed everyday comfort.', '想找剪裁舒适随性、全长阔腿印花设计的女士日常休闲街拍裤。'),
        ('Find high-waisted tapered stretch jeans built with deep pockets for daily styling.', '想要一条贴合身形的高腰微锥弹力女士口袋牛仔长裤。'),
        ('Seeking lightweight breathable stretch denim pants suited for daily travel and office wear.', '求推荐透气轻盈的弹力牛仔裤，适合通勤上班与长途旅行舒适穿着。'),
        ('Find plus-size mid-rise skinny women jeans offering gentle stretch and zipper closure.', '找一款合身显瘦、拉链门襟配舒适微弹面料的女士大码中腰小脚牛仔裤。'),
    ],
    '07_shampoo': [
        ('Find biodegradable conditioning shampoo bars formulated with fresh scents for family travel trips.', '寻找无塑料包装、便携环保的天然清香洗发护发皂，适合全家旅行。'),
        ('Find a vegan gentle shampoo set enriched with nourishing proteins and plant extracts.', '想要温和纯素不刺激、蕴含修护蛋白与植物精华的洗发护发套装。'),
        ('Find a gentle paraben-free daily haircare shampoo safe for regular family scalp cleansing.', '求推荐成分温和、无防腐剂添加的日常护发洁净洗发露。'),
        ('Find a natural oil-balancing vegan shampoo formulated gently for normal or oily hair.', '想选专为中性及偏油发质设计的天然温和草本控油洗发水。'),
        ('Find a sulfate-free hydrating shampoo replenishing moisture for dry fine brittle hair strands.', '需要轻盈无硫酸盐配方、能深层补水滋养细软干燥发质的保湿洗发水。'),
        ('Find a gentle sulfate-free shampoo protecting color vibrancy and keratin hair smoothing treatments.', '寻找无硫酸盐温和护色固发芯洗发水，延续染烫与柔顺护理效果。'),
    ],
    '08_speaker': [
        ('Want a slim portable Bluetooth speaker offering handle grip, stereo sound, and microphone.', '想买带便携握持、立体声音效与通话麦克风的超薄无线蓝牙音箱。'),
        ('Need a portable Bluetooth stereo speaker supporting card playback, radio, and AUX input.', '寻找兼容插卡、调频收音与有线输入的多功能便携立体声蓝牙音箱。'),
        ('Want a compact rechargeable portable mini wireless speaker with subtle mood lighting.', '需要自带变色氛围灯的可充电迷你紧凑型无线便携小音箱。'),
        ('Need a rugged waterproof outdoor Bluetooth speaker delivering deep bass and wide range.', '求购抗摔防水、低音强劲且连接距离远的重型户外无线音箱。'),
        ('Want a stylish miniature Bluetooth speaker supporting wireless stereo pairing and easy control.', '想要小巧精致、支持双机无线互联打造立体声效的迷你蓝牙音箱。'),
        ('Need a mini wireless speaker featuring party lights, radio tuner, and USB audio.', '想选带酷炫灯光、调频广播及USB播放的迷你无线音箱。'),
    ],
    '09_toothbrush': [
        ('Find ultra-fine soft bristle manual toothbrushes gentle enough for adults and young children.', '找适合全家成人和儿童使用、配超细柔护刷毛的手动护龈牙刷。'),
        ('Find a waterproof electric U-shaped toothbrush offering soft bristles and multiple gentle modes.', '求带多种清洁模式、全身防水与柔软刷毛的U形电动牙刷。'),
        ('Find gentle everyday manual toothbrushes built cleanly for safe family oral cleaning routines.', '寻找握感舒适、刷毛安全且适合全家人每日清洁口腔的常规牙刷。'),
        ('Find a gentle charcoal bristle toothbrush featuring tapered head and upright stand base.', '想要含竹炭软毛、锥形洁净刷头与可站立底座的护齿牙刷。'),
        ('Find soft silicone U-shaped kids toothbrushes sized comfortably for growing children.', '寻找按低幼与学龄儿童口腔尺寸区分设计的柔软U形硅胶牙刷。'),
        ('Find a waterproof sonic electric toothbrush featuring non-slip grip and replaceable brush heads.', '想买防滑防水机身并支持方便拆换柔软刷头的声波电动牙刷。'),
    ],
}

# Outcome-blind clean non-brand 2-6 word real heldout queries (exact substring in heldout_queries.jsonl AND training_bundle.json)
ORACLE_V3_HELDOUT_TEXTS: dict[str, list[tuple[str, str]]] = {
    '01_camera': [
        ('security cameras outdoor', '真实搜索原句（esci_train_91485）：户外安防摄像头'),
        ('indoor security camera', '真实搜索原句（esci_train_53095）：室内安防摄像头'),
        ('wireless home security camera system', '真实搜索原句（esci_train_112187）：家用无线安防摄像头系统'),
        ('security camera with pan tilt', '真实搜索原句（esci_train_91483）：带云台旋转的安防摄像头'),
    ],
    '02_chair': [
        ('chair hair salon', '真实搜索原句（msmarco_websearch_train_4796412）：美发沙龙椅'),
        ('barber chairs for sale', '真实搜索原句（msmarco_websearch_train_1643831）：待售理发椅'),
        ('hair salon chairs', '真实搜索原句（msmarco_websearch_train_8108306）：发廊椅'),
        ('salon styling chairs wholesale', '真实搜索原句（msmarco_websearch_train_5472019）：批发美发造型椅'),
    ],
    '03_cookies': [
        ('butter cookie dough', '真实搜索原句（orcas_5305269）：黄油曲奇面团'),
        ('frozen cookie dough', '真实搜索原句（orcas_10565019）：冷冻曲奇面团'),
        ('raw cookie dough', '真实搜索原句（orcas_4076126）：可生食曲奇面团'),
        ('chocolate chip cookie dough', '真实搜索原句（orcas_6365129）：巧克力豆曲奇面团'),
    ],
    '04_curtain': [
        ('cute curtains for bedroom', '真实搜索原句（esci_train_31005）：卧室可爱窗帘'),
        ('matching curtain sets', '真实搜索原句（esci_train_66301）：成套搭配窗帘'),
        ('kids curtain', '真实搜索原句（esci_train_57762）：儿童房窗帘'),
        ('fancy curtain', '真实搜索原句（esci_train_39554）：精致装饰窗帘'),
    ],
    '05_hair_extension': [
        ('real hair extensions', '真实搜索原句（orcas_7304189）：真人发接发'),
        ('best clip in hair extensions', '真实搜索原句（orcas_8006362）：好用的卡扣接发'),
        ('hair extensions real human hair', '真实搜索原句（orcas_11728311）：纯真人发接发'),
        ('clip in hair extensions', '真实搜索原句（orcas_9740362）：夹式接发'),
    ],
    '06_jeans': [
        ('plus size jeans', '真实搜索原句（esci_train_81516）：大码牛仔裤'),
        ("woman's jeans low rise", '真实搜索原句（esci_train_112416）：女士低腰牛仔裤'),
        ('ladies jeans', '真实搜索原句（esci_train_59568）：女式牛仔裤'),
        ('curvy jeans for women high waist', '真实搜索原句（esci_train_30892）：女士高腰修饰曲线牛仔裤'),
    ],
    '07_shampoo': [
        ('shampoo for oily hair women', '真实搜索原句（esci_train_92234）：女士油性发质洗发水'),
        ('shampoo for hair loss', '真实搜索原句（esci_train_92233）：防脱发洗发水'),
        ('tea tree shampoo and conditioner', '真实搜索原句（esci_train_101021）：茶树精油洗护套装'),
        ('shampoo and conditioner bars', '真实搜索原句（esci_train_92219）：固态洗护发皂'),
    ],
    '08_speaker': [
        ('bluetooth speaker loud', '真实搜索原句（esci_train_17632）：大音量蓝牙音箱'),
        ('best and loudest bluetooth speaker', '真实搜索原句（esci_train_15022）：音质响亮的蓝牙音箱'),
        ('small speakers bluetooth wireless', '真实搜索原句（esci_train_94913）：迷你无线蓝牙音箱'),
        ('waterproof speaker bluetooth', '真实搜索原句（esci_train_110179）：防水蓝牙音箱'),
    ],
    '09_toothbrush': [
        ('baby finger toothbrush', '真实搜索原句（esci_train_11919）：婴儿指套牙刷'),
        ('kids toothbrush', '真实搜索原句（esci_train_58043）：儿童牙刷'),
        ('eco friendly toothbrushes medium bristles', '真实搜索原句（esci_train_36637）：环保中毛牙刷'),
        ('life without plastic toothbrush', '真实搜索原句（esci_train_62154）：无塑料环保牙刷'),
    ],
}

# zero_overlap_v3 (20 per group): isomorphic 10/6/4 structure, zero literal/3-gram overlap with training_bundle & target_concept_text
# category_description: first 6 explicit (10-18w), last 4 implicit_anchor_free=True (10-18w), group mean 12-15w
ZERO_V3_DESC: dict[str, list[tuple[str, str, bool]]] = {
    '01_camera': [
        ('Compact indoor monitoring security camera enabling smartphone live streaming and local event storage.', '紧凑室内监测安防摄像头，支持手机端实时推流观看与本地视频回放记录。', False),
        ('Outdoor wireless property security camera delivering crisp dark recordings and smartphone motion notifications.', '户外无线住宅安防摄像头，带来清晰暗光影像与手机端动态消息推送。', False),
        ('Weather-resistant home doorway security camera providing wide field views and instant talk audio.', '防风雨家用门口安防摄像头，视野开阔且支持随时开启语音双向对话。', False),
        ('Plug-in nursery video monitoring camera enabling quiet room operation and smooth remote viewing.', '插电式婴儿房视频看护摄像头，运行静谧并在移动端稳定远程监看。', False),
        ('Driveway monitoring security camera capturing clear dark footage without requiring monthly subscription plans.', '无需订阅月费即可拍摄清楚暗光画面的车道监护安防摄像头。', False),
        ('Renter-friendly apartment entryway security camera supporting tool-free wall mounting and app clip playback.', '适租客使用的免打孔公寓门厅安防摄像头，手机可随时回看片段。', False),
        ('Front-door weatherproof delivery watcher capturing clear dusk video and sending phone motion alerts.', '门口防雨快递守护仪，能在黄昏微光下清晰录像并向手机推送提醒。', True),
        ('Smart backyard gate sentinel recording dim evening activity locally onto expandable flash cards.', '智能院门守护哨兵，可将昏暗晚间活动直接保存在机身扩展存储卡中。', True),
        ('Indoor entryway visual guardian letting families monitor pets and speak live from afar.', '室内门厅视觉看护装置，出门在外也能随时了解宠物近况并远程说话安抚。', True),
        ('Compact porch video monitor engineered for sunlit doorways needing clear balanced shadow exposure.', '适合门廊逆光入口的紧凑视频看护设备，平衡亮部阴影看清来访面孔。', True),
    ],
    '02_chair': [
        ('Swivel barber chair upholstered for simple hair-clipping cleanup between back-to-back client appointments.', '顺滑旋转理发椅，便于忙碌剪发客流间隙快速清理落发并维持整洁。', False),
        ('Cushioned stylist salon chair providing lumbar support through lengthy hair-coloring appointments.', '配护腰软包支撑的美发沙龙造型椅，久坐染烫护理依然自在惬意。', False),
        ('Space-saving boutique barbershop chair engineered cleanly to fit tight single booths in studios.', '为小型精品工位打造的高效紧凑款理发椅，节省走廊通道占地面积。', False),
        ('Commercial salon chair featuring stable swivel locking and smooth elevation for taller stylists.', '配稳定旋转锁定与大行程升降的商用美发师工位座椅，适配不同身高操作。', False),
        ('Adjustable salon shampoo and cutting chair constructed for steady balance on tiled studio floors.', '立足稳固防滑的洗剪吹多用可调美发椅，在光滑沙龙地砖上平稳安心。', False),
        ('Traditional grooming barber chair offering reclining comfort during hot-towel beard trimming sessions.', '兼顾经典造型与舒适仰卧角度的传统修面胡须护理理发专座椅。', False),
        ('Commercial hairstyling workstation seat equipped with easy-step elevation pump for smooth client height shifts.', '配备轻松脚踩液压泵的美发工位客座，安静平稳调整顾客坐姿高度。', True),
        ('Washable client grooming recliner built for quick daily wiping across studio tattoo and beard work.', '易拭洗表面材质打造的理容纹身修面工作室专用后仰躺座，日常省心。', True),
        ('Swivel studio seating unit featuring cushioned elbow rests and balanced support for long sessions.', '配有舒适扶手垫与稳妥支撑结构的工作室旋转座椅，缓解长时间等待疲劳。', True),
        ('Durable salon workstation lounger crafted with padded back support to ease neck strain gently.', '专注承托颈肩与腰背的厚垫工位工作躺座，让顾客护理体验更放松。', True),
    ],
    '03_cookies': [
        ('Sweet refrigerated chocolate cookie dough crafted for warm chewy homemade dessert treats anytime.', '随时烘烤成松软温热手工曲奇的甜香冷藏巧克力风味烘焙生面团。', False),
        ('Soft buttery cookie dough rounds portioned for easy oven baking during busy weekday nights.', '分粒规整的香酥黄油曲奇面团圆坯，工作日夜晚也能快速烤好享用。', False),
        ('Frozen bakery chocolate cookie dough baking evenly without drying edges on standard kitchen sheets.', '在普通家用烤盘上受热均匀、不干焦裂的冷冻巧克力风味烘焙曲奇坯。', False),
        ('Allergen-conscious gluten-free sweet cookie dough formulated cleanly without artificial coloring or harsh additives.', '不含人工色素及多余添加剂、满足无麸质轻食需求的甜美烘焙曲奇面团。', False),
        ('Warm festive homemade cookie dough suited cleanly for decorating sweet treats alongside young kids.', '适合周末亲子家庭一起动手塑形装饰节日甜蜜点心的松香曲奇生面团。', False),
        ('Quick snackable dessert cookie dough crafted with wholesome flour and clean sweet butter flavor.', '采用放心热熟面料调制醇正奶油风味、解馋又省事的小食曲奇甜面团。', False),
        ('Safe spoonable sweet dessert batter designed to swirl cleanly into homemade frozen ice cream sundaes.', '可直接用勺挖取拌入冰激凌圣代碗享用的安全无蛋甜品点心面糊。', True),
        ('Oven-ready buttery chocolate sweet dough pucks delivering crisp chewy bakery bites from home kitchens.', '无需揉面直接入炉、烤后外圈香酥内软的黄油可可风味分装烘焙小面饼。', True),
        ('Peanut-butter chocolate dessert dough pieces bringing rich bakery aroma into home kitchens in minutes.', '只需十几分钟即可在厨房飘满烘焙香气的花生酱可可块风味甜点面团。', True),
        ('Dairy-free sweet baking chunks formulated cleanly for guilt-free plant-based snacking straight from cold storage.', '冷藏取出即可畅享或烘烤、无乳制品添加的植物基醇甜点心块。', True),
    ],
    '04_curtain': [
        ('Room-darkening bedroom window curtains designed to shade daytime sleepers from warm afternoon glare cleanly.', '帮助夜班人群遮挡刺眼午后阳光、营造昏暗睡眠环境的卧室遮光窗帘。', False),
        ('Machine-washable living room window curtains built to keep crisp shape after routine home laundry.', '经过家庭常规水洗依然保持舒展垂顺、免打理的客厅布艺长窗帘。', False),
        ('Insulated privacy window curtains reducing warm summer sunlight transfer through large sunny apartment windows.', '减少夏季烈日热量渗入大型向阳窗户的隔热隐私防护型布艺窗帘。', False),
        ('Decorative fabric curtains styled to soften living room acoustics and frame cozy dining spaces.', '柔化室内空旷回声并装饰餐厅客厅落地窗框的质感垂坠软装布艺帘。', False),
        ('Lightweight bright window curtains filtering morning light evenly without making kitchens feel overly dim.', '均匀柔化晨间强光、同时保留厨房空间通透明亮感的轻盈家用窗帘。', False),
        ('Versatile patio entryway window curtains gliding smoothly to divide open apartments into quiet zones.', '滑动顺畅的阳台推拉玻璃门窗帘，兼顾开放式公寓起居睡眠区域柔和分隔。', False),
        ('Heavy insulating apartment window coverings designed to block streetlights and quiet noisy urban traffic.', '阻挡夜间街道路灯强光并削弱都市车流噪音的厚实隔热卧室布面罩帘。', True),
        ('Airy soft light-filtering window textiles maintaining daytime apartment privacy while welcoming gentle breezes.', '既保障白天居家生活隐私、又让轻柔微风顺畅透入室内的轻盈纱质挂帘布。', True),
        ('Wrinkle-resistant indoor patio fabric hangings built to shield hardwood flooring against sunny fading.', '有效阻隔紫外线防止室内木地板褪色的抗皱布艺玻璃门长幅遮光布。', True),
        ('Easy-slide washable kitchen window hangings designed for tidy light control near home sink counters.', '悬挂于厨房水槽旁便于随手拉合调光的耐洗耐脏短款小窗织物帘布。', True),
    ],
    '05_hair_extension': [
        ('Clip-in straight styling hair extensions providing quick everyday volume without leaving visible bulky seams.', '快速夹戴即可自然增厚日常发量、根部平整不易显露接缝的直顺真人接发。', False),
        ('Soft wavy natural hair extensions tolerating low-heat curling tools while keeping smooth lustrous shine.', '耐受低温卷发棒造型并保持柔和天然光泽质感的水波纹柔顺真人接发片。', False),
        ('Lightweight braiding crochet hair extensions offering soft scalp feel for comfortable full-head protective hairstyles.', '编织全头防护发式依然轻盈不拉扯头皮的柔软舒适款钩针麻花接发。', False),
        ('Heat-stylable synthetic wavy hair extensions built with natural matte finish for effortless party looks.', '无假滑反光的哑光质感耐高温合成长卷接发，轻松打造派对氛围造型。', False),
        ('Thick double-wefted virgin hair extensions holding lush fuller ends cleanly after gentle routine washing.', '多次温柔清洗仍保留发尾饱满厚度的双层织密款实发接发发束。', False),
        ('Easy snap-on clip hair extensions blending invisibly along fine hairlines in under five minutes.', '不到五分钟就能在细软发丝间完成隐形固发并融入发尾的快戴暗夹接发。', False),
        ('Soft silicone-grip clip-on strands adding instant crown fullness cleanly for thin fragile everyday locks.', '配备防扯柔护硅胶发夹的发顶蓬松增厚发束，温柔融入细软日常长发。', True),
        ('Smooth unprocessed virgin locks ready cleanly for gentle styling, dyeing, and seamless salon blending.', '可供安全染烫塑形的未受损原生真人直发发束，轻松融发不显突兀分层。', True),
        ('Pre-looped lightweight crochet twisting tresses created for quick summer protective braided styling anywhere.', '预备贴心套圈设计、适合夏季户外度假轻松编织防护发的轻柔螺旋发束。', True),
        ('Bouncy tightly coiled human locks matching dense curl textures without dry stiff shedding.', '贴合高弹自然紧密小卷纹理的柔软发条，不易干涩掉发且弹性持久。', True),
    ],
    '06_jeans': [
        ('Stretchable cotton women jeans constructed to retain firm knee recovery throughout long sitting shifts.', '长时间办公室伏案久坐后膝部依然不易鼓包的弹力抗变形女士舒适牛仔裤。', False),
        ('Soft lightweight travel denim jeans offering breathable comfort for walking all day outdoors.', '适合户外全天候步行旅游穿搭的轻量透气亲肤休闲女式牛仔裤。', False),
        ('Flattering high-rise wide jeans crafted with spacious front pockets large enough for phones.', '高腰阔腿修饰腿型且拥有加深实用前兜、方便放手机的百搭休闲牛仔裤。', False),
        ('Loose streetwear boyfriend jeans crafted with durable seam stitching that endures regular home laundering.', '日常洗衣机洗仍能保持线迹工整牢固的宽松慵懒男友风街头版型牛仔裤。', False),
        ('Tailored bootcut everyday jeans elongating leg lines effortlessly when styled over heeled ankle boots.', '搭短靴穿出门尤显双腿修长利落剪裁的经典日常女式微喇牛仔裤。', False),
        ('Contoured-waistband comfort denim jeans eliminating awkward back gapping cleanly when bending down walking.', '后腰曲线精准贴合、弯腰走动不留尴尬空隙的直筒舒适弹力修身牛仔裤。', False),
        ('Loose cargo street bottoms featuring spacious pocket storage and comfortable soft everyday cotton blend.', '拥有宽大工装收纳贴袋、质感软糯亲肤且百搭滑板鞋的宽松街头长裤。', True),
        ('High-waisted slimming pull-on straight bottoms crafted for clean waist support and all-day office comfort.', '穿脱极其便捷且带有收腹平顺腰线剪裁的通勤全天候直筒日常休闲长裤。', True),
        ('Relaxed wide-leg weekend streetwear bottoms tailored cleanly from breathable woven cloth for warm seasons.', '温和春秋天气穿不闷热、宽松廓形垂坠走路带风的周末逛街阔腿休闲长裤。', True),
        ('Stretchy outdoor climbing twill bottoms allowing full hip flexibility beneath athletic waist harnesses.', '穿着攀岩安全带时跨步大幅抬腿毫无束缚感的高弹性户外耐磨长裤。', True),
    ],
    '07_shampoo': [
        ('Gentle soothing hydrating shampoo formulated for dry sensitive scalps needing clean wash without tightness.', '专为干燥敏感头皮设计、温和净彻发丝且洗后绝不紧绷拔干的舒缓洗发水。', False),
        ('Clarifying balancing botanical shampoo effectively lifting heavy styling buildup while leaving natural soft shine.', '清透温和去除发胶造型残留油脂同时保留自然光泽感的植物平衡洗发水。', False),
        ('Lightweight volumizing botanical shampoo boosting soft crown lift during humid summer mornings cleanly.', '潮湿闷热天依然令头顶发根保持蓬松空气感不塌陷的清爽无硅洗发水。', False),
        ('Repairing keratin moisturizing shampoo smoothing frayed split ends cleanly on brittle over-processed hair.', '专门安抚受损干燥毛躁发尾分叉、顺滑强韧易断脆弱发丝的角蛋白修护洗发水。', False),
        ('Fragrance-free mild family shampoo soothing redness without irritating delicate kids skin or scalps.', '无香精低敏温和成分配方，全家大人及敏感肌肤儿童都能安心使用的洁发露。', False),
        ('Nourishing botanical oil shampoo helping coarse thick curls stay defined, soft, and frizz-free daily.', '深度滋润粗硬沙发与天生卷曲发质、锁住弹性顺滑减少毛躁打结的植物油洗发水。', False),
        ('Sulfate-free scalp cleanser lifting oily buildup gently while calming irritated sensitive flaky roots cleanly.', '不含硫酸盐温和去除多余皮脂并舒缓干燥发痒起屑发根的温和头皮洁净液。', True),
        ('Tone-correcting botanical hair wash designed to neutralize brassy warmth across lightened bleached locks.', '专用于漂浅发色护理、帮助洗去暗黄铜色调重现冷感透亮发丝光泽的洁发乳。', True),
        ('Salt-free conditioning liquid cleanser preserving smooth keratin straightening treatments smoothly for months.', '无氯化钠成分温柔洁净发丝、全力守住沙龙蛋白拉直柔顺时效的护色洁发露。', True),
        ('Refreshing all-in-one gym shower wash cleansing sweaty hair and tired skin cleanly after workouts.', '运动大汗后一步净爽清洗头皮发丝与疲惫肌肤的多效植物香氛沐浴洁发露。', True),
    ],
    '08_speaker': [
        ('Waterproof wireless acoustic Bluetooth speaker filling open backyard pool parties cleanly without distortion.', '足够撑起庭院泳池聚会气氛、大声场播放破音很少的防水无线便携蓝牙音箱。', False),
        ('Compact travel Bluetooth speaker offering reliable long wireless distance across open home patios.', '在院落阳台间走动仍保持稳健远距离无线连接的便携小巧旅行蓝牙音箱。', False),
        ('Dual-driver travel Bluetooth speaker engineered with acoustic bass radiators for rich stereo audio separation.', '通过双发声单元与低音辐射振膜提供层次分明立体声效的便携蓝牙音箱。', False),
        ('Rugged trail Bluetooth speaker featuring protective outer body housing across long mountain camping weekends.', '配有抗震防护外壳、无惧磕碰灰尘且陪伴长途山野露营周末的户外蓝牙音箱。', False),
        ('Compact desktop Bluetooth speaker offering quick multipoint pairing cleanly across laptops and phones.', '办公桌上可在电脑会议与手机音乐间秒速双设备无缝切换的无线蓝牙音箱。', False),
        ('Splashproof outdoor acoustic Bluetooth speaker supporting fast USB recharging during long road travel trips.', '防泼水耐用且支持高速充电、自驾长途旅行随手携带畅听的便携蓝牙音箱。', False),
        ('Trail portable wireless stereo music player built for rainy picnics alongside clear voice calling.', '徒步淋雨野餐都不怕且通话收音非常清晰的轻便无线立体声随身播放器。', True),
        ('Compact cylindrical patio audio unit delivering punchy outdoor rhythm alongside ambient lantern glow.', '自带温馨氛围夜灯与浑厚户外低频节奏感的紧凑圆柱形庭院音频小钢炮。', True),
        ('Retro desktop wireless acoustic box featuring classic radio dials and rich warm home room playback.', '拥有经典调频收音旋钮与醇厚人声听感的复古家居桌面无线音频主机。', True),
        ('High-output outdoor karaoke sound tower bundling lively lights with handheld microphone singing inputs.', '配动感派对彩灯与手持人声话筒插口的户外庭院移动式欢唱功放系统。', True),
    ],
    '09_toothbrush': [
        ('Ultra-soft bristle manual toothbrushes protecting tender swollen gums cleanly during gentle daily brushing.', '极细柔软刷毛细心呵护敏感红肿牙龈、温柔带走日常口腔污渍的手动护龈牙刷。', False),
        ('Quiet rechargeable motor toothbrush built cleanly with smooth gentle vibration and ventilated cap.', '马达安静低噪且贴心附带防尘旅行护帽的可充电日常家用声波电动牙刷。', False),
        ('Compact toddler training toothbrush engineered with chubby easy-grip non-slip handle fitting tiny fingers.', '圆胖防滑握柄方便低龄宝宝一手紧抓练习刷牙的入门训练安全小牙刷。', False),
        ('U-shaped soft silicone kids toothbrush gently wrapping baby molars clean without awkward gagging.', '柔软硅胶材质温柔包覆幼童后槽乳牙清洁、减少干呕不适的儿童U形牙刷。', False),
        ('Kids rechargeable electric toothbrush equipped with playful cleaning timer encouraging healthy morning habits cleanly.', '借助趣味定时提醒功能帮助宝贝培养早晚自主刷牙好习惯的儿童电动牙刷。', False),
        ('Fine-tapered manual toothbrush gently sweeping sticky food plaque along delicate pediatric gumlines daily.', '纤细尖毛轻刷幼童脆弱牙龈沟缝并温柔清除饮食软垢的儿童日常护齿牙刷。', False),
        ('Boil-safe food-grade soft silicone U-arch mouth cleaner cleaning top and bottom rows together gently.', '可直接沸水烫洗消毒且一次温和包覆上下整排乳牙洁齿的U形硅胶洁牙器。', True),
        ('Gentle ultrasonic pediatric oral wand offering soft vibration speeds tailored for sensitive growing gums.', '针对幼嫩发育期牙龈调校多档轻柔声波节奏的儿童舒适护齿清洁电动棒。', True),
        ('Natural plant-fiber kids dental care brush fitted with cushioned bristle heads safe for first teeth.', '选用自然植物纤维环保柄配高柔护齿小圆头保护宝宝初萌乳牙的儿童洁牙具。', True),
        ('Cute cartoon shaped infant gum cleaner motivating little kids happily toward clean bedtime oral routines.', '凭借童趣可爱动物造型引导宝宝晚上开心配合清洁牙齿的硅胶口腔护理器。', True),
    ],
}

# zero_overlap_v3 model_paraphrase: 2 explicit + 4 implicit_anchor_free=True, 8-16w each, group mean 10-13w
ZERO_V3_MP: dict[str, list[tuple[str, str, bool]]] = {
    '01_camera': [
        ('Can you suggest compact indoor monitoring cameras with quiet motor panning?', '能否推荐一款云台转动静音无声的室内小型安防监控摄像头？', False),
        ('Which weatherproof outdoor security cameras send porch alerts without false leaf alarms?', '哪款户外防水安防摄像头推送信赖门口动静提醒且不易因树叶晃动误报？', False),
        ('What wire-free doorway watcher lets homeowners check evening deliveries live via phones?', '有什么免接电线的门厅守望设备能让我在手机上看清晚间快递动静？', True),
        ('Seeking outdoor entry monitors capable of brightening dim backyard gates cleanly after dusk.', '想要一台能在光线昏暗时提亮后院门廊画面的户外门禁监控仪。', True),
        ('Which compact living room pet watcher allows clear real-time two-way talking remotely?', '哪款放在客厅小巧的看宠设备支持画质顺畅且清晰的远程语音对话安抚？', True),
        ('Help select durable outdoor perimeter guardians supporting safe local card video recording.', '帮我选一只不怕刮风下雨、可直接把视频存进本地插卡里的院落守护器。', True),
    ],
    '02_chair': [
        ('Which reclining barber chair offers steadier steel frame support across busy shifts?', '哪款可后仰理发椅在全天客流高峰时拥有更稳固扎实的钢架承托力？', False),
        ('What compact hydraulic salon seating chair moves smoothly upward without squeaking noise?', '什么紧凑型液压沙龙美发椅在踩踏调高降落时安静丝滑不发出异响？', False),
        ('Could you recommend easy-wipe vinyl hairstyling workstation seating with sturdy footrest support?', '能否推荐一款配结实踏板且一擦即净皮面的沙龙美发工位专用客座？', True),
        ('Which commercial grooming recliner feels cushioned enough through lengthy haircoloring appointments cleanly?', '哪款商用理容后仰工作躺座的泡棉垫厚实舒适，久坐染发剃须不累？', True),
        ('Want a steady swivel client station seat fitting neatly inside tight booths.', '想找一款不占地、能轻松摆进狭小单人理容间里的稳妥旋转工作客座。', True),
        ('What hydraulic studio lounger rotates cleanly and locks securely at any client angle?', '有什么沙龙液压造型躺座能在任意角度顺滑转动并牢牢锁死位置？', True),
    ],
    '03_cookies': [
        ('Which frozen chocolate cookie dough keeps thick soft centers instead of spreading flat?', '哪款冷冻巧克力曲奇生面团放进烤箱后能鼓起松软芯体而不会摊成薄饼？', False),
        ('Where find wholesome home refrigerated dough safe cleanly for weekend baking nights?', '有没有配料干净健康、适合周末一家人现烤吃的冷藏风味曲奇生面团？', False),
        ('What egg-free sweet baking chunks let hungry children enjoy safe treats with spoons?', '哪款不含生鸡蛋的甜品生面团能让孩子们放心直接从冰盒里挖勺当零食吃？', True),
        ('Want rich butter dessert rounds baking crisp tender edges inside home ovens.', '想要十多分钟内就能烤出外脆里嫩边缘的醇香黄油分装甜点生面饼。', True),
        ('Which gluten-free plant-based baking batter tastes delicious without leaving chalky artificial aftertaste?', '哪种无麸质植物基甜品面团烤完口感依然醇正香甜且无怪异人工回味？', True),
        ('Help me find pre-portioned peanut dessert dough convenient for small solo evening snacks.', '帮我找一款提前分成小块、晚上独自想吃点心时随手烤几块的花生味甜面团。', True),
    ],
    '04_curtain': [
        ('Which dense insulated blackout curtains soften loud urban street traffic best overnight?', '哪款三层梭织遮光窗帘在深夜时分削弱马路车辆喧嚣噪声效果最为明显？', False),
        ('Could you find wrinkle-resistant living room curtains draping neatly straight from laundry?', '能否帮找几款洗衣机脱水拿出来直接挂上就平顺有型的客厅长窗帘？', False),
        ('What dense light-blocking window hangings keep bright morning sunshine out of bedrooms completely?', '哪类高密度挡光窗户布帘能把刺眼朝东阳光彻底隔绝在睡眠房间之外？', True),
        ('I want airy light-diffusing window textiles softening glare across bright work desks.', '需要能柔化强烈反光保护电脑工作视线、又清爽透气的半透薄纱窗织物。', True),
        ('Which thermal sliding patio glass coverings stop winter drafts without looking bulky?', '哪些推拉门隔热遮蔽布料能有效挡住冬日缝隙冷风且视觉上清爽俐落？', True),
        ('Help me pick washable short kitchen window hangings for cozy sink privacy.', '帮我挑一款清洗打理省力、挂在水槽窗台保护平时隐私的耐脏短款织物挂布。', True),
    ],
    '05_hair_extension': [
        ('Which natural virgin hair extensions blend cleanly against thin roots without heavy pulling?', '哪款真人发接发能让发根薄弱处隐形衔接同时不给头皮带来沉重垂坠感？', False),
        ('Can clip-on wavy hair extensions tolerate curling tools while keeping soft natural shine?', '夹式直顺接发能否耐住卷发棒热造型且依然保持柔润亮泽感？', False),
        ('What pre-looped lightweight braiding tresses feel softest on sensitive scalps during summer?', '哪种预先套圈设计的编发发束在炎热夏天满头编制时触感最轻柔护头皮？', True),
        ('Want double-wefted virgin locks thick enough at tips creating fuller everyday styling.', '想买从根部到发梢都饱满厚实的双层缝制真人顺发束做日常丰盈造型。', True),
        ('Which invisible quick snap-on strands attach cleanly within minutes before morning work commutes?', '哪类快速暗夹接发束能在早上出门赶通勤前几分钟稳当夹上还不露出夹扣？', True),
        ('Help me find heat-friendly synthetic curly highlight strands adding colorful party flair safely.', '帮我找一款不用漂染伤本发、直接用发夹瞬间点缀亮丽卷曲派对挑染色彩的发束。', True),
    ],
    '06_jeans': [
        ('Which stretch cotton women jeans prevent awkward waist gapping when sitting or bending?', '哪款女士棉弹牛仔裤在坐下或弯腰取物时能紧贴后腰不露出空档尴尬？', False),
        ('Can lightweight relaxed denim jeans keep comfortable breathability during humid walking trips?', '轻薄款宽松阔腿牛仔裤在湿热气候旅游走路时会不会足够透气舒适？', False),
        ('What high-waisted stretchy straight street bottoms pair cleanly with boots and tucked sweaters?', '有没有高腰弹力直筒街头风日常长裤，塞毛衣搭短靴都十分干练利落？', True),
        ('Want deep cargo flap-pocket street bottoms able cleanly holding daily phones upright.', '想找带有扎实工装翻盖大容量侧袋、装上重手机也依然垂挺不垮的休闲长裤。', True),
        ('Which breathable outdoor climbing stretch twill bottoms flex cleanly beneath tight waist harnesses?', '哪些户外攀岩高弹耐磨长裤在贴身安全带束缚下还能毫无阻力自由屈膝伸腿？', True),
        ('Help me choose soft vintage relaxed streetwear bottoms retaining crisp shape after laundry.', '帮我选一条洗水自然、水洗多次后仍然维持良好版型挺括感的复古宽松街潮裤。', True),
    ],
    '07_shampoo': [
        ('Which mild botanical shampoo soothes dry flaky roots without stripping natural strand shine?', '哪款温和植物配方的洗发水在安抚头屑干痒头皮同时还不洗涩天然毛鳞片光泽？', False),
        ('Could you suggest mild sulfate-free cleansers protecting fresh dyed color tones longer?', '能否推荐一瓶无硫酸盐温和配方、帮助刚染好的鲜亮发色锁色更久的洗发露？', False),
        ('What lightweight clarifying scalp wash lifts oily sweat buildup cleanly after morning workouts?', '有什么质地清爽的温和头皮洗洁液能洗尽晨跑汗渍油腻、不令头顶扁塌油腻？', True),
        ('Want mild fragrance-free family cleanser soothing infants and parents sensitive skin.', '想要无香精无泪低刺激的全家洁发沐浴好物，娇嫩宝宝和敏肌都能共享。', True),
        ('Which brass-toning clarifying hydrating hair wash keeps lightened curls soft and luminous?', '哪类能矫正去黄提亮的植物保湿洁发乳可以让漂浅卷发不干枯毛糙？', True),
        ('Help me find solid travel hair-cleansing bars foaming nicely without leaking inside luggage.', '帮我挑几块起泡丰富还不怕飞机行李箱洒漏的天然浓缩固态洗护旅行皂。', True),
    ],
    '08_speaker': [
        ('Which waterproof wireless outdoor audio speakers produce punchy bass across backyard pools?', '哪款防水便携蓝牙音箱在空旷户外泳池露台边播音乐依然拥有有力澎湃低音？', False),
        ('Can desktop Bluetooth speakers switch seamlessly between laptop video meetings and mobile music?', '桌上放着的无线蓝牙音箱能否顺滑在笔记本视频会议与手机切歌间自如切换？', False),
        ('What compact rugged outdoor music player handles rain splashes during weekend mountain camping?', '有没有小巧扛造又无惧下雨溅水的户外音乐播放机，方便带着山地露营周末听歌？', True),
        ('Want a pocket wireless audio unit offering crisp clear outdoor phone calling.', '想要一台刚好装进兜里、支持大音量且清晰免提通话拾音的便携无线音频小盒。', True),
        ('Which warm retro wooden radio player fills home rooms cleanly without harsh distortion?', '哪款复古木纹造型便携调频收音音频机音色温暖饱满、开大声也不发闷破音？', True),
        ('Help me choose an outdoor waterproof party audio tower with glowing LEDs.', '帮我选一只带有绚丽氛围灯光、续航持久且无惧夜晚庭院露水湿气的户外派对音响。', True),
    ],
    '09_toothbrush': [
        ('Which ultra-soft manual toothbrush removes gumline plaque gently without irritating bleeding gums?', '哪款极细超软毛手动牙刷能干净清走牙龈沟软垢，同时毫不扎痛脆弱出血牙龈？', False),
        ('Can playful curved silicone kids toothbrushes motivate toddlers toward thorough brushing habits?', '童趣软硅胶弧形儿童牙刷能否帮助抗拒刷牙的磨蹭小宝宝渐渐爱上刷满时间？', False),
        ('What quiet pediatric sonic oral wand cleans gently with helpful timer alerts?', '有什么震动静音轻柔、带有定时引导的儿童充电护齿声波棒？', True),
        ('Want an eco bamboo toddler grip brush featuring chubby easy-hold handle shapes.', '想要小巧防滑粗柄设计方便幼童小手握持的天然植物柄宝宝洁齿小刷。', True),
        ('Which boil-safe soft mouth cleaners reach back baby molars cleanly every morning?', '哪款能开水放心高温杀菌的食品级硅胶洁齿器可轻松贴合宝宝大牙洗净？', True),
        ('Help pick a quiet travel oral wand carrying clean ventilated storage cap.', '帮我选一只带有透气防尘护盖且低噪耐用的日常差旅便携洁齿清洁棒。', True),
    ],
}

# zero_overlap_v3 heldout_search_query: 4 per group (2-5w), 0 overlap and 0 3-gram overlap with training_bundle & target_concept_text
ZERO_V3_HELDOUT: dict[str, list[tuple[str, str]]] = {
    '01_camera': [
        ('porch monitoring security camera', '门廊看护安防摄像头'),
        ('driveway motion security camera', '车道移动侦测安防摄像头'),
        ('entryway night surveillance camera', '门厅夜视监控摄像头'),
        ('outdoor wireless monitoring camera', '户外无线看护摄像头'),
    ],
    '02_chair': [
        ('reclining swivel barber chair', '后仰旋转理发椅'),
        ('salon styling swivel chair', '沙龙造型旋转椅'),
        ('vintage swivel barber chair', '复古旋转理发椅'),
        ('barbershop styling reclining chair', '理发店后仰造型椅'),
    ],
    '03_cookies': [
        ('ready bake dessert dough', '直接进炉甜点面团'),
        ('homemade baking cookie batter', '家庭现烤曲奇面糊'),
        ('soft chew cookie dough', '松软有嚼劲曲奇面团'),
        ('eggless spoonable cookie dough', '无蛋即食勺挖曲奇面团'),
    ],
    '04_curtain': [
        ('room darkening blackout curtains', '卧室全暗遮光窗帘'),
        ('thermal insulated bedroom curtains', '隔热保暖卧室窗帘'),
        ('light filtering living room curtains', '柔和滤光客厅窗帘'),
        ('sliding door privacy curtains', '推拉玻璃门隐私窗帘'),
    ],
    '05_hair_extension': [
        ('seamless clip human hair', '无痕卡扣真人接发'),
        ('natural curly braiding hair', '自然卷度编发束'),
        ('heat friendly hair strands', '耐热造型接发片'),
        ('soft wavy crochet hair', '柔顺水波钩针接发'),
    ],
    '06_jeans': [
        ('relaxed fit street jeans', '宽松街头风牛仔裤'),
        ('stretch waist women jeans', '高弹舒适腰头女牛仔裤'),
        ('loose relaxed cotton jeans', '宽松舒适纯棉牛仔裤'),
        ('comfortable travel denim pants', '旅行舒适透气牛仔长裤'),
    ],
    '07_shampoo': [
        ('gentle moisturizing scalp shampoo', '温和保湿舒缓头皮洗发水'),
        ('sulfate free hair shampoo', '无硫酸盐护发洗发水'),
        ('botanical daily cleansing shampoo', '植物日常温和洁发露'),
        ('color safe repair shampoo', '护色修护滋养洗发水'),
    ],
    '08_speaker': [
        ('trail portable acoustic speaker', '户外便携无线音箱'),
        ('compact travel music speaker', '便携旅行音乐音箱'),
        ('heavy bass party speaker', '重低音派对音响'),
        ('waterproof beach Bluetooth speaker', '沙滩防水蓝牙音箱'),
    ],
    '09_toothbrush': [
        ('soft silicone kids toothbrush', '软硅胶儿童牙刷'),
        ('toddler training oral toothbrush', '幼童护齿训练牙刷'),
        ('ultra soft sensitive toothbrush', '超细软毛抗敏牙刷'),
        ('rechargeable sonic kids toothbrush', '充电儿童声波牙刷'),
    ],
}


def build_oracle_v3(base_groups: list[str], all_training_norms: set[str]) -> dict:
    o1_manifest = read_json(ORACLE_V1_ROOT / 'manifest.json')
    group_entries = {}
    for group in base_groups:
        src_q = read_json(ORACLE_V1_ROOT / group / 'queries.json')
        src_zh = read_json(ORACLE_V1_ROOT / group / 'queries.zh.json')
        desc_rows = {
            str(r['id']): r['text']
            for l in (REPO / f'data/public/visual_inception_inputs/catalog_v5/{group}/descriptions.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()
            for r in [json.loads(l)]
        }
        mp_rows = {
            str(r['id']): r['text']
            for l in (REPO / f'data/public/visual_inception_inputs_v5/{group}/paraphrases/model_paraphrases.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()
            for r in [json.loads(l)]
        }
        hq_rows = {
            str(r['text']): r['id']
            for l in (REPO / f'data/public/visual_inception_inputs_v5/{group}/heldout_queries.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()
            for r in [json.loads(l)]
        }
        hq_file = f'data/public/visual_inception_inputs_v5/{group}/heldout_queries.jsonl'

        records, zh_records, instructions = [], [], []
        cd_idx = mp_idx = hs_idx = 0
        for rec, zrec in zip(src_q['query_records'], src_zh['query_records']):
            rec2 = copy.deepcopy(rec)
            zrec2 = copy.deepcopy(zrec)
            rec2['id'] = rec['id'].replace('_oracle_', '_oraclev3_')
            zrec2['id'] = rec2['id']
            rec2['implicit_anchor_free'] = False
            fam = rec['query_family']
            if fam == 'category_description':
                en_t, zh_t = ORACLE_V3_DESC[group][cd_idx]
                raw_text = desc_rows[str(rec['source_row_id'])]
                rec2['instruction'] = en_t
                rec2['instruction_zh'] = zh_t
                rec2['distilled_from_raw_description'] = True
                rec2['raw_source_text_sha256'] = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()
                zrec2['instruction'] = zh_t
                cd_idx += 1
            elif fam == 'model_paraphrase':
                en_t, zh_t = ORACLE_V3_MP[group][mp_idx]
                raw_text = mp_rows[str(rec['source_row_id'])]
                rec2['instruction'] = en_t
                rec2['instruction_zh'] = zh_t
                rec2['distilled_from_raw_paraphrase'] = True
                rec2['raw_source_text_sha256'] = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()
                zrec2['instruction'] = zh_t
                mp_idx += 1
            else:
                en_t, zh_t = ORACLE_V3_HELDOUT_TEXTS[group][hs_idx]
                assert en_t in hq_rows, f'Missing heldout text {en_t!r} in {group}'
                rec2['instruction'] = en_t
                rec2['instruction_zh'] = zh_t
                rec2['source_row_id'] = hq_rows[en_t]
                rec2['source_file'] = hq_file
                zrec2['instruction'] = zh_t
                hs_idx += 1
            records.append(rec2)
            zh_records.append(zrec2)
            instructions.append(rec2['instruction'])

        instr_sha = _instructions_sha256(instructions)
        en_doc = copy.deepcopy(src_q)
        en_doc['evaluation_protocol'] = ORACLE_PROTOCOL_V3
        en_doc['instruction_policy'] = 'training_oracle_v3_denoised_aligned_20'
        en_doc['instructions'] = instructions
        en_doc['instructions_sha256'] = instr_sha
        en_doc['query_records'] = records
        zh_doc = copy.deepcopy(src_zh)
        zh_doc['query_records'] = zh_records

        gdir = ORACLE_V3_ROOT / group
        gdir.mkdir(parents=True, exist_ok=True)
        qp, zp = gdir / 'queries.json', gdir / 'queries.zh.json'
        freeze_json(qp, en_doc)
        freeze_json(zp, zh_doc)

        ent = copy.deepcopy(o1_manifest['groups'][group])
        ent['sha256'] = sha256_file(qp)
        ent['chinese_review_sha256'] = sha256_file(zp)
        ent['instructions_sha256'] = instr_sha
        group_entries[group] = ent

    m = copy.deepcopy(o1_manifest)
    m['protocol'] = ORACLE_PROTOCOL_V3
    m['instruction_policy'] = 'training_oracle_v3_denoised_aligned_20'
    m['v1_source_manifest_sha256'] = sha256_file(ORACLE_V1_ROOT / 'manifest.json')
    m['groups'] = group_entries
    freeze_json(ORACLE_V3_ROOT / 'manifest.json', m)
    return m


def build_zero_v3(base_groups: list[str], base_manifest: dict, o1_manifest: dict) -> dict:
    group_entries = {}
    for group in base_groups:
        o3_doc = read_json(ORACLE_V3_ROOT / group / 'queries.json')
        tb = read_json(REPO / f'data/public/visual_inception/{group}/training_bundle.json')
        btexts = [q['text'] for q in tb['queries']] + [tb.get('target_concept_text', '')]
        b3grams = set()
        for t in btexts:
            b3grams.update(_token_3grams(t))

        records, zh_records, instructions = [], [], []
        idx = 1
        for fam, items, zh_fam in (
            ('category_description', ZERO_V3_DESC[group], ZERO_V3_FAMILY_LABELS_ZH['category_description']),
            ('model_paraphrase', ZERO_V3_MP[group], ZERO_V3_FAMILY_LABELS_ZH['model_paraphrase']),
            ('heldout_search_query', [(t, z, False) for t, z in ZERO_V3_HELDOUT[group]], ZERO_V3_FAMILY_LABELS_ZH['heldout_search_query']),
        ):
            for en_t, zh_t, is_imp in items:
                hit_3 = _token_3grams(en_t) & b3grams
                if hit_3:
                    raise ValueError(f'3-gram leak in zero_overlap_v3/{group}/{fam}: {en_t!r} -> {hit_3}')
                rid = f'{group}_zerov3_{idx:02d}'
                idx += 1
                records.append(dict(
                    id=rid,
                    query_family=fam,
                    instruction=en_t,
                    instruction_zh=zh_t,
                    implicit_anchor_free=is_imp,
                ))
                zh_records.append(dict(
                    id=rid,
                    query_family=fam,
                    query_family_zh=zh_fam,
                    instruction=zh_t,
                ))
                instructions.append(en_t)

        instr_sha = _instructions_sha256(instructions)
        en_doc = dict(
            schema_version=2,
            status='ready',
            method='visual_inception',
            evaluation_protocol=PROTOCOL_V3,
            group=group,
            query_mode='text',
            target_public_id=o3_doc['target_public_id'],
            category=o3_doc['category'],
            source_type='zero_overlap_isomorphic_10_6_4_non_overlap',
            language='en',
            instruction_policy='zero_overlap_v3_isomorphic_10_6_4_20',
            instructions=instructions,
            instructions_sha256=instr_sha,
            query_records=records,
            query_family_counts=dict(ORACLE_FAMILY_COUNTS),
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
            category=o3_doc['category'],
            query_records=zh_records,
        )
        gdir = ZERO_V3_ROOT / group
        gdir.mkdir(parents=True, exist_ok=True)
        qp, zp = gdir / 'queries.json', gdir / 'queries.zh.json'
        freeze_json(qp, en_doc)
        freeze_json(zp, zh_doc)

        group_entries[group] = dict(
            path=f'{group}/queries.json',
            sha256=sha256_file(qp),
            chinese_review_path=f'{group}/queries.zh.json',
            chinese_review_sha256=sha256_file(zp),
            target_public_id=o3_doc['target_public_id'],
            category=o3_doc['category'],
            query_count=20,
            instructions_sha256=instr_sha,
            query_family_counts=dict(ORACLE_FAMILY_COUNTS),
            public_features_sha256=base_manifest['groups'][group].get('features_sha256'),
        )

    m = dict(
        schema_version=1,
        protocol=PROTOCOL_V3,
        method='visual_inception',
        query_mode='text',
        source_type='zero_overlap_isomorphic_10_6_4_non_overlap',
        language='en',
        instruction_policy='zero_overlap_v3_isomorphic_10_6_4_20',
        group_count=len(base_groups),
        queries_per_group=20,
        total_queries=len(base_groups) * 20,
        query_families=list(ORACLE_FAMILIES),
        queries_per_family=dict(ORACLE_FAMILY_COUNTS),
        is_real_search_log=False,
        optimization_may_read=False,
        formal_memory_contents_used_for_authoring=False,
        existing_formal_queries_used_for_authoring=False,
        outcome_selection_used=False,
        base_manifest_sha256=sha256_file(EVALUATION_BASE / 'manifest.json'),
        training_overlap_audit=copy.deepcopy(o1_manifest['training_overlap_audit']),
        groups=group_entries,
    )
    m['training_overlap_audit']['normalized_exact_overlap_count'] = 0
    m['training_overlap_audit']['expected_overlap_ratio'] = 0.0
    m['training_overlap_audit']['three_gram_overlap_count'] = 0
    freeze_json(ZERO_V3_ROOT / 'manifest.json', m)
    return m


def build_overlap_30_v3(base_groups: list[str], base_manifest: dict, o1_manifest: dict) -> dict:
    group_entries = {}
    for group in base_groups:
        o3_doc = read_json(ORACLE_V3_ROOT / group / 'queries.json')
        z3_doc = read_json(ZERO_V3_ROOT / group / 'queries.json')
        o3_recs = o3_doc['query_records']
        z3_recs = z3_doc['query_records']

        o_cd = [r for r in o3_recs if r['query_family'] == 'category_description'][:3]
        o_mp = [r for r in o3_recs if r['query_family'] == 'model_paraphrase'][:2]
        o_hs = [r for r in o3_recs if r['query_family'] == 'heldout_search_query'][:1]

        z_cd_exp = [r for r in z3_recs if r['query_family'] == 'category_description' and not r['implicit_anchor_free']][:4]
        z_cd_imp = [r for r in z3_recs if r['query_family'] == 'category_description' and r['implicit_anchor_free']][:3]
        z_mp_exp = [r for r in z3_recs if r['query_family'] == 'model_paraphrase' and not r['implicit_anchor_free']][:1]
        z_mp_imp = [r for r in z3_recs if r['query_family'] == 'model_paraphrase' and r['implicit_anchor_free']][:3]
        z_hs = [r for r in z3_recs if r['query_family'] == 'heldout_search_query'][:3]

        selected = (
            [('oracle_v3', r) for r in o_cd]
            + [('zero_overlap_v3', r) for r in (z_cd_exp + z_cd_imp)]
            + [('oracle_v3', r) for r in o_mp]
            + [('zero_overlap_v3', r) for r in (z_mp_exp + z_mp_imp)]
            + [('oracle_v3', r) for r in o_hs]
            + [('zero_overlap_v3', r) for r in z_hs]
        )
        assert len(selected) == 20

        records, zh_records, instructions = [], [], []
        for idx, (src_tag, r) in enumerate(selected, 1):
            rid = f'{group}_overlap30v3_{idx:02d}'
            rec = copy.deepcopy(r)
            rec['id'] = rid
            rec['overlap_source'] = src_tag
            if src_tag == 'oracle_v3':
                rec['reused_oracle_id'] = r['id']
            else:
                rec['reused_zero_id'] = r['id']
            records.append(rec)
            zh_records.append(dict(
                id=rid,
                query_family=r['query_family'],
                query_family_zh=OVERLAP_30_V3_FAMILY_LABELS_ZH[r['query_family']],
                instruction=r['instruction_zh'],
            ))
            instructions.append(r['instruction'])

        instr_sha = _instructions_sha256(instructions)
        en_doc = dict(
            schema_version=2,
            status='ready',
            method='visual_inception',
            evaluation_protocol=OVERLAP_30_PROTOCOL_V3,
            group=group,
            query_mode='text',
            target_public_id=o3_doc['target_public_id'],
            category=o3_doc['category'],
            source_type='strict_nested_30pct_oracle_v3_70pct_zero_v3',
            language='en',
            instruction_policy='category_query_overlap_30_v3_nested',
            instructions=instructions,
            instructions_sha256=instr_sha,
            query_records=records,
            query_family_counts=dict(ORACLE_FAMILY_COUNTS),
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
            category=o3_doc['category'],
            query_records=zh_records,
        )
        gdir = OVERLAP_V3_ROOT / group
        gdir.mkdir(parents=True, exist_ok=True)
        qp, zp = gdir / 'queries.json', gdir / 'queries.zh.json'
        freeze_json(qp, en_doc)
        freeze_json(zp, zh_doc)

        group_entries[group] = dict(
            path=f'{group}/queries.json',
            sha256=sha256_file(qp),
            chinese_review_path=f'{group}/queries.zh.json',
            chinese_review_sha256=sha256_file(zp),
            target_public_id=o3_doc['target_public_id'],
            category=o3_doc['category'],
            query_count=20,
            instructions_sha256=instr_sha,
            query_family_counts=dict(ORACLE_FAMILY_COUNTS),
            public_features_sha256=base_manifest['groups'][group].get('features_sha256'),
        )

    m = dict(
        schema_version=1,
        protocol=OVERLAP_30_PROTOCOL_V3,
        method='visual_inception',
        query_mode='text',
        source_type='strict_nested_30pct_oracle_v3_70pct_zero_v3',
        language='en',
        instruction_policy='category_query_overlap_30_v3_nested',
        group_count=len(base_groups),
        queries_per_group=20,
        total_queries=len(base_groups) * 20,
        query_families=list(ORACLE_FAMILIES),
        queries_per_family=dict(ORACLE_FAMILY_COUNTS),
        is_real_search_log=False,
        optimization_may_read=False,
        formal_memory_contents_used_for_authoring=False,
        existing_formal_queries_used_for_authoring=False,
        outcome_selection_used=False,
        base_manifest_sha256=sha256_file(EVALUATION_BASE / 'manifest.json'),
        training_oracle_manifest_sha256=sha256_file(ORACLE_V3_ROOT / 'manifest.json'),
        zero_overlap_manifest_sha256=sha256_file(ZERO_V3_ROOT / 'manifest.json'),
        training_overlap_audit=copy.deepcopy(o1_manifest['training_overlap_audit']),
        groups=group_entries,
    )
    m['training_overlap_audit']['normalized_exact_overlap_count'] = 54
    m['training_overlap_audit']['expected_overlap_ratio'] = 0.3
    freeze_json(OVERLAP_V3_ROOT / 'manifest.json', m)
    return m


def main() -> None:
    base_manifest = read_json(EVALUATION_BASE / 'manifest.json')
    base_groups = list(base_manifest['groups'])
    o1_manifest = read_json(ORACLE_V1_ROOT / 'manifest.json')
    snapshots = o1_manifest['training_overlap_audit']['training_source_snapshots']
    all_training_norms = set()
    for snap in snapshots:
        for line in (REPO / snap['path']).read_text(encoding='utf-8').splitlines():
            if line.strip():
                all_training_norms.add(normalize_query_text(json.loads(line)['text']))

    build_oracle_v3(base_groups, all_training_norms)
    build_zero_v3(base_groups, base_manifest, o1_manifest)
    build_overlap_30_v3(base_groups, base_manifest, o1_manifest)

    # Print length balance table across all 9 groups & verify all 9 suites
    print('group | oracle_v3(cd/mp/hs/all) | overlap_30_v3(all) | zero_v3(cd/mp/hs/all) | span')
    for g in base_groups:
        def stats(root):
            recs = read_json(root / g / 'queries.json')['query_records']
            cd = [len(r['instruction'].split()) for r in recs if r['query_family'] == 'category_description']
            mp = [len(r['instruction'].split()) for r in recs if r['query_family'] == 'model_paraphrase']
            hs = [len(r['instruction'].split()) for r in recs if r['query_family'] == 'heldout_search_query']
            all_w = cd + mp + hs
            return sum(cd)/len(cd), sum(mp)/len(mp), sum(hs)/len(hs), sum(all_w)/len(all_w)
        ocd, omp, ohs, oa = stats(ORACLE_V3_ROOT)
        _, _, _, ova = stats(OVERLAP_V3_ROOT)
        zcd, zmp, zhs, za = stats(ZERO_V3_ROOT)
        span = max(oa, ova, za) - min(oa, ova, za)
        print(f'{g} | {ocd:.1f}/{omp:.1f}/{ohs:.1f}/{oa:.2f} | {ova:.2f} | {zcd:.1f}/{zmp:.1f}/{zhs:.1f}/{za:.2f} | {span:.2f}')

    for suite in (
        'training_oracle', 'overlap_30', 'zero_overlap',
        'training_oracle_v2', 'overlap_30_v2', 'zero_overlap_v2',
        'training_oracle_v3', 'overlap_30_v3', 'zero_overlap_v3',
    ):
        res = validate_category_query_suite(query_suite=suite)
        print(f'[VALIDATED] {suite}: status={res["status"]}, protocol={res["protocol"]}, overlap={res["training_normalized_exact_overlap_count"]}')


if __name__ == '__main__':
    main()
