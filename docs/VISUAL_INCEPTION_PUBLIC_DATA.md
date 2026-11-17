# 视觉记忆植入公开训练资料

二〇二六年十一月十七日已实际准备公开图片、商品描述、稀宇模型改写和独立真实搜索查询。用户最终确认保留每组五十／三十／二十，共一百条查询；九类各十个商品合并成同一个九十商品代理库，全部组使用同一候选集合。本轮未执行真实大型代理模型的扰动优化、正式检索评估或购物。

## 原文明确内容与本项目约定

[原论文第3.2节](https://arxiv.org/html/2604.16966v1#S3.SS2)明确五成查询来自目标类别的商品描述，但未指定不同商品个数、每商品允许几条、是否有放回。不能将“五十个不同商品”或“同一商品五十种描述”写成作者要求。[附录第A.2节](https://arxiv.org/html/2604.16966v1#A1.SS2)仅补充总查询数一百、小批量十六。[附录第B.4节](https://arxiv.org/html/2604.16966v1#A2.SS4)未补充商品抽样单位。

本项目补足细节的约定为：优先从多个独立同类商品抽取不同的真实公开描述；八组各五十个商品、一商品一条。曲奇面团只有二十二个严格同类独立商品，其五十条描述使用真实商品要点、说明原句和完整说明，每商品实际最多三条，保留原字段、索引或字符偏移。没有将模型改写填入公开描述来源。

原文也未公开三成模型改写逐条对应哪批输入。本项目让改写引用本组冻结描述：新提示每次向服务提交冻结随机顺序的前三十条，逐条生成一条改写；公开描述的五十条均保留在检索查询池。曲奇面团先前已成功生成的版本使用五十条输入请求三十条输出，按原提示身份保留。真实来源文件分别记录提示版本和输入摘要，不覆盖旧身份。

## 已准备的资料

| 内容 | 数量及状态 |
|---|---|
| 共用代理记忆库 | 九类各十张，共九十张真实公开商品图片；九十个商品及解码像素均互异 |
| 公开描述 | 九组各五十条，共四百五十条 |
| 模型改写 | 九组各三十条，共二百七十条，由本机声明的稀宇第三代模型真实生成 |
| 独立真实搜索查询 | 八组各二十条；理发椅十二条，共一百七十二条 |
| 完整训练包 | 除理发椅外八组，分别有一百条查询和同一九十商品候选集合 |
| 理发椅训练包 | 真实查询缺八条，按已选数据不足即停止的规则不冻结、不训练 |

公开商品来自本项目完整目录的有界索引读取，候补商品全局共四百二十二个，均与整个公开实验快照的二百零七个排除商品编号不相交。候选顺序使用固定种子四十二，不使用正式询问、名次或优化表现挑商品；标题人工审阅只判断主体品类，排除配件和错误商品类型。下载按该固定顺序选取最先成功且图片不重复的十张，九组汇总后再次核对全部公开源图和目标图的解码像素。随机独立抽样降低选择偏差，但不能保证不存在过拟合。

目录原始摘要沿用已冻结目录迁移清单。本轮核对文件字节数、修改时间与公开数据库结构，实际导出子集全部重新计算摘要，未重新串行散列七吉字节完整目录。

搜索查询包括亚马逊公开购物搜索原始训练分区及微软公开真实网页搜索查询。微软网页搜索不是购物日志；已过滤食谱、操作问答、沙龙服务和非主体商品。理发椅另保存十二条严格英语查询及三源证据；不足二十，不能靠重复、合成查询或洗头椅等近邻类补足。详见[查询来源说明](../data/public/visual_inception_corpus/QUERY_SOURCES.md)。

## 文件位置

- 共用代理库：`data/public/visual_inception_inputs_v5/proxy_memory/proxy_memory_manifest.json`，同目录的 `candidates.jsonl`(逐行候选对象文件)和图片目录。
- 公开描述及出处：`data/public/visual_inception_inputs/catalog_v5/<组名>/descriptions.jsonl`。
- 真实改写及生成身份：`data/public/visual_inception_inputs_v5/<组名>/paraphrases/model_paraphrases.jsonl`、同目录的 `generation_provenance.json`(生成来源文件)。
- 独立真实查询：`data/public/visual_inception_inputs_v5/<组名>/heldout_queries.jsonl`。
- 原文逐行归属核验：`data/public/visual_inception_inputs_v5/query_original_text_verification.json`。
- 优化消费的冻结包：`data/public/visual_inception/<组名>/training_bundle.json`(训练包清单)。每个包包含完整的来源行及图片，便于独立搬迁与核验；各完整包中的九十商品集合一致。
- 总状态：`data/public/visual_inception_preparation_status.json`。

真实密钥仅从忽略的本地服务配置读入调用进程，调用结束后恢复临时环境变量，不进入公开资料。该模型不支持温度设置，请求不发送温度字段；来源文件记录服务实际默认温度。采用稀宇模型及公开代理库等是项目适配，不能标成作者精确检查点和原始查询数据的逐字节复现。

## 分阶段重用与校验

`tools/build_visual_inception_public_training.py`(公开训练资料编排工具)依次支持图片准备、共用库合并、改写和冻结，均可按已冻结来源恢复；来源、图对、模型或服务地址变化时拒绝复用旧结果。改写阶段是真实服务调用，只有需要生成的组才调用；冻结阶段只消费既有资料。

```powershell
python -B -X utf8 tools/build_visual_inception_public_training.py images
python -B -X utf8 tools/build_visual_inception_public_training.py proxy-memory
python -B -X utf8 tools/build_visual_inception_public_training.py paraphrases
python -B -X utf8 tools/build_visual_inception_public_training.py freeze
```

最后一条对九组执行时，理发椅因查询配额不足返回失败，其余八组可独立核验。不应在理发椅资料补齐前将九组完整训练包根目录用于全九组优化。补齐必须更换冻结输入版本并重新核对来源，不覆盖既有清单。
