# 可选商城运行依赖

当前九组的冻结图片生成与完整共享记忆检索只使用项目内已冻结的图对、特点和评估输入，不需要启动商城或加载英文语言模型。重新从完整商品目录选图时需要恢复项目商品数据库，但仍不需要购物服务。只有明确启用最终购物时，才需要以下可选服务依赖。

旧实验实际使用 pyserini（文本搜索接口）第一点六版、spaCy（自然语言处理库）第三点八点十四版，以及 en_core_web_lg（大型英语语言模型）第三点八版。小型英语模型不能在缺失时自动替代，否则会改变既有购物目标及奖励语义。商城所用上游源码、许可证、数据恢复块及恢复工具位于本项目资源与源码目录，不从相邻旧项目加载。

服务器先按主部署说明准备项目环境，并安装 Java（二十一版或更新爪哇运行时）。之后在项目根目录运行：

```bash
bash tools/setup_webshop.sh
python -B -X utf8 -m src.webshop.bundle restore
python -B -X utf8 tools/check_webshop_runtime.py --require --output output/deployment/webshop_runtime.json
```

商城安装脚本使用现有项目解释器，不创建第二套不明环境。普通依赖优先使用中国大陆镜像；英文模型轮子默认来自发布方固定版本地址，可通过 AMBILAGE_SPACY_MODEL_WHEEL（英语模型轮子地址或本地文件路径）显式提供可用镜像或已经准备的轮子。权重安装在运行环境中，不随代码仓库同步。安装结束记录实际依赖版本，便于服务器清空后恢复。

JAVA_HOME（爪哇运行时目录）若已显式设置，必须包含真实可执行文件，检查工具不会在错误配置后悄悄使用另一套系统运行时。AMBILAGE_PYTHON（项目解释器路径）可以显式指定已准备的运行环境；默认使用项目内虚拟环境。

AMBILAGE_WEBSHOP_SPACY_MODEL（商城英语模型设置）默认为大型英语模型包名，也可以指向同版本模型的本地目录。目录的元数据必须声明英语、大型核心模型及第三点八版；相对模型目录按项目根目录解释，不依赖启动命令所在位置。

依赖检查只读取安装版本、模型目录元数据和运行时版本，不加载语言模型、不启动商城、不发送购物模型请求。只有明确创建商城服务时才加载语言模型；本地单元测试使用微型模拟依赖，不证明远程服务器或真实模型已经运行通过。

官方安装要求核对来源：[文本搜索接口第一点六版](https://pypi.org/project/pyserini/1.6.0/)、[搜索接口官方安装说明](https://github.com/castorini/pyserini/blob/master/docs/installation.md)、[大型英语模型固定版本](https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-3.8.0)。
