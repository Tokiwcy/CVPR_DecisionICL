# Qwen2.5-VL-7B NF4 五数据集重跑协议

2026-09-13 用户取消Cars，授权改跑Qwen2.5-VL-7B，遵照此前明确选择4bit。Cars及SDPA修复预检停止，旧dense目录保留暂停，不再接续dense。

模型为本地Qwen2.5-VL-7B-Instruct，以bitsandbytes NF4 double quant量化，BF16 compute，device_map全部cuda:0，禁止CPU/disk offload；逐个Linear4bit验证uint8存储、NF4和nested状态。保存实际量化模块数和device_map。分类target batch=2，SDPA、left padding、min_pixels65536/max_pixels401408、官方普通/CIRCLES分组提示、greedy最多512token、类别名strict exact-match；不采用旧class-code单token协议。allocator94%、固定target组前empty_cache。模型改变后不得继承dense或Qwen3的因子、描述、参数或分类缓存。

顺序DTD、Aircraft、CUB、Dogs、Flowers102，Cars/Caltech不跑。每项1shot五方法、2shot七方法，共60结果。沿用已核验且冻结的原五数据集划分，逐图SHA和跨split隔离检查。Qwen3完整10块的input/model/source/plan/cache/complete/audit先验证；仅复用同encoder、相同图像/顺序且hash通过的SigLIP/CLIP图像特征。

每个数据集先分类validation预检：1shot普通、2shot普通、2shot CIRCLES，固定2、重复raw、尾1；另用最大面积train示例及validation查询测试两类2shot输入。四张最大面积validation图测试实际发现生成负载；其生成内容不用来选因子、不进入train bank。取得唯一GPU锁后才运行GPU步骤。通过运行预检不等于所有可能输入绝对不会OOM，正式失败即停并诊断，不静默改变batch/像素/精度。

新NF4模型在train平衡12图seed42独立发现，每请求4图、3批×6视角=18请求，每次最多6组、原固定一次schema重试。精确去重后用train视觉证据响应去重（semantic .90、response .98、4096对seed42），无人工语义筛选。SigLIP2-base float16 GPU batch64生成新bank对应证据；先冻结支持、train/validation证据，再计算test证据。whole-image图文分数不称空间grounding。

方法：Random、CLIP RICES、classification-adapted SigLIP MUIER、RICES+Atom类别等权raw、MUIER+Atom类别等权raw，2shot另加CIRCLES及CIRCLES+Atom（1 IR+1 CIR的2-demo适配）。全训练池直接检索。每dataset×shot×Atom parent针对本次最终bank和准确方法分别完整validation五折及全量重新学习所有适用参数，无参数baseline不添加权重。只对相同且验证过的NF4新合同允许复用自身缓存。

2shot生产描述前，用32张validation图（大面积/极端长宽比，不依据分类正确率）比较batch2/4/8，原两步属性/反事实caption提示；检查非空、数量、重复raw、正式split尾批大小，reserved<=13GiB、通过者最快5%内优先较小。选择凭据绑定2shot冻结来源，正式OOM不静默降batch。先完成描述再独立校准，不读取test结果来调方法。

入口 scripts/run_qwen25_nf4_five_dataset.py，后端scripts/qwen25_nf4_official_backend.py；新根artifacts/qwen25_nf4_five_dataset_20260913。按数据集先1shot后2shot串行。每块完整后核验计划/缓存/覆盖/accuracy/macro/paired rescue-harm，再更新独立累计表与主线记录。仅探索性结果。每小时监控此NF4队列，60项完整审计后暂停监控，不恢复Cars或dense。
