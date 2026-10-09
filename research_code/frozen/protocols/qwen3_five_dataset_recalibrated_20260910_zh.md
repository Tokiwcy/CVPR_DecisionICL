# Qwen3-VL-4B 五数据集独立重校准重跑

2026-09-10 用户要求暂停旧任务，修复检查后重跑1-shot五方法、2-shot七方法，全部覆盖DTD、Caltech101、Aircraft、CUB、Dogs，每小时监控。

新隔离根：`artifacts/qwen3_five_dataset_recalibrated_20260910`。每个数据集先完成1-shot五方法，再完成2-shot七方法，之后进入下一数据集。旧结果、旧校准和旧队列hold保留。

五方法：Random、官方CLIP ViT-g RICES、classification-adapted SigLIP MUIER、CLIP RICES＋Atom类别等权raw、SigLIP MUIER＋Atom类别等权raw。2-shot增加CIRCLES和CIRCLES＋Atom类别等权raw。MUIER保留分类适配定义：image cosine加query image与candidate train label文字的cosine；不冒充官方question-memory MUIER。CIRCLES为官方结构的总2-demo预算适配（1 IR＋1 CIR），允许两通道同图，不声称论文32-demo复现。所有检索覆盖完整冻结train池，不进行RICES候选重排。

普通方法使用官方普通分类提示，CIRCLES使用官方反事实分组、属性、caption和重复query提示。模型Qwen3-VL-4B-Instruct BF16、SDPA、target batch1、greedy max_new_tokens512；直接类别名生成，strip/lower exact-match。旧class-code提示下的预测禁止复用。此前1-shot的SigLIP RICES改为与最新2-shot一致的官方CLIP RICES，因此是明确的新合同。

每个shot×数据集×Atom parent从冻结train evidence重建类别原型，完整validation分层五折及完整拟合独立重新选择alpha；不读取旧calibration或demo plan。沿用既定alpha网格和并列优先parent质量规则。alpha是parent质量，所有atom groups平分1-alpha。监督为validation类别预测正确率；同类demo检索为诊断，不称目标accuracy优化。raw分支不自动回退。Random/RICES/MUIER为无学习参数基线，不虚构权重。固定支持集、raw evidence及标准化与现有mainline保持一致。

缓存：复用逐文件核验的冻结SigLIP evidence/embedding、CLIP image/text embedding与属性/caption。未完成caption沿原dataset批次续跑：DTD/Caltech/Aircraft batch2，CUB/Dogs batch8；禁止OOM后静默改变batch。新校准后，仅当模型/processor/完整文本提示/有序demo及图像hash/生成设置完全一致时迁移旧official target预测；方法名称和新检索参数不进入预测语义，prompt上下文进入缓存键。每条迁移保留source key，源plan/protocol/source/cache均有hash凭据。坏JSON尾行、冲突重复键、合同漂移均停止，不自动吞掉。

启动前：CPU合同测试，五数据集输入/哈希/划分隔离核验，全部数据集1-shot/2-shot普通与反事实提示validation-only GPU smoke，旧进程退出及GPU空闲。每项计划冻结绑定全部本地源码、模型文件、数据输入、参数和资源哈希。每项完成独立重算目标accuracy/macro、paired rescue/harm、bootstrap CI和McNemar并核验覆盖；更新主线证据。五项都是已有探索性数据，不能升级为确认性结果；test不用于选参数、prompt、方法或批次。异常停止保留缓存，监控先诊断，hash/协议失败不得盲目恢复。
