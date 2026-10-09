# Qwen3 决策因素类别对应负控（2-shot）

目的：检验主表增益是否依赖因素证据与类别的正确对应。此消融只改变 train 集构造的类别因素原型的类别对应；原 Qwen3 决策因素 bank、SigLIP2 图像证据、CLIP RICES parent、完整 train pool、2-shot 官方分类提示与目标模型全部不变。将按字母序排列的类别原型循环右移一位（无固定点），不更改 query 证据、train 图像或标签。该负控不是新的因素发现方法，也不把它称为跨模型 bank swap。

为控制时间，预先固定 DTD（纹理）、Aircraft（交通工具细粒度）、CUB（动物细粒度）三个数据集，且仅运行 2-shot RICES + rotated-factor 对照。与已有同数据集同 shot 的 RICES parent、matched Decision 和 Generic direct-sample 比较；不重跑这些历史方法。不根据 test 成绩扩充或筛选数据集。

每个数据集对该变体重新在 train/validation 上完成五折与全量 validation alpha/类别等权校准，不继承 matched Decision 的系数。允许 alpha=1 的 parent-only 端点。所有参数写盘并冻结后才构造 test demo 计划。test 不用于挑选轮换、alpha、提示或方法。计划使用完整 train pool；分类固定 batch=2、greedy、官方类别精确解析，不能把非精确类别输出补为正确。目标推理每个固定 batch 的两个伙伴一起持久化，恢复时只跳过两者都完整的 batch。

轮换是确定性的；模型推理也固定为 greedy。因此不进行整项多 seed 重复。已有主表和本轮均报告逐 query 配对 rescue/harm、bootstrap 95% CI 与双侧 exact McNemar 原始 p。若需要随机性敏感度，优先在 CPU validation 上检查多个预先固定的轮换，而不把它伪装成独立目标模型重复运行。本轮仅以事先固定的一位轮换作为目标负控。

完整结果必须经过独立脚本复算 complete/hash、三数据集计划和固定批次覆盖、accuracy/macro/invalid、与 parent/Decision/Generic 的配对统计，才进入论文消融表。
