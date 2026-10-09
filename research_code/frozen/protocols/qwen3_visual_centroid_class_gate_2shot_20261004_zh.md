# Qwen3 2-shot RICES 的冻结视觉类质心门控对照

目的：控制主方法额外使用训练标签构造类别原型、在 validation 上校准以及先选类别再类内检索这三项机制。这个对照保留完全相同的 RICES parent、类别门控、完整训练池类内 top-2、目标模型、prompt、固定 batch2 和严格答案解析；只将目标模型引出的因素状态证据换成现有冻结 CLIP ViT-g 的普通全图视觉特征。

范围预先固定 Qwen3-VL-4B BF16、DTD/Aircraft/CUB、2-shot、RICES 一个 parent。每数据集仅新方法 `rices+clip_centroid_class_gate`；既有 RICES 和 RICES+Decision 只作冻结对照，不重跑。若 CPU 计划显示与已审计原始推理存在完全一致的固定 batch 伙伴，才按完整 prompt/batch 合同复用；不能按单条预测或不同 batch 伙伴偷换。旧实验根和源码不改，新方法在独立根运行。

每类的视觉原型为该类所有训练图像的单位 CLIP image embedding 平均，再做 L2 归一化。查询与类原型的余弦为普通视觉类别分数。RICES parent 类别分数仍是该类所有训练示例与查询的最大 CLIP 余弦；两个分数分量分别在该查询的类别轴上标准化，再按 `alpha * parent + (1-alpha) * centroid` 选类。所选类别内仍用原 RICES 分数在**完整训练池该类全部图像**中取两个有序、不同示例。这不是先取 RICES 候选再重排，也不引入新的视觉模型。

每 dataset 对这一个精确变体独立执行既有 stratified 5-fold / seed42 和完整 validation 类别 accuracy 校准，alpha 网格与并列规则同主方法；不继承 Decision 的 alpha。校准写盘并冻结后才读取 test manifest 构造计划；test 不用于改原型、alpha、样本顺序、提示或纳入范围。目标模型仍为原 Qwen3 BF16、greedy512、官方普通2-shot类别提示、固定 batch2；非精确类别输出保持 invalid。

流程：先 CPU 预检原数据、CLIP sidecar、父方法完整 hash；独立训练/validation 校准、test 计划冻结，统计与已审计主表计划相同的有序 demo 对和新增固定 batch 需求。只有这一步通过，才在单 GPU 双共享锁下启动新目标推理。每完整数据集另用**独立审计脚本**重算冻结文件 hash、校准、计划、fixed-batch raw 伙伴、完整覆盖、accuracy/macro/invalid、相对 parent 和 Decision 的 paired rescue/harm、bootstrap 95% CI 与双侧 exact McNemar 原始 p。审计通过前不进论文总账。不报告 Holm。
