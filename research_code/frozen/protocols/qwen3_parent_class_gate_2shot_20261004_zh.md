# Qwen3 2-shot RICES 去因素类别门控消融

用户授权范围固定为 DTD、FGVC-Aircraft、CUB-200 三个完整 test 集；目标模型 Qwen3-VL-4B-Instruct BF16、2-shot、官方普通分类提示、greedy 512、固定 batch 2。原 RICES 与 RICES+Decision 已审计结果仅作配对参照，不重跑。

本变体只删除 Decision 因素证据：用训练库中 RICES 相似度最高的图像所属类别选类，再在该类的**完整训练池**按原 RICES CLIP 分数稳定排序取两个不同示例。它保留了“先选类、再类内取示例”的结构；没有因素 bank、因素分数、类别原型或可学习系数。1-shot 与原 RICES 完全相同，故只做 2-shot。固定并列规则采用原官方 np.argsort 后反转 top-k；不按 test 结果改规则。

训练、validation 和 test 的原图 manifest 与 CLIP sidecar 均来自已独立审计的 Qwen3 主表。CPU 阶段先独立重算 validation 的无参数诊断及 test 检索计划，再以完整模型、提示、图像 SHA、批内两个 query 与 demo 的精确合同匹配已审计的视觉类质心 raw batch。只有完整 batch 伙伴完全相同才复用其字节；其余批次新推理。不复用单条预测或不同伙伴的输出。

每个数据集完成后独立脚本重放计划、核验冻结源码与数据 SHA、全部固定 raw batch 的合同/伙伴、严格类别解析、accuracy/macro/invalid、相对 RICES 和 RICES+Decision 的逐 query rescue/harm、图级 bootstrap 95% CI 与双侧 exact McNemar 原始 p。三个完整块均审计通过才更新论文实验总账。此实验不使用 test 标签选择参数，也不报告 Holm 校正。

并列分数沿用冻结官方实现的 np.argsort 后反转 top-k 规则。
