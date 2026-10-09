# Qwen3 4B Generic 逐样本检索：1/2-shot 执行协议

用户于 2026-09-21 修订 Generic visual 的用法：Generic 因素不建立类别 prototype，也不在类别内平均。每个 query 与完整冻结 train pool 中每个样本直接比较 Generic group-state 证据，各 group 等权平均；随后分别与 RICES、classification-adapted MUIER 的逐样本 parent score 融合并直接排序。2-shot CIRCLES+Generic 对 original 与 composed 两个逐样本通道分别融合后各取一个 demonstration。该实现不是候选重排，没有类别预测或类内二次检索。

范围固定为 Qwen3-VL-4B-Instruct BF16、DTD/Aircraft/CUB/Dogs/Flowers102。1-shot 只跑 RICES+Generic 与 MUIER+Generic；2-shot只跑 RICES+Generic、MUIER+Generic、CIRCLES+Generic，共25个新结果。已有 parent 与 Decision 仅引用，不生成新预测。

每个 dataset × shot × parent 独立用 train/validation 五折及完整 validation 选择融合 alpha；目标为所选 demonstration 的同类比例，平局偏向较大 parent 质量。test 不参与 alpha、因素、提示或方法选择。Generic bank、SigLIP2 frozen evidence、CLIP/CIRCLES descriptions、数据 manifest 与 Qwen 模型合同均按来源 hash 验证。目标推理保持官方分类提示、Qwen3 BF16 greedy、固定 batch=2；完整 prompt 和有序 demo 完全相同且来源也是 batch2 时可复用旧原始输出，其余上下文新推理。

完成后报告 target accuracy/macro、相对既有 parent 与 Decision 的 paired rescue/harm、bootstrap 95% CI 和双侧 exact McNemar 原始 p，不报告 Holm。随后恢复 VICL；VICL+Generic 也采用本协议的逐样本 Generic 定义，VICL 与 VICL+Decision 保持原冻结定义。
