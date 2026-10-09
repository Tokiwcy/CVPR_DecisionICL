# Qwen3 五数据集主表三 seed 完整目标重跑

用户于 2026-10-04 明确要求 Qwen3-VL-4B-Instruct BF16 的五数据集三 seed，并进一步明确“都重跑”。原主表 seed=42；新增 seed=17、73，固定于任何新增 test 推理之前。范围为 DTD、Aircraft、CUB、Dogs、Flowers102 的完整 test，1-shot 五方法、2-shot 七方法，每个新 seed 均为 60 个方法项。Caltech101 不在已审计五数据集主表范围。

随机性只作用于 Random 方法的 train 示例抽样、validation 分层五折。因素发现库、train/validation/test 划分、SigLIP 冻结证据、CLIP 图像/描述嵌入、目标模型、方法定义、提示、类别解析、alpha 网格及完整 validation 拟合保持原主表合同。每个新 seed 的 Decision 方法必须重新完成对应五折及完整 validation 拟合并冻结参数，test 不用于选参。固定计划和 greedy 512 下的其他方法可能与原 seed 使用相同有序示例，这不改变用户要求：**两个新 seed 的目标预测全部由本次 Qwen3 推理产生，不从旧 seed 或其他方法实验导入 target raw/cache/predictions**。同一 seed 内完全相同的模型输入上下文可在新运行内部合并一次，必须记录方法到查询的映射；不跨 seed 合并。

保持原主表的实际执行批次：DTD 1-shot 为原单条执行，其余九块为固定 batch2；2-shot CIRCLES 保持原属性/反事实描述与官方分组提示，允许两通道同 demo。train 和 validation 资源仅可在 SHA256 相等且来源审计通过时只读复用，test 描述/图像嵌入也只作为冻结输入资源复用；这不等同于复用目标预测。

隔离根 `artifacts/qwen3_main_three_seed_full_20261004`，seed17 与 seed73 分开。单 GPU 双共享锁串行，异常保存已有新推理缓存并先诊断。每完整块独立核验输入和源码哈希、seed 专属 calibration 和 test 计划、所有 query/方法覆盖、目标 raw 为本次新生成、严格解析、accuracy/macro/invalid、Decision 相对 parent 的配对 rescue/harm、bootstrap 95% CI 与双侧 exact McNemar 原始 p。seed42 采用既有独立审计结果；三 seed 汇总应分别列出原 seed 和新增 seed，均值与样本标准差仅用于同一合同下的方法项，并明确因素库固定的条件。
