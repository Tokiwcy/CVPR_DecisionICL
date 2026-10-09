# Gemma4 12B NF4 主表实施合同（2026-09-26）

范围为五数据集DTD、Aircraft、CUB、Dogs、Flowers102，1-shot五方法25项，2-shot七方法35项。Random、CLIP RICES、classification-adapted SigLIP MUIER及两parent+Decision；2-shot另含CIRCLES及CIRCLES+Decision。不是额外Generic/来源消融。所有test仍为探索性。

目标模型固定google/gemma-4-12B-it，revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7，NF4 double quant/BF16 compute、SDPA、全GPU无offload、thinking关闭、max_soft_tokens280、目标固定batch2、greedy512。权重读取采用经直接字节核验的进程内pread，解决Windows mmap原生崩溃，不改权重或安装库。官方parse_response提取最终content，再严格类别精确匹配。

发现阶段只读train：每数据集seed42类别平衡12图，3批×4图×6视角=18请求。matched-decision提示沿用run_qwen3_decision_generic_20260920中的冻结定义，目标来源改为Gemma。每请求3072 tokens；至多一次固定格式重试，追加与现有mainline相同的“Your previous response was not valid for the required schema. Return a complete valid JSON object only.”。每次raw在解析前保存，继续采用原严格1–6组、每组2–6状态及lint合同；不人工补写applies_to，不跳过失败group，不按test选择因子。若两次失败则停止诊断。预检发现响应来自validation，仅用于负载/格式检查，不纳入bank。

原始图像特征仅在encoder、sample顺序、manifest SHA合同核验后复用。SigLIP frozen全图image-text证据不是spatial grounding。train-only完全去重及响应/语义联合去重按已有主线实现，生成该模型独立bank与train/validation证据。该阶段不读test manifest。

后续每dataset×shot×parent独立train/validation五折及完整校准，validation类别命中率为目标，类别聚合→类别选择→类内parent排序。CIRCLES保持1 IR+1 CIR及本模型的属性/反事实描述；不可引用Qwen描述冒充Gemma生成。校准后才冻结test计划，缓存绑定完整提示、图像SHA及顺序、模型/processor/精度与固定batch伙伴。bank相同的1/2shot可以共享发现及底层证据，但适用选择参数分别重学。无参数parent无需权重。

执行闸门：完整五数据集train/validation的baseline1/2shot、CIRCLES属性/描述/分类及四图discovery GPU预检通过后，才能启动train bank发现；主表runner、独立校准、test计划与独立审计仍须另行实施并验证。当前文档不意味着主表test已运行。每完整结果更新总账和主线，报告accuracy/macro、rescue/harm、bootstrap95%CI和exact McNemar原始p，不报告Holm。

2026-09-26 validation诊断后的机械格式兼容：缺失applies_to且其余必填group字段齐全、唯一额外键以applies_开头并含非空字符串时，仅将此键改名为applies_to，保留原值；歧义、缺值和空值仍报错。每次改名记录原始/规范化hash、group索引、旧新键和值；不增生成预算，最多一次原有格式重试。对应gemma12_discovery_format_20260926.py通过两份真实失败响应和六个负例CPU核验。目标输出先按首个EOS裁去batch传输padding，再使用官方解析，所有非精确类别内容原样算invalid；不因validation模型答错而调整检索或因子。
