# 新增 HNS training-seed checkpoint 发布索引

状态：全部上传并验证完成。最终全文件校验完成时间：2026-09-13 16:29:59，新加坡时间。

主 collection：[Spectral Surgery](https://huggingface.co/collections/tianzl66/spectral-surgery-6a8fde803843c3f48b8fdcb1)。12 个仓库同时加入了原有 Code、Math、Instruction Following 子 collection；未覆盖历史仓库。

发布范围：Qwen3-8B、Llama-3.1-8B-Instruct × Magicoder、MetaMath、Tulu × 新增训练 seeds43/44。根目录权重是最终训练得到的 **原始 LoRA**，不是完整基座模型，也不是 HNS 修改后的权重。约 2.4 GB 的发布资料包含约 2.055 GB LoRA 权重、保存的 tokenizer、配置、结果及复现资料。HNS 全网格成绩、元数据与重建代码已附上，派生 HNS 权重未上传。

## 仓库与本任务成绩

以下是每个仓库根目录 LoRA 的本任务主指标，单位为百分比。每张模型卡另有 Base、0+0 重构控制及全部九种 HNS 设置的完整表格。

| 模型 | 训练任务 / 基准 | Seed43：仓库 / LoRA得分 | Seed44：仓库 / LoRA得分 |
|---|---|---|---|
| Qwen3-8B | Magicoder / HumanEval | [Seed43](https://huggingface.co/tianzl66/Qwen3-8B-Magicoder-50K-LoRA-E1-Seed43) / 62.80% | [Seed44](https://huggingface.co/tianzl66/Qwen3-8B-Magicoder-50K-LoRA-E1-Seed44) / 64.63% |
| Qwen3-8B | MetaMath / GSM8K | [Seed43](https://huggingface.co/tianzl66/Qwen3-8B-MetaMathQA-50K-LoRA-Seed43) / 84.00% | [Seed44](https://huggingface.co/tianzl66/Qwen3-8B-MetaMathQA-50K-LoRA-Seed44) / 84.00% |
| Qwen3-8B | Tulu / IFEval | [Seed43](https://huggingface.co/tianzl66/Qwen3-8B-InstructionFollowing-LoRA-Seed43) / 66.17% | [Seed44](https://huggingface.co/tianzl66/Qwen3-8B-InstructionFollowing-LoRA-Seed44) / 66.91% |
| Llama-3.1-8B-Instruct | Magicoder / HumanEval | [Seed43](https://huggingface.co/tianzl66/Llama-3.1-8B-Instruct-Magicoder-50K-LoRA-E1-Seed43) / 55.49% | [Seed44](https://huggingface.co/tianzl66/Llama-3.1-8B-Instruct-Magicoder-50K-LoRA-E1-Seed44) / 53.66% |
| Llama-3.1-8B-Instruct | MetaMath / GSM8K | [Seed43](https://huggingface.co/tianzl66/Llama-3.1-8B-Instruct-MetaMathQA-50K-LoRA-Seed43) / 75.36% | [Seed44](https://huggingface.co/tianzl66/Llama-3.1-8B-Instruct-MetaMathQA-50K-LoRA-Seed44) / 74.60% |
| Llama-3.1-8B-Instruct | Tulu / IFEval | [Seed43](https://huggingface.co/tianzl66/Llama-3.1-8B-Instruct-InstructionFollowing-LoRA-Seed43) / 62.85% | [Seed44](https://huggingface.co/tianzl66/Llama-3.1-8B-Instruct-InstructionFollowing-LoRA-Seed44) / 63.03% |

## 每个仓库的可复现资料

- 完整模型卡：实际样本数、epoch、更新步数、sequence length、micro/global batch、梯度累积、LR、scheduler、Adam 参数、LoRA 参数、聊天模板、response-only loss、截断与 padding、训练 seed 和数据子集 seed。
- `run_args.json`、`run_config.json`、`training_args.json`、最终 `trainer_state.json`：明确区分请求参数、预处理前估计和实际生效配置/步数。
- `publication.json`：精确基座下载 revision、基座分片/tokenizer SHA256、原始训练权重 SHA256、环境版本、原始配置归一化说明、同 seed 的其他任务 checkpoint 信息。
- `data/*`：实际本地训练文件的 SHA256 和选样行号。原始训练文本不再分发；上游数据 revision 未记录的限制已明确写出。
- `evaluation/*`：完整主指标、推理配置、实际基准输入文件及校验值、所有对应条件的压缩逐样本预测/打分数据。
- `hns/*`：网格配置、build manifest 和谱修改元数据；`0+0` 标为重构控制，不是 HNS 编辑。
- `code/*`：源代码快照、逐文件代码校验值和 train/build-hns/evaluate 复现助手。HNS 重建保留同基座/seed 的三个任务 adapter 注册顺序。
- `comparison/*`：原始运行与新 seeds 的逐项成绩、训练配置审计；不把历史42合并为已验证同配置的第三个 seed。
- `MANIFEST.sha256`：整个准备好的发布包的文件校验值，清单不包含自身。

JSON 凭证字段和私有文件路径已清理。仅对发布副本的 adapter_config 规范化基座模型 ID/revision；训练 safetensors 原字节未修改。中间 checkpoint、optimizer pickle、training_args.bin、基座权重和 scheduler 日志未上传。

## 必须保留的配置与统计边界

**实际 warmup=0 steps。** 新增12次训练虽然在 CLI 请求了 .03/.05/.10 比例，Transformers5.16.1 不接受 warmup_ratio；签名过滤后该参数没有生效。已更正此前本地汇总，模型卡同时列出请求比例和实际 zero-warmup。Magicoder 50K 为截断过滤前选择数，实际训练 Qwen49,936 / Llama49,941。

新 micro-batch/padding、实际 warmup、历史部分 Llama recipe 和数据身份未充分核实，因此不能宣称42/43/44是严格同配置三-training-seed结果。43/44是同新增recipe的两个训练重复。新 SD 不是 CI；完整重训的 bitwise 复现并未在发布过程中额外验证。

Llama43 成功推理用131072 batched tokens和默认编译缓存，其余新增组用65536并禁用编译缓存；复现助手的稳定性设置与历史Llama43缓存条件的差异已明确说明。IFEval重构控制本身的收益不可全归因于HNS。模型卡未混入较早HF卡片中不同prompt/生成预算的成绩，也未附上未完成的遗忘分数。

## 最终验证与版本追踪

12个训练权重远端LFS SHA256均与原始本地文件一致。**720个发布文件**在固定上传commit的远端LFS SHA256 / Git blob SHA1与准备好的清单逐项一致；12个仓库的主collection及任务子collection归属全部验证通过。

第一个仓库曾带入复现命令测试生成的单个Python字节码缓存。已从新发布仓库移除该生成缓存并更新清单，权重/配置/结果无变化。旧commit仍保留恢复路径，清理前后commit记录在上传receipt；本地缓存副本已移入临时隔离目录，没有删除训练源文件。

- [逐仓库commit、权重校验和上传receipt](hns_hf_upload_receipt_20260913.json)
- [全文件/collection验证记录](hns_hf_upload_verification_20260913.json)
- [三个运行的大表与配置审计](hns_three_seed_diagonal_20260913.md)
- 发布工具：`scripts/publish_hns_training_seeds_hf.py`。
