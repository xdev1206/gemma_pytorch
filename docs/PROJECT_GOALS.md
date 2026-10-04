# 项目目标

## 参考项目

本项目以 [Google DeepMind Gemma](https://github.com/google-deepmind/gemma)
为主要参考，跟踪其模型架构、分词器、预处理流程、权重加载方式和推理
行为。上游项目的实现和公开文档是功能定义与结果对齐的依据。

## 核心目标

在本仓库中使用 PyTorch 复刻参考项目的核心能力，形成可独立安装、运行
和验证的 Gemma 实现。复刻工作应优先覆盖：

1. Gemma 文本模型及其配置、权重加载和生成流程。
2. Gemma 多模态模型涉及的图像预处理、SigLIP 视觉编码器和文本生成。
3. CPU、CUDA GPU 以及 PyTorch/XLA 运行路径。
4. 与参考项目一致的 tokenizer 行为、输入格式、缓存逻辑和输出语义。

## 一致性对齐模型

模型基准统一通过本项目根目录下的 `models/` 软链接访问，软链接指向统一
管理的模型目录。对齐模型包括：

- `gemma-3-1b-it`
- `gemma-4-E2B-it`

实现模块边界、代码组织和数据流见 [模型实现架构](ARCHITECTURE.md)。Gemma 4
实现集中在 `gemma/gemma4/`，Gemma 3 保持现有平铺模块，以减少与上游合并时的差异。

如果 `models/` 软链接或任一模型不存在，运行脚本时必须提示用户从对应的
Hugging Face 页面下载模型，并显示期望的本地目录和必需文件；依赖模型的测试
必须失败，不能通过跳过测试伪装成验证成功。
Gemma 3 下载地址为
<https://huggingface.co/google/gemma-3-1b-it>，Gemma 4 下载地址为
<https://huggingface.co/google/gemma-4-E2B-it>。每次对齐验证应记录模型名称、prompt 或图片、
运行设备、dtype 以及比较容差；不得只记录最终输出而省略运行条件。

## 实现原则

- 以 PyTorch 原生模块、张量操作和设备管理为基础，保持代码清晰、可调试。
- 优先保证模型结构、张量形状、数据类型和数值结果的兼容性，再进行性能优化。
- 复用本项目已有的 `gemma/`、`scripts/` 和 `docker/` 组织方式，避免引入无关抽象。
- 对因 PyTorch API、硬件能力或 checkpoint 格式产生的差异，在代码和文档中明确说明。

## 验收标准

- 能依据 `README.md` 完成依赖安装，并使用公开 checkpoint 启动文本推理。
- 多模态模型能够处理仓库示例图片并生成结果。
- 在可用硬件上验证 CPU、CUDA 或 XLA 路径，并记录命令、模型版本和硬件信息。
- 使用上述固定模型、prompt、图片和合理数值容差，与参考项目进行输出或行为对比。
- 新增模型能力时同步更新测试、运行示例和相关文档。

## 对齐测试架构

Gemma 4 验收采用清单、契约、checkpoint 加载、运行时行为和参考数值对齐五层结构。
`tests/data/alignment_cases.json` 固定模型、prompt、图片、设备、dtype、seed 和容差；
`tests/test_manifest.py` 验证清单和资源；`tests/test_alignment.py` 验证组件契约；
`tests/test_runtime_alignment.py` 在显式开启时加载真实 checkpoint，覆盖文本、图片
next-token 和音频 multimodal forward；`tests/test_gemma4_reference_alignment.py`
使用同一 checkpoint 的 Transformers Gemma 4 实现比较文本 logits、贪心生成、视觉特征
和音频特征。
使用 `python -m tests.run_alignment --profile contract` 运行快速检查；使用
`--profile runtime-smoke` 或 `--profile gemma4-reference` 运行严格的真实
checkpoint 对齐，严格 profile 中任何 skip 都会失败。
如果 `models/` 或依赖缺失，测试必须明确提示下载模型或安装依赖。完整 Gemma 4
checkpoint 数值对齐必须同时满足 logits 最大/平均绝对误差、文本多 token 贪心生成、
视觉特征、音频特征、音频文本 logits 和图像行为的参考比较；next-token 单点匹配
不能替代数值验收。prompt、backend 和误差阈值均由 manifest 驱动，并写入测试报告。
当前文本与参考默认输出仍有约 `0.5` 的最大 logits 误差，但根因已确定为 attention backend 不一致：Transformers 参考默认使用 `sdpa`，本地实现使用 eager。正式参考对齐测试在 `from_pretrained` 时显式传入 `attn_implementation="eager"`，并断言顶层和语言模型配置均为 eager；在此契约下，本地 logits 与参考逐元素一致（`max_abs=0`），文本贪心生成也已通过。生成器同时读取 checkpoint 的 EOS ID 列表（`1`、`106`），在生成 EOS 后停止，匹配 Transformers 的序列长度行为。本地与参考默认 `sdpa` 的差异为 `max_abs=0.5、mean_abs≈0.05424`，参考 `sdpa` 与 eager 也有同样差异。第 0 层主输入、input Norm、Q/K/V 投影及其 Norm、RoPE 前后 Q/K 均一致；参考 eager attention 下 GQA 后 K/V、softmax 权重、value 聚合、`o_proj` 输入及输出也均为逐元素一致。因此不应继续修改 eager attention 数学逻辑；若未来验收 SDPA，必须另行实现并验证对应 backend。
音频对齐已确认 subsample 输出与参考完全一致；位置编码现在先以 FP32 计算 `inv_timescales`，再转换为当前模型默认 dtype，匹配参考的构造后转换顺序，也修复了本地 checkpoint loader 不会转换非持久 buffer 的问题。运行时位置编码回归测试现已完全一致（`MSE=0`、`max_abs=0`）。本地现在传播两层 subsample mask，构造参考风格的 4D sliding mask，转换为 BF16 additive mask，执行相同的 blocked padding/gather，并在 eager attention 中使用 `attention_mask.logical_not()`。当前 Gemma 4 reference profile 已覆盖音频 tower 和音频文本 logits；本次完整测试报告中两者均为 `MSE=0`、`max_abs=0`。后续若继续做层级诊断，应以严格 profile 的结构化误差报告为准。

视觉对齐逐层检查确认 patch embedding、layer 0 输入 norm、Q/K/V 投影及 Q/K norm 完全一致；首次差异出现在 self-attention 内部。参考在 RoPE 旋转前将 cos/sin 转为 BF16，本地此前在 FP32 cos/sin 下完成乘加后才转换；本地已同步为先转换 cos/sin 再旋转。
RoPE 修复后，16 个 vision layer 的有效 token 边界均完全一致；剩余差异来自 pooling 的 dtype 顺序。参考先将 FP32 平均池化结果转换为 BF16，再乘 `sqrt(hidden_size)`，本地已同步该顺序。
视觉 pooling 的最终缩放也已显式按参考执行：BF16 池化结果先转 FP32，再乘 `sqrt(hidden_size)` 后转换回模型 dtype。
视觉 tower 不再提前裁剪 padding patch，而是与参考一样保留 padding 参与零值池化，再使用 pool mask 移除 padding soft tokens，以保持矩阵归约顺序一致。
真实 checkpoint 验证中，视觉特征和图文 logits 对齐测试均已通过。

Gemma 3 1B 当前仅保留文本 smoke baseline，Gemma 3 多模态和数值容差属于待完善项，
不纳入本轮 Gemma 4 验收门槛。

## 范围边界

本项目聚焦 Gemma 模型的 PyTorch 推理实现及其必要基础设施；训练平台、
服务化部署和与参考项目无关的通用框架功能不属于当前目标，除非另有明确
需求。
