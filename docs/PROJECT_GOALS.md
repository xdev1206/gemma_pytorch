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

如果 `models/` 软链接或任一模型不存在，运行脚本时必须提示用户先下载
模型，再继续执行一致性验证。每次对齐验证应记录模型名称、prompt 或图片、
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

## 对齐测试数据

测试用例和输入清单位于 `tests/`：`tests/data/alignment_cases.json` 固定
模型、prompt、图片及多模态交错输入；确定性测试位于 `tests/test_alignment.py`。
使用 `python -m unittest tests.test_alignment -v` 运行；如果 `models/` 或依赖
缺失，测试应明确提示下载模型或安装依赖。真实 checkpoint 数值对齐应将临时
输出放在仓库外，并记录模型、设备、dtype、生成长度和容差。Gemma 3 1B 已完成
`model.safetensors` loader 适配，并已完成一次固定 prompt 的 CUDA smoke 对齐；Gemma 4 E2B 已完成
文本、视觉、音频骨干、tokenizer 和图像 processor 已适配，并已完成固定文本
prompt 的 CUDA 下一个 token 对齐；音频 processor 及完整多模态数值一致性对齐
仍需实现严格的全场景数值容差覆盖。固定图像+文本 CUDA/bfloat16 用例已完成
端到端对齐：top-1/top-5 token 一致，平均 logits 误差约为 `0.21`。

## 范围边界

本项目聚焦 Gemma 模型的 PyTorch 推理实现及其必要基础设施；训练平台、
服务化部署和与参考项目无关的通用框架功能不属于当前目标，除非另有明确
需求。
