# Gemma 模型实现架构

## 当前代码组织

项目保留上游的 `gemma/` 平铺结构，同时仅按模型版本增加子目录，降低后续
同步上游代码时的差异：

```text
gemma/
├── config.py, model.py, tokenizer.py       # Gemma 3 文本基础模块
├── gemma3_model.py, gemma3_preprocessor.py # Gemma 3 多模态组合与预处理
├── gemma3/checkpoint.py                    # Gemma 3 权重映射
├── gemma4/                                 # Gemma 4 独立实现包
│   ├── gemma4_config.py, gemma4_model.py   # 文本主干与组合模型
│   ├── gemma4_vision.py, gemma4_audio.py   # 视觉/音频塔
│   ├── gemma4_processor.py, gemma4_tokenizer.py
│   └── checkpoint.py                       # 权重映射
├── checkpoint_utils.py                     # 公共 safetensors 操作
└── checkpoint.py                           # 兼容导出层
```

`scripts/` 放置文本、多模态和 XLA 运行入口；`tests/` 放置资源清单、组件契约、
checkpoint、运行时和参考实现对齐测试；`tests/run_alignment.py` 按 profile 统一运行并
生成 JSON/Markdown 报告。仓库根目录的 `models/` 是本地模型软链接，不提交模型文件。

## 测试代码组织

```text
tests/
├── data/alignment_cases.json          # 模型、输入、设备、dtype、阈值
├── alignment_support.py               # manifest、模型和图片资源检查
├── test_manifest.py                   # manifest schema 和资源完整性
├── test_alignment.py                  # 无 checkpoint 的组件契约
├── test_checkpoint_loading.py         # 权重 key 和 loader 契约
├── test_runtime_alignment.py          # 本地真实 checkpoint smoke
├── test_gemma4_reference_alignment.py # Transformers 数值/行为对齐
├── alignment_report.py                # 结果和误差指标序列化
└── run_alignment.py                   # profile runner 和退出码门禁
```

测试代码分为四层：

1. `contract`：验证配置、预处理、模型形状、token 插入和 checkpoint 映射，不能
   依赖运行时设备。
2. `runtime-smoke`：使用 manifest 指定的真实模型和设备，验证 Gemma 3 文本以及
   Gemma 4 文本、图像和音频路径可运行。
3. `gemma4-reference`：加载同一 checkpoint 的 Transformers 实现，比较文本 logits、
   greedy generation、视觉特征、音频特征、音频文本 logits 和图文 logits。
4. `all`：按固定顺序执行以上三层并生成汇总报告。

`alignment_cases.json` 是输入和阈值的唯一来源；测试不得重新硬编码 prompt、
backend 或误差阈值。`AlignmentTestResult` 将每个 case 的状态、耗时和结构化
误差写入报告。`runtime-smoke`、`gemma4-reference` 和带 `--strict` 的 `all`
profile 将 skip 视为失败，确保报告 PASS 表示验收项确实执行。

## Gemma3

Gemma3 由 `gemma3_model.py` 组合文本解码器和 SigLIP 视觉编码器。文本路径
为 Token Embedding、hidden-size 缩放、多层 Gemma Decoder、最终 RMSNorm 和
LM Head；Decoder 使用 RMSNorm、RoPE、Gated MLP、KV Cache，并在 Global
Attention 与 Local Sliding Attention 之间切换。

图像路径如下：

```text
PIL 图像 -> pan-and-scan -> SigLIP Patch/Transformer
         -> 4x4 average pooling -> RMSNorm -> 投影 -> 替换 image placeholder
```

图像 placeholder 还会参与图像区域双向注意力，同时保持文本因果注意力。

Gemma 3 的权重加载位于 `gemma/gemma3/checkpoint.py`，文本模型仍复用
`gemma/model.py`；`gemma/gemma3/__init__.py` 只暴露该版本 loader。Gemma 3
当前主要作为文本 smoke baseline，多模态数值验收仍待完善。

## Gemma4 E2B

Gemma4 的文本主干在 `gemma/gemma4/gemma4_model.py`，由主 Token Embedding、Context-PLE、
Token-PLE 和多层 Decoder 组成。Context-PLE 与 Token-PLE 合并后，为每层生成
独立的 per-layer input。每层支持 Sliding/Full Attention，使用 Q/K/V RMSNorm、
类型相关 RoPE，并按配置复用 K/V；末端使用 RMSNorm、权重绑定的 LM Head 和
logit softcap。

Gemma4 的多模态结构为：

```text
图像 patch -> Gemma4VisionModel -> padding-aware pooling -> 视觉投影 ┐
音频 log-mel -> Conv2D subsample -> 相对位置音频层 -> 音频投影 ─────┤
                                                                    └─ 替换特殊 token -> Gemma4 文本解码器
```

视觉塔包含 patch projection、二维位置 id、多维 RoPE、RMSNorm/Attention/MLP
层和参考顺序的 BF16 pooling。音频塔包含两层 Conv2D 降采样、chunked 相对位置
Attention、局部上下文 mask、depthwise LightConv、残差 FFN 和 attention softcap。
`gemma/gemma4/gemma4_processor.py` 负责图像 patch 与 16 kHz log-mel 特征。权重加载按版本
拆分：Gemma3 使用 `gemma/gemma3/checkpoint.py`，Gemma4 使用
`gemma/gemma4/checkpoint.py`，公共 safetensors 操作位于公开的
`gemma/checkpoint_utils.py`；根目录 `gemma/checkpoint.py` 仅保留兼容转发。

Gemma 4 包的 `__init__.py` 提供常用模型、配置、处理器和 tokenizer 的统一导出，
内部模块使用包内相对导入，避免与 Gemma 3 的同名基础模块耦合。

Gemma 4 的端到端路径是：配置与 tokenizer 读取 -> 文本 token embedding、
Context-PLE/Token-PLE -> 文本 Decoder -> tied LM head；存在图像或音频时，
对应 tower 先生成投影后的 soft tokens，再按特殊 token 位置写入文本序列。
`Gemma4TextConfig` 从 `config.json` 读取层数、头数、head dimension、attention
类型、KV 共享和 softcap 等结构参数，不应在调用方重复硬编码。

## 公共边界与验证架构

版本专属 loader 只负责 checkpoint key 到本地模块的映射；`checkpoint_utils.py`
负责 safetensors 打开、路径解析和参数/缓冲区安全复制。根 loader 仅为兼容旧调用
保留转发，不承载版本逻辑。测试按以下顺序形成验收链：

```text
manifest/resource -> component contracts -> checkpoint mapping
                  -> local runtime smoke -> Transformers reference comparison
                  -> profiled alignment_report.json/.md
```

模型缺失时，测试输出下载地址、`models/<model-id>/` 期望目录和缺失文件，并以失败
结束，不能通过 skip 伪装为成功。`tests/run_alignment.py` 提供
`contract`、`runtime-smoke`、`gemma4-reference` 和 `all` 四个 profile；其中
runtime/reference profile，以及带 `--strict` 的 `all` profile，将 skip、设备
不可用和模型缺失都判为失败。Gemma 4 正式参考契约固定为 BF16 +
manifest 指定的 Transformers `eager` attention，覆盖文本 logits、贪心生成、
视觉特征、音频特征、音频文本 logits 和图文 logits。测试报告记录实际环境、
阈值与 max/mean absolute error。SDPA、其他 dtype/设备以及 Gemma 3 多模态数值
对齐不属于当前验收范围。

## 当前验证边界

Gemma4 正式参考验收固定为 BF16，并显式使用 Transformers `eager` attention；
分别检查文本 logits、贪心生成、视觉特征、音频特征、音频文本 logits 和图文 logits。SDPA、其他
dtype/设备，以及 Gemma3 多模态数值一致性暂不属于当前验收范围。
