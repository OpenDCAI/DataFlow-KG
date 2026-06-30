# DataFlow-KG

<p align="center">
  <img src="static/dataflow-KG%20framework.png" alt="DataFlow-KG framework" width="100%">
</p>

<p align="center">
  <b>DataFlow-KG</b>：一个由 LLM 驱动的知识图谱处理库
</p>

<p align="center">
  使用可组合的 DataFlow-KG 算子来构建、增强、推理、检索并落地知识图谱应用。
</p>

<p align="center">
  <a href="https://github.com/OpenDCAI/DataFlow-KG">GitHub</a> |
  <a href="https://zhp-li197.github.io/DataFlow-KG-Doc/zh/">文档</a> |
  <a href="README.md">English README</a>
</p>

---

## 0. 动态

## 1. 🤖 项目概览

DataFlow-KG 是一个构建在 [DataFlow](https://github.com/OpenDCAI/DataFlow) 生态之上的、由大语言模型驱动的知识图谱处理库。它旨在为知识图谱构建、推理、检索、查询以及领域应用提供可复用、可扩展、模块化的算子。[DataFlow](https://github.com/OpenDCAI/DataFlow) 原项目为数据中心的 LLM 工作流提供了一个非常干净、优雅且高度可扩展的基础框架。

不同于把 KG 工作流写成分散的脚本，DataFlow-KG 按图谱类型和应用场景组织算子包。这些算子可以进一步组合成更大的 pipeline，包括但不限于：

- 知识图谱构建
- 图谱推理
- 图谱检索
- 领域知识图谱应用

DataFlow-KG 致力于成为图谱中心 LLM 应用研发的一层统一基础设施。

## 2. ✨ 关键特性

### 2.1 面向 KG 工作流的模块化算子库

DataFlow-KG 提供了一组可复用算子，可灵活组合为图谱构建、图谱增强、图谱推理、图谱检索以及任务定制处理的 pipeline。它们不是孤立的小工具，而是面向端到端工作流设计的组件，适合可扩展、可复现的图谱数据工程。

### 2.2 统一支持多种 KG 范式

该框架在同一套体系下支持多种图谱场景，包括通用知识图谱、常识知识图谱、时序知识图谱、多模态知识图谱、超关系知识图谱、Graph RAG，以及多种领域知识图谱。作为 DataFlow 的扩展，DataFlow-KG 延续了可组合算子和 pipeline 式处理的设计理念，便于接入更大的数据准备与应用流程。

### 2.3 覆盖研究到应用的完整链路

框架同时面向科研和实际垂直应用，从基础的 KG 构建到面向特定领域的部署，均可支持。

## 3. 🔍 安装

### 3.1 创建并激活 Python 环境

```bash
conda create -n dfkg python=3.10
conda activate dfkg
```

### 3.2 安装 DataFlow-KG

```bash
pip install uv
uv pip install dataflow-kg
```

如果你希望启用**本地 GPU 推理**，可以使用：

```bash
conda create -n dfkg python=3.10
conda activate dfkg

pip install uv
uv pip install dataflow-kg[vllm]
```

> DataFlow-KG 支持 Python >= 3.10。

### 3.3 验证安装

你可以通过下面的命令检查安装是否成功：

```bash
dfkg -v
```

如果安装正确，并且你使用的是最新版本，输出会类似：

```log
open-dataflow-kg codebase version: 1.0.1
        Checking for updates...
        Local version:  1.0.1
        PyPI newest version:  1.0.1
        You are using the latest version: 1.0.1.
```

此外，也可以使用 `dfkg env` 查看当前硬件和软件环境，这在排查 bug 时很有帮助：

```bash
dfkg env
```

## 4. 🚀 快速开始

DataFlow-KG 采用“**代码生成 + 按需修改 + 脚本执行**”的工作方式。通常你会先用 CLI 初始化项目，再根据需要调整生成的 pipeline 脚本，最后直接运行 Python 文件来执行工作流。

你可以按以下三步上手。

### 4.1 初始化项目

在一个空目录中执行：

```bash
dfkg init
```

### 4.2 选择 pipeline 类型

不同目录下同名 pipeline 往往代表不同依赖条件下的递进版本：

| 目录 | 所需资源 |
| --- | --- |
| `api_pipelines` | CPU + LLM API |
| `gpu_pipelines` | CPU + API + 本地 GPU |

> **建议：** 如果你刚接触 DataFlow-KG，优先从 `api_pipelines` 开始。
> 如果后续具备本地 GPU，再把 `LLMServing` 替换成本地模型后端即可。

### 4.3 运行你的第一个 pipeline

进入任意 pipeline 目录，例如：

```bash
cd api_pipelines
```

打开生成的 Python pipeline 文件。大多数情况下，你只需要确认两项配置：

#### 4.3.1 输入数据路径

```python
self.storage = FileStorage(
    first_entry_file_name="<path_to_dataset>"
)
```

默认情况下，这里会指向仓库提供的示例数据，因此可以直接运行；你也可以替换成自己的数据路径。

#### 4.3.2 LLM Serving 配置

如果你使用的是基于 API 的 serving 后端，请先设置 API Key。

**Linux / macOS**

```bash
export DF_API_KEY=sk-xxxxx
```

**Windows CMD**

```bat
set DF_API_KEY=sk-xxxxx
```

**PowerShell**

```powershell
$env:DF_API_KEY="sk-xxxxx"
```

随后运行 pipeline 脚本：

```bash
python xxx_pipeline.py
```

---

## 5. 📚 许可证

DataFlow-KG 采用 **Apache License 2.0** 开源协议。

## 6. 🎓 引用

如果你在研究中使用了 DataFlow-KG，请引用：

```bibtex
@misc{dataflowkg2026,
  title={DataFlow-KG: LLM-Driven Knowledge Graph Processing Library},
  author={DataFlow-KG Team},
  year={2026},
  howpublished={\url{https://github.com/OpenDCAI/DataFlow-KG}}
}
```
