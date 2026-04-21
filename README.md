# DataFlow-KG

<p align="center">
  <b>DataFlow-KG</b>: An LLM-Driven Knowledge Graph Processing Library
</p>

<p align="center">
  Build, enrich, reason over, and operationalize knowledge graphs with composable DataFlow-KG operators.
</p>

<p align="center">
  <a href="https://github.com/OpenDCAI/DataFlow-KG">GitHub</a> |
  <a href="https://zhp-li197.github.io/DataFlow-KG-Doc/zh/">Documentation</a>
</p>

---

## 0. News

## 1. 🤖 Overview

DataFlow-KG is an LLM-driven knowledge graph processing library built on top of the DataFlow ecosystem. It is designed to provide reusable, extensible, and modular operators for knowledge graph construction, reasoning, retrieval, querying, and domain-specific applications.

Rather than treating KG workflows as isolated scripts, DataFlow-KG organizes graph capabilities into operator packages by graph type and application scenario. These operators can be composed into larger pipelines, including but not limited to:

- knowledge graph construction
- graph reasoning
- graph retrieval
- domain-specific knowledge graph applications

DataFlow-KG aims to serve as a unified infrastructure layer for research and development on graph-centric LLM applications.


## 2. ✨ Key Features

### 2.1. Modular Operator Library for KG Workflows
DataFlow-KG provides reusable operators that can be flexibly composed into pipelines for graph construction, graph enrichment, reasoning, retrieval, and task-specific graph processing. Operators are not standalone utilities. They are designed to be assembled into end-to-end workflows, enabling scalable and reproducible graph data engineering.

### 2.2 Unified Support for Multiple KG Paradigms
The library supports a broad range of graph settings in one framework, including general KG, commonsense KG, temporal KG, multimodal KG, hyper-relational KG, Graph RAG, and domain-specific KGs. As an extension of DataFlow, DataFlow-KG follows the same design philosophy of composable operators and pipeline-based processing, making it easy to integrate with broader data preparation workflows.

### 2.3. Research-to-Application Coverage
The framework is designed for both research scenarios and practical vertical applications, supporting graph processing tasks from foundational KG construction to specialized domain deployment.


## 3. 🔍 Installation

### 3.1. Create and activate a Python environment

```bash
conda create -n dfkg python=3.10
conda activate dfkg
````

### 3.2. Install DataFlow-KG

```bash
pip install uv
uv pip install dataflow-kg
```

If you want to enable **local GPU inference**, use:

```bash
conda create -n dfkg python=3.10
conda activate dfkg

pip install uv
uv pip install dataflow-kg[vllm]
```

> DataFlow-KG supports Python >= 3.10.

### 3.3. Verify the installation

You can check whether the installation is successful with:

```bash
dfkg -v
```

If the installation is correct and DataFlow-KG is the latest release, you will see something like:

```log
open-dataflow-kg codebase version: 0.0.2
        Checking for updates...
        Local version:  0.0.2
        PyPI newest version:  0.0.2
        You are using the latest version: 0.0.2.
```

In addition, the `dfkg env` command can be used to inspect the current hardware and software environment, which is useful for bug reporting:

```bash
dfkg env
```


## 4. 🚀 Quickstart

DataFlow-KG follows a **code generation + custom modification + script execution** workflow.  In practice, you initialize a project with the CLI, customize the generated pipeline script if needed, and then run the Python file to execute your workflow.

You can get started in **three steps**.

### 4.1. Initialize a project

Run the following command in an empty directory:

```bash
dfkg init
````

### 4.2. Choose a pipeline type

Pipelines with the same name across different folders are usually incremental variants with different dependency requirements:

| Directory       | Required Resources    |
| --------------- | --------------------- |
| `cpu_pipelines` | CPU only              |
| `api_pipelines` | CPU + LLM API         |
| `gpu_pipelines` | CPU + API + local GPU |

> **Tip:** If you are new to DataFlow-KG, start with `api_pipelines`.
> Later, if you have a local GPU, you can replace `LLMServing` with a local model backend.


### 4.3. Run your first pipeline

Go into any pipeline directory, for example:

```bash
cd api_pipelines
```

Open one of the generated Python pipeline files. In most cases, you only need to check two configurations:

#### 4.3.1 Input data path

```python
self.storage = FileStorage(
    first_entry_file_name="<path_to_dataset>"
)
```

By default, this points to the provided example dataset, so you can run it directly.
You can also replace it with your own dataset path.

#### 4.3.2 LLM serving configuration

If you are using an API-based serving backend, set the API key first.

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

Then run the pipeline script:

```bash
python xxx_pipeline.py
```

---



## 5. 📚 Licence

DataFlow-KG is released under the **Apache License 2.0**.



## 6. 🎓 Citation
If you use DataFlow-KG in your research, please cite:

```bibtex
@misc{dataflowkg2026,
  title={DataFlow-KG: LLM-Driven Knowledge Graph Processing Library},
  author={DataFlow-KG Team},
  year={2026},
  howpublished={\url{https://github.com/OpenDCAI/DataFlow-KG}}
}
