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

## 1. Overview

DataFlow-KG is an LLM-driven knowledge graph processing library built on top of the DataFlow ecosystem. It is designed to provide reusable, extensible, and modular operators for knowledge graph construction, reasoning, retrieval, querying, and domain-specific applications.

Rather than treating KG workflows as isolated scripts, DataFlow-KG organizes graph capabilities into operator packages by graph type and application scenario. These operators can be composed into larger pipelines, including but not limited to:

- knowledge graph construction
- graph transformation and enrichment
- graph reasoning
- graph retrieval and Graph RAG
- evaluation and downstream task support
- domain-specific knowledge graph applications

DataFlow-KG aims to serve as a unified infrastructure layer for research and development on graph-centric LLM applications.

---

## 2. ✨ Key Features

### 1. Modular Operator Library for KG Workflows
DataFlow-KG provides reusable operators that can be flexibly composed into pipelines for graph construction, graph enrichment, reasoning, retrieval, and task-specific graph processing.

### 2. Unified Support for Multiple KG Paradigms
The library supports a broad range of graph settings in one framework, including general KG, commonsense KG, temporal KG, multimodal KG, hyper-relational KG, Graph RAG, and domain-specific KGs.

### 3. Pipeline-Oriented Design
Operators are not standalone utilities. They are designed to be assembled into end-to-end workflows, enabling scalable and reproducible graph data engineering.

### 4. Native Integration with the DataFlow Ecosystem
As an extension of DataFlow, DataFlow-KG follows the same design philosophy of composable operators and pipeline-based processing, making it easy to integrate with broader data preparation workflows.

### 5. Research-to-Application Coverage
The framework is designed for both research scenarios and practical vertical applications, supporting graph processing tasks from foundational KG construction to specialized domain deployment.

---

## 3. Installation

### 1. Clone the repository

```bash
git clone https://github.com/OpenDCAI/DataFlow-KG.git
cd DataFlow-KG


---

## 4. Quickstart


## 5. Licence


## 6. Citation
