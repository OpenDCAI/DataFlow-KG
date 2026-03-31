# DataFlow-KG

## Overview

DataFlow-KG is a llm-driven knowledge graph processing library, which is an extension version of the awesome repo DataFlow. It aims to provide reusable, extensible, and modular operators for knowledge graph construction, reasoning, retrieval, querying, and domain-specific applications.

DataFlow-KG organizes knowledge graph capabilities into different operator packages by graph type and application scenario. Each package contains specialized operators that can be composed into larger pipelines for data processing, graph construction, graph enrichment, reasoning, retrieval, evaluation, and downstream task support.


## Supported Knowledge Graph Categories

DataFlow-KG currently supports the following categories:

- **General Knowledge Graph (`general_kg`)**  
  General-purpose operators for universal KG construction, transformation, filtering, enhancement, and evaluation.

- **Commonsense Knowledge Graph (`commensense_kg`)**  
  Operators for commonsense knowledge extraction, normalization, and reasoning.

- **Graph Reasoning (`graph_reasoning`)**  
  Operators for graph-based inference, reasoning path construction, and reasoning-oriented graph processing.

- **Graph RAG (`graph_rag`)**  
  Operators for retrieval-augmented generation over graph-structured knowledge.

- **Hyper-relational Knowledge Graph (`hyper_relation_kg`)**  
  Operators for representing and processing hyper-relational facts with qualifiers or statement-level attributes.

- **Multimodal Knowledge Graph (`multi_model_kg`)**  
  Operators for integrating text, images, and other modalities into unified graph representations.

- **Temporal Knowledge Graph (`temporal_kg`)**  
  Operators for time-aware knowledge extraction, temporal relation modeling, and temporal reasoning.

- **Domain-Specific Knowledge Graph (`domain_kg`)**  
  Operators for domain-specific knowledge graph construction and applications.


---

## Repository Structure

```text
DataFlow-KG/
├── core_kg/
├── commensense_kg/
├── graph_reasoning/
├── graph_rag/
├── hyper_relation_kg/
├── multi_model_kg/
├── temporal_kg/
├── geospatial_kg/
└── legal_kg/