# SSR: Structured Subgraph Retrieval for Temporal Knowledge Graph Question Answering with LLMs

This repository contains the implementation of **SSR**, a framework for Temporal Knowledge Graph Question Answering (TKGQA) with Large Language Models (LLMs), accepted at SIGIR 2026.

📄 Paper: *SSR: Structured Subgraph Retrieval for Temporal Knowledge Graph Question Answering with LLMs*  
Authors: Ying Zhang, Li Zhang, Wenya Guo*, Shilong Ping, Xinying Qian  
(*Corresponding Author)

---

## Overview

Temporal Knowledge Graph Question Answering (TKGQA) requires reasoning over temporal facts of the form *(head, relation, tail, timestamp)*.

Existing LLM-based approaches typically:
- Linearize quadruples into text
- Perform top-*n* semantic retrieval

This leads to:
- Loss of structural constraints
- Loss of temporal constraints
- Noisy retrieval results

---

## SSR Framework

SSR addresses these issues via **structured retrieval instead of similarity-based ranking**.

The framework consists of four key components:

1. **Coarse-grained Text Retrieval**  
   Provides background knowledge for semantic grounding.

2. **Temporal Question Parser**  
   Uses LLMs to extract:
   - Subgraph pattern Ψ(q, F)
   - Temporal constraint Φ(q, F)

3. **Fine-grained Subgraph Retrieval**  
   Filters TKG facts using:
   - Structural constraints
   - Temporal constraints  
   → Ensures precise and complete retrieval

4. **Temporal Subgraph Compression**  
   Compresses retrieved subgraphs while preserving:
   - Start and end events  
   - Temporal evolution

Finally, the compressed subgraph is fed into the LLM for answer generation.

---

## Key Advantages

- Preserves **explicit graph structure**  
- Enforces **strict temporal constraints**  
- Eliminates noise from semantic retrieval  
- Improves reasoning on **complex temporal queries**

---

## Results

SSR achieves **state-of-the-art performance** on:

- **MultiTQ**
- **TimelineKGQA**

Key improvements include:

- +0.059 Hits@1 over strong baseline (RTQA) :contentReference[oaicite:0]{index=0}  
- Significant gains on **multi-hop** and **entity prediction** questions :contentReference[oaicite:1]{index=1}  
- Consistent improvements across different LLM backbones :contentReference[oaicite:2]{index=2}  

---

## Citation

```bibtex
@inproceedings{zhang2026ssr,
  title={SSR: Structured Subgraph Retrieval for Temporal Knowledge Graph Question Answering with LLMs},
  author={Zhang, Ying and Zhang, Li and Guo, Wenya and Ping, Shilong and Qian, Xinying},
  booktitle={Proceedings of SIGIR 2026},
  year={2026}
}
