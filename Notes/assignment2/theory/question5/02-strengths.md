# Assignment 2 — Question 5.2: strengths of the ViT paper

**Goal:** identify defensible strengths and connect each claim to evidence in the paper rather than making an unsupported list of praise.

- Primary source: [Dosovitskiy et al., ViT, ICLR 2021](https://arxiv.org/html/2010.11929v2).
- [Previous: 5.1 — Summary](01-summary.md)
- [Next: 5.3 — Weaknesses](03-weaknesses.md)

**Status:** notes for the Strengths section of the one-page review; no LaTeX review has been written.

## 1. Simple and influential conceptual approach

The pure ViT uses a fairly standard Transformer encoder on linearly embedded image patches, without requiring a CNN backbone or a specialized attention pattern. This provides a clear test of whether the standard architecture can work for image recognition at scale. The patch representation reduces the token count relative to treating each pixel as a token, while global self-attention can connect distant image regions.

**Why this is a strength:** the method connects the successful sequence-modeling toolkit to images through a simple interface. It is a significant empirical demonstration of scaling rather than a claim that patch-based attention had never been considered; the paper explicitly discusses prior related work. See [Sections 2–3](https://arxiv.org/html/2010.11929v2#S2).

## 2. Strong transfer results and compute comparison

The paper evaluates transfer to several datasets and reports strong performance after large-scale pretraining. For example, its reported best result reaches 88.55% ImageNet accuracy. The authors also compare to BiT and Noisy Student and report pretraining cost in TPUv3-core-days. In their comparison, a JFT-pretrained ViT-L/16 exceeds the reported BiT-L results across the table's tasks with less pretraining compute.

**Why this is a strength:** the authors do more than propose an architecture; they show that it can be competitive in a realistic pretrain-and-transfer setting and discuss a relevant compute tradeoff. Qualify the claim: this is **pretraining compute under the compared setups**, not a universal total-training or inference-cost advantage. The authors themselves note that schedules and other choices affect such comparisons. See [Section 4.2](https://arxiv.org/html/2010.11929v2#S4.SS2).

## 3. More than one experimental angle

The paper compares pure ViTs, CNNs and CNN–Transformer hybrids; investigates multiple model sizes and pretraining data scales; and studies performance against pretraining compute. VTAB contributes natural, specialized and structured classification tasks rather than only one benchmark. The authors also investigate learned representations and report a preliminary self-supervised experiment.

**Why this is a strength:** these studies support analysis of *when* the architecture works, not just its single best score. In particular, the smaller-data results provide evidence for the tradeoff between CNN inductive bias and pretraining scale. The paper does not hide that ViT can perform worse with less pretraining data. See [Sections 4.1–4.6](https://arxiv.org/html/2010.11929v2#S4).

## 4. Usefulness and reproducibility

The authors release [fine-tuning code and pretrained models](https://github.com/google-research/vision_transformer), and report a strong model pretrained on public ImageNet-21k as well as results using JFT-300M. This makes downstream investigation easier than reporting results alone.

**Qualification:** the availability of code and a public-data model does not make the largest JFT-based experiments fully reproducible for everyone. Distinguish access to pretrained models from access to the complete data and compute needed to reproduce pretraining.

## How to use this in the review

Choose two or three strengths, support each with a specific example, and say *why* it matters. A concise review could emphasize the simple architecture, multi-scale comparisons, and strong transfer results, while keeping compute claims appropriately scoped.
