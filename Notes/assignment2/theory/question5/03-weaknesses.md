# Assignment 2 — Question 5.3: weaknesses and possible improvements

**Goal:** develop fair, evidence-based criticisms of the ViT paper and suggest concrete ways to test or address them.

- Primary source: [Dosovitskiy et al., ViT, ICLR 2021](https://arxiv.org/html/2010.11929v2).
- [Previous: 5.2 — Strengths](02-strengths.md)
- [Summary and experimental context](01-summary.md)

**Status:** notes for the Weaknesses section of the one-page review; no LaTeX review has been written.

## 1. Dependence on large-scale pretraining

The authors find that ViT can overfit more and perform worse than comparable ResNets when pretrained on smaller datasets; large datasets improve the relative outcome. The largest headline results use JFT-300M, an in-house dataset, and substantial accelerator resources.

**Why it matters:** groups without such data or hardware may not be able to reproduce the largest pretraining experiments, and performance with limited data may matter in practice. This is not a claim that ViT *always* requires JFT: the paper reports results using public ImageNet-21k and studies smaller datasets. See [Section 4.3](https://arxiv.org/html/2010.11929v2#S4.SS3).

**Possible improvement:** report more controlled comparisons on public datasets and under smaller training budgets, including multiple model sizes and carefully tuned regularization for each architecture. The paper does study data size and basic regularization already; the suggestion is to broaden that analysis, not to claim it is absent.

## 2. High-resolution and patch-size costs

For non-overlapping patches, the number of image tokens is $N=HW/P^2$. A standard dense self-attention score matrix scales as $N^2$: reducing patch size or increasing resolution therefore makes the attention component more costly. For example, halving $P$ at fixed image size gives four times as many patches and roughly sixteen times as many patch-to-patch score entries.

**Why it matters:** small patches may retain finer detail but increase attention compute and memory. This is a complexity statement about the attention matrix, **not** a claim that total runtime or memory always grows sixteenfold: projection/MLP work, implementation and hardware also matter. The paper discusses patch-size costs and empirical throughput; see [Sections 4.1 and Appendix D.5](https://arxiv.org/html/2010.11929v2#A4.SS5).

**Possible improvement:** test hierarchical or efficient-attention variants, or different patch sizes, at matched accuracy, resolution and compute. These are potential follow-up designs, not experiments the paper is being credited with.

## 3. Scope of evaluation and baselines

The main reported tasks are image classification and classification-style transfer, including VTAB's diverse task groups. This is valuable, but does not by itself establish how the same design performs on every dense-prediction task such as segmentation or detection, or under distribution shift.

**Possible improvement:** evaluate transfer to detection and segmentation, and study robustness and efficiency on relevant deployment hardware. Keep the distinction between *not established by this paper's evaluation* and *proven ineffective*: the former is a scope limitation, the latter is not supported here.

## 4. Comparability and learning setup

The paper uses substantial comparisons, including a controlled performance-versus-pretraining-compute study. Still, architecture, hyperparameters, pretraining schedule and data can affect comparisons, so a strong result in one setup does not prove that the architecture alone caused every reported advantage. Pretraining compute alone also does not measure inference latency, memory or total downstream fine-tuning cost.

**Possible improvement:** supplement the reported comparison with matched-data, matched-budget baselines and separate measures of pretraining, fine-tuning and inference. Do not claim the paper made no attempt at a controlled comparison—it did.

The primary pretraining is supervised. The masked-patch self-supervised experiment is preliminary and, in the reported setting, trails supervised pretraining. More extensive tests of label efficiency and self-supervised objectives would help assess whether dependence on labeled data can be reduced; see [Section 4.6](https://arxiv.org/html/2010.11929v2#S4.SS6).

## How to use this in the review

Select two or three limitations, explain why they matter, and suggest a plausible test or improvement. Keep criticisms tied to the paper's actual scope and avoid hindsight claims that the authors failed to compare methods developed after publication.
