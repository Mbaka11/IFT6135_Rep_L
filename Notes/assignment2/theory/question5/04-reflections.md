# Assignment 2 — Question 5.4: ViT reflections

**Goal:** connect the paper to course concepts and develop a specific, testable research idea for the Reflections section of the one-page review.

- [ViT paper, version 2](https://arxiv.org/html/2010.11929v2)
- [5.1 — Summary](01-summary.md) · [5.2 — Strengths](02-strengths.md) · [5.3 — Weaknesses](03-weaknesses.md)

## 1. Connections to the course

**CNNs and inductive bias.** Convolutions encode local connectivity and shared filters. ViT instead turns image patches into tokens and uses self-attention to combine information across them. Patch extraction and position embeddings still provide spatial structure; ViT is not completely free of spatial assumptions. The paper's data-scale experiments illustrate a tradeoff: CNN biases can help with smaller pretraining datasets, while ViT performs strongly with large-scale pretraining.

**Attention and Transformers.** As in the attention mechanisms studied in class, queries and keys determine weights used to combine values. ViT applies multi-head self-attention to patch embeddings, so a token can incorporate information from other image regions. Its Transformer encoder also uses residual connections, layer normalization and MLP blocks. A learned classification token supplies the representation used by the classifier.

**Optimization and regularization.** The paper examines the effect of pretraining scale and uses regularization; its smaller-data experiments highlight that model capacity alone is not enough to guarantee generalization. This links to the course's discussion of overfitting and regularization.

## 2. A concrete future experiment

An idea worth testing is whether introducing a modest local bias helps ViT when **pretraining data is limited**, without sacrificing its ability to combine distant regions.

- **Hypothesis:** under a small public-data budget, a model that combines local feature extraction with global attention will transfer better than a pure ViT of comparable cost. This is a hypothesis, not a result established by the proposed experiment.
- **Comparison:** train a pure ViT and a CNN–Transformer hybrid on the same public pretraining dataset, then fine-tune them on the same held-out tasks. The paper itself already studies hybrids, so the new contribution would be a focused follow-up under specific smaller data budgets, not the idea of a hybrid itself.
- **Controls:** use multiple training-set sizes and random seeds, comparable pretraining compute, input resolution and tuning budgets; report model size and actual cost rather than assuming they are identical. Equalizing every resource simultaneously may not be possible, so report any remaining differences.
- **Measurements:** downstream accuracy and variation across seeds, pretraining and fine-tuning compute, and inference latency/memory. Include a strong CNN baseline to see whether the hybrid actually improves the tradeoff.
- **Interpretation:** if the hybrid helps only at low data volumes, this would support the hypothesis within those tested settings; if it does not, revisit the bias or training setup rather than claiming local bias is universally unhelpful.

## 3. Other possible directions

One could also investigate self-supervised pretraining or efficient attention for higher-resolution images. The paper already includes a **preliminary** masked-patch experiment, so a proposal should explain what broader comparison or new objective it would add rather than claiming the paper ignored self-supervision.

## For the review

The review is subjective. Adapt the proposed experiment to your own interests and use first person only if it reflects your perspective. In the final one-page review, connect the idea briefly to the course and state the hypothesis and comparison; the full protocol above is for studying and planning, not necessarily for inclusion verbatim.
