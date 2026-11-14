# Assignment 2 — Question 5.1: ViT paper summary

**Goal:** understand the problem, method and main experimental findings before writing the summary section of the one-page review. These are study notes, not a finished review.

- Primary source: Dosovitskiy et al., [*An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929v2), ICLR 2021; [paper text](https://arxiv.org/html/2010.11929v2).
- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Next: 5.2 — Strengths](02-strengths.md)

**Status:** notes only; the Question 5 LaTeX review is blank. The assignment asks for **one review with four sections**—Summary, Strengths, Weaknesses and Reflections—not separate answers to the guiding questions.

## 1. Problem and contribution

At the time of the paper, CNNs dominated image recognition, and many attention-based vision models retained convolutional components or specialized spatial attention. The authors ask whether a fairly standard Transformer encoder can classify images effectively **without relying on a convolutional backbone**, particularly when pretrained on large image datasets. See [Introduction and related work](https://arxiv.org/html/2010.11929v2#S1).

Their main contribution is the **Vision Transformer (ViT)**: turn an image into a sequence of patch embeddings, feed it to a Transformer encoder and use its output for classification. The key empirical claim is that, with sufficiently large supervised pretraining, this simple design transfers well to image-recognition tasks and can compare favorably with strong CNN baselines. This is not a claim that the paper invented all patch-based image Transformers: the authors discuss related patch-based work in Section 2.

## 2. How ViT processes an image

For an image of height $H$, width $W$, and $C$ channels, choose square patches of side $P$. Assuming both dimensions are divisible by $P$, there are

$$
N=\frac{HW}{P^2}
$$

non-overlapping patches. For example, a $224\times224$ image with $16\times16$ patches gives $14\times14=196$ patches.

1. **Patch embedding:** flatten each $P\times P\times C$ patch and apply a learned linear projection to a $D$-dimensional vector.
2. **Sequence and positions:** prepend a learned classification token and add learned positional embeddings. The $196$-patch example has $197$ input tokens including the class token.
3. **Transformer encoder:** alternate multi-head self-attention and MLP blocks, with layer normalization and residual connections. Attention lets patch representations interact across the image.
4. **Classification:** use the final classification-token representation as the input to a classification head.
5. **Transfer:** supervised pretraining is followed by fine-tuning on a target task, replacing the head as needed. For a different fine-tuning resolution the paper interpolates the learned position embeddings.

Source: [Section 3 and Figure 1](https://arxiv.org/html/2010.11929v2#S3). The paper also studies **hybrids** that use CNN feature maps; distinguish those from the pure patch-based ViT.

## 3. Contrast with CNNs

CNNs build locality, spatial neighborhood structure and translation equivariance into their convolutional layers. ViT's attention can mix information globally and has fewer image-specific inductive biases. It is not entirely free of spatial assumptions: patch extraction imposes local grouping, and the paper uses 2D structure when interpolating positional embeddings for different resolutions. Learned positional embeddings also carry position information.

The paper's point is not that CNNs cannot use global information or that ViT is always superior. Rather, its experiments show a strong **data-scale tradeoff**: CNN biases can help with smaller datasets, while ViT benefits considerably from large-scale pretraining.

## 4. Main results and context

The authors compare ViT, modified ResNet/BiT and CNN–Transformer hybrids, with pretraining datasets ranging from ImageNet to ImageNet-21k and JFT-300M. They evaluate transfer on ImageNet, CIFAR, Pets, Flowers and VTAB; see [Sections 4.1–4.4](https://arxiv.org/html/2010.11929v2#S4).

- Their best model reports **88.55% ImageNet**, **94.55% CIFAR-100**, and **77.63% VTAB**; these are pretrained-and-transferred results, not training from scratch on the target alone. The introduction also reports 90.72% on ImageNet-ReaL.
- The paper reports that a ViT-L/16 pretrained on JFT-300M surpasses the compared BiT-L baseline on the reported tasks with lower measured pretraining compute; the larger ViT-H/14 improves some results further. This is a comparison under the paper's specific setups, not proof that every ViT is cheaper than every CNN.
- In its experiments with smaller pretraining datasets, ViT can underperform comparable CNNs and overfit more; the relative results improve with larger pretraining sets. Its public ImageNet-21k pretrained model also performs well.

The main pretraining experiments are **supervised**, not self-supervised. The paper additionally reports a preliminary masked-patch self-supervision experiment; do not describe that small exploration as the main method.

## What to retain for the review

Summarize the question, patch-to-Transformer pipeline, large-scale pretraining/transfer and one or two qualified findings. Keep the discussion short enough for the requested one-page review; save evaluation and limitations for their own sections.
