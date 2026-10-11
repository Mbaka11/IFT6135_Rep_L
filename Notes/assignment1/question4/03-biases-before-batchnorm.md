# Assignment 1 — Question 4.3: which parameters can be removed?

**Answer:** the biases of both convolutional layers, because each convolution is immediately followed by BatchNorm.

- [Assignment statement](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)
- [Previous: 4.2 — Shapes and parameter counts](02-shapes-and-parameters.md)
- [BatchNorm course slide, physical PDF page 66](../../../Slides/Lecture_3_convnets.pdf#page=66)

**Notes only: the LaTeX answer remains blank.**

## Quick reminders

- **Bias:** a learned constant added to a layer's weighted sum. In a convolution, one bias is shared by all positions of an output channel.
- **Mean and variance:** the mean is the average; variance is the average squared deviation from that mean, measuring spread.
- **Normalization:** subtract the mean to center values around zero, then divide by the standard deviation (the square root of variance) to rescale their spread. BatchNorm adds learned scale/shift afterward.
- **Training versus inference:** during training, BatchNorm computes statistics from the current mini-batch (a group of examples). At inference (prediction), it uses stored statistics. This distinction matters when deleting a trained model's bias.

## 1. Write what happens in one output channel

Let $z_t$ be the convolution output **without** bias, and let $b$ be that channel's bias. The index $t$ runs over the examples and spatial positions used to compute the channel's BatchNorm statistics.

The biased convolution produces

$$
x_t=z_t+b.
$$

The same $b$ is added at every position in this channel. Training-mode BatchNorm outputs

$$
\operatorname{BN}(x_t)
=\gamma\frac{x_t-\mu_x}{\sqrt{\sigma_x^2+\epsilon}}+\beta,
$$

where $\mu_x$ and $\sigma_x^2$ are the channel's mean and variance, $\epsilon>0$ is a stability constant, and $\gamma,\beta$ are learned scale/shift parameters.

## 2. Show the mean subtraction cancels the bias

If there are $M$ values in the normalization set,

$$
\begin{aligned}
\mu_x
&=\frac1M\sum_{t=1}^M(z_t+b)\\
&=\frac1M\sum_{t=1}^M z_t+\frac{Mb}{M}\\
&=\mu_z+b.
\end{aligned}
$$

Therefore,

$$
x_t-\mu_x=(z_t+b)-(\mu_z+b)=z_t-\mu_z.
$$

The variance is unchanged too, because it uses these same centered values:

$$
\sigma_x^2=\frac1M\sum_t(x_t-\mu_x)^2
=\frac1M\sum_t(z_t-\mu_z)^2=\sigma_z^2.
$$

Substitute both results into BatchNorm:

$$
\boxed{
\operatorname{BN}(z_t+b)
=\gamma\frac{z_t-\mu_z}{\sqrt{\sigma_z^2+\epsilon}}+\beta
=\operatorname{BN}(z_t).
}
$$

So the convolution bias does not change the normalized training output. It is redundant, and we can omit it when constructing this architecture.

**Small check:** with $z=[1,3]$ and $b=5$, the biased values are $[6,8]$. Their means are 2 and 7 respectively, so both centered lists are $[-1,1]$. Both also have variance 1.

## 3. Apply this to the assignment

| Layer | Removable biases | Parameters before → after |
| --- | ---: | ---: |
| Conv(4,8), immediately before BatchNorm | 8 | $392\to384$ |
| Conv(3,16), immediately before BatchNorm | 16 | $1168\to1152$ |
| **Total removed** | **24** | — |

This reduction is independent of the pooling convention:

- Main table from 4.2: $165458-24=165434$ trainable parameters.
- Non-overlapping-pooling alternative: $4178-24=4154$.

Keep the convolution weights, BatchNorm's $\gamma,\beta$, and the final FC biases. In particular, $\beta$ supplies a learned shift **after** normalization; it is not canceled by that normalization. The FC layer has no following BatchNorm to cancel its bias.

## 4. A brief inference caveat

At inference, BatchNorm uses stored statistics instead of recomputing the mean. If removing bias from an **already trained** model, deleting it while keeping the stored mean unchanged is not generally identical.

To preserve the same output, also replace the stored mean by

$$
\mu_{\mathrm{run}}'=\mu_{\mathrm{run}}-b.
$$

Then $z_t-\mu_{\mathrm{run}}'=z_t+b-\mu_{\mathrm{run}}$, exactly matching the old centered input. When designing the network from the start, use bias-free convolutions before BatchNorm and learn their statistics normally.

## What to submit later

Identify the two convolution biases, state that BatchNorm's mean subtraction cancels them, and briefly show $(z+b)-(\mu_z+b)=z-\mu_z$. The model has 24 such redundant parameters.
