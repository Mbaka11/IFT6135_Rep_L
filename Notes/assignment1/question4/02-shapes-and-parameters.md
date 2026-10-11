# Assignment 1 — Question 4.2: output shapes and parameter counts

**Goal:** propagate the input shape through the CNN and count its trainable parameters.

- [Assignment statement](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)
- [Previous: 4.1 — Equivalent convolution](01-equivalent-convolution.md)
- [Next: 4.3 — Removable biases](03-biases-before-batchnorm.md)

**Notes only: the LaTeX answer remains blank.**

## Quick reminders

- **Channels:** parallel grids of values. RGB input has 3 channels; `Conv(4,8)` uses $4\times4$ kernels spanning all input channels and produces 8 output feature maps.
- **Padding:** adds border positions around the input, usually filled with zeros for convolution. Padding 1 adds one position on each side, so a dimension $N$ becomes $N+2$ before placing windows.
- **Max pooling:** keeps the largest value in each window, separately per channel; it does not learn the window's coefficients.
- **Trainable parameters versus outputs:** parameters are numbers adjusted during learning; output entries are values computed from the input. More output positions do not necessarily mean more parameters.

## 1. Settle the conventions first

The assignment says “padding 1 and stride 1 where appropriate,” without explicitly fixing the pooling settings. This note uses **padding 1 and stride 1 for pooling in the main table**, then gives a short alternative using the course's non-overlapping pooling convention ([slides, physical PDF page 47](../../../Slides/Lecture_3_convnets.pdf#page=47)). Confirm the intended convention before submitting: the resulting dimensions differ substantially.

Shapes are written as $(H,W,C)$, omitting the batch dimension. For the last two rows, $(1,1,D)$ is a way to keep the requested three-entry table format; flattening and FC actually produce $D$-component feature vectors, not spatial images.

## 2. The three formulas we need

For a convolution or pooling window, one spatial dimension becomes

$$
\boxed{N_{\mathrm{out}}=
\left\lfloor\frac{N_{\mathrm{in}}+2P-K}{S}\right\rfloor+1.}
$$

Here $K$ is the window size, $P$ is padding on each side, and $S$ is stride. Apply it separately to height and width. The floor rounds down if the last window does not fit completely.

For convolution, each output channel has $K^2C_{\mathrm{in}}$ weights and one bias:

$$
\boxed{\#\text{Conv parameters}=(K^2C_{\mathrm{in}}+1)C_{\mathrm{out}}.}
$$

The same kernel weights are reused at every spatial position, so height and width do not multiply this parameter count.

**FC** means fully connected, or dense: every output uses all input features. Each of the $D_{\mathrm{out}}$ outputs has one weight per input feature and one bias:

$$
\boxed{\#\text{FC parameters}=(D_{\mathrm{in}}+1)D_{\mathrm{out}}.}
$$

**BatchNorm** normalizes values within each channel, then applies a learned scale and shift. It preserves shape and learns **two numbers per channel**, $\gamma_c$ and $\beta_c$: $2C$ parameters. Running means/variances are statistics, not trainable parameters. ReLU, pooling, and flattening have no trainable parameters.

## 3. Follow the layers in order

### First convolution: Conv(4,8), stride 2

Start with $(64,64,3)$, and use $K=4$, $P=1$, $S=2$:

$$
N_{\mathrm{out}}
=\left\lfloor\frac{64+2-4}{2}\right\rfloor+1
=31+1=32.
$$

There are 8 output channels, so the output is $(32,32,8)$. Its count is

$$
(4^2\cdot3+1)\cdot8=(48+1)\cdot8=392.
$$

The following BatchNorm has $2\cdot8=16$ parameters. It and ReLU keep the shape unchanged.

### MaxPooling(2)

For the main convention, $K=2$, $P=1$, $S=1$:

$$
N_{\mathrm{out}}=\frac{32+2-2}{1}+1=33.
$$

Output: $(33,33,8)$, with 0 parameters. Pooling preserves the channel count. The increase from 32 to 33 is not a mistake: padding extends the set of possible windows, and stride 1 does not perform downsampling.

### Second convolution: Conv(3,16)

Use $K=3$, $P=1$, $S=1$, with **8 input channels** from the preceding layer:

$$
N_{\mathrm{out}}=\frac{33+2-3}{1}+1=33,
$$

$$
\#\text{parameters}=(3^2\cdot8+1)\cdot16
=(72+1)\cdot16=1168.
$$

Output: $(33,33,16)$. The following BatchNorm has $2\cdot16=32$ parameters. BatchNorm and ReLU again preserve the shape.

### MaxPooling(4)

Use $K=4$, $P=1$, $S=1$:

$$
N_{\mathrm{out}}=\frac{33+2-4}{1}+1=32.
$$

Output: $(32,32,16)$, with 0 parameters.

### Flatten and FC(10)

Flatten keeps every entry but removes the spatial organization:

$$
D_{\mathrm{in}}=32\cdot32\cdot16=16384.
$$

It has no parameters. FC maps those 16384 features to 10 outputs:

$$
\#\text{parameters}=(16384+1)\cdot10=163850.
$$

## 4. Completed table: stated defaults applied to pooling

| Layer | Output dimensions $(H,W,C)$ | Trainable parameters |
| --- | --- | ---: |
| Input | $(64,64,3)$ | 0 |
| Conv(4,8), stride 2 | $(32,32,8)$ | 392 |
| BatchNorm | $(32,32,8)$ | 16 |
| ReLU | $(32,32,8)$ | 0 |
| MaxPooling(2) | $(33,33,8)$ | 0 |
| Conv(3,16) | $(33,33,16)$ | 1168 |
| BatchNorm | $(33,33,16)$ | 32 |
| ReLU | $(33,33,16)$ | 0 |
| MaxPooling(4) | $(32,32,16)$ | 0 |
| Reshape | $(1,1,16384)$ | 0 |
| FC(10) | $(1,1,10)$ | 163850 |
| **Total** | — | **165458** |

The convolution biases are included here because 4.2 asks for the layers as specified. Part 4.3 explains which ones can subsequently be removed.

## 5. Short comparison: non-overlapping pooling

If pooling instead has **no padding and stride equal to its window size**, only the downstream shapes and FC count change:

$$
\text{Pool}(2):\quad
\left\lfloor\frac{32-2}{2}\right\rfloor+1=16,
$$

$$
\text{Pool}(4):\quad
\left\lfloor\frac{16-4}{4}\right\rfloor+1=4.
$$

| Affected layer(s) | Alternative output | Parameters |
| --- | --- | ---: |
| MaxPooling(2) | $(16,16,8)$ | 0 |
| Conv(3,16), then BatchNorm/ReLU | $(16,16,16)$ | 1168, then 32/0 |
| MaxPooling(4) | $(4,4,16)$ | 0 |
| Reshape | $(1,1,256)$ | 0 |
| FC(10) | $(1,1,10)$ | $(256+1)\cdot10=2570$ |

The alternative total is $392+16+1168+32+2570=4178$. Do not combine shapes from one convention with the parameter count from the other.
