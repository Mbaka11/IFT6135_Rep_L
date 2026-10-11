# Assignment 1 — Question 4.1: combine the three layers

**Goal:** replace the two convolutions and the average-pooling layer with one exact layer, and give its weights.

- [Assignment statement](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)
- [Next: 4.2 — Shapes and parameter counts](02-shapes-and-parameters.md)

**Notes only: the LaTeX answer remains blank.** Use valid/no-padding layers and the usual CNN convention of unflipped kernels. The padding defaults stated in 4.2 are separate from this part.

## Quick reminders

- **Kernel/window:** a kernel is the small array of convolution weights applied to a local input patch. A $3\times3$ kernel covers 3 rows and 3 columns per input channel.
- **Stride:** how far the window moves between neighboring output positions, measured in the current layer's input. With no padding, stride 3 gives window starts at $0,3,6,\ldots$.
- **Pooling:** summarizes a window separately in each channel. Average pooling takes the mean; max pooling takes the largest value. For a $2\times2$ window containing $1,3,5,7$, the average is 4 and the maximum is 7. Window size determines what is combined; stride determines where the next window starts.

## 1. Why can these layers be combined?

The sequence is

$$
\operatorname{Conv}(3,\text{stride }3)
\;\longrightarrow\;
\operatorname{AvgPool}(2,\text{stride }2)
\;\longrightarrow\;
\operatorname{Conv}(2,\text{stride }2).
$$

Convolution takes weighted sums. Average pooling takes a fixed weighted sum: each entry of a $2\times2$ window has weight $1/4$. With no intervening activation, composing these operations still gives a weighted sum of input pixels. Biases can also be combined into one bias.

This would not generally work with ReLU or max pooling between the convolutions, because they are nonlinear.

## 2. Track which input pixel contributes to one output

First use one channel. Let:

- $A_{ij}$ be the first $3\times3$ kernel, with $i,j\in\{0,1,2\}$;
- $B_{ab}$ be the last $2\times2$ kernel, with $a,b\in\{0,1\}$;
- $u,v\in\{0,1\}$ index the average-pooling window;
- $(p,q)$ be a final output position. All spatial indices start at zero.

Follow the **row** index backwards through the layers:

| Layer being traced | Selected position |
| --- | --- |
| Last convolution selects a pooled row | $2p+a$ |
| Pooling selects a first-convolution row | $2(2p+a)+u$ |
| First convolution selects an input row | $3[2(2p+a)+u]+i$ |

Expand the last expression:

$$
3[2(2p+a)+u]+i=12p+6a+3u+i.
$$

The column calculation is identical:

$$
12q+6b+3v+j.
$$

Therefore:

1. Increasing $p$ or $q$ by 1 moves the input starting position by **12**: the effective stride is $3\cdot2\cdot2=12$.
2. The row offset $6a+3u+i$ ranges from 0 to $6+3+2=11$: one output can depend on **12 rows**. Likewise it can depend on 12 columns.

The replacement is thus a **$12\times12$ convolution with stride 12**.

## 3. Multiply the weights along that path

The input pixel passes through three multipliers:

$$
A_{ij}\quad\times\quad\frac14\quad\times\quad B_{ab}.
$$

Match that coefficient to the corresponding position of the fused kernel $K$:

$$
\boxed{
K_{6a+3u+i,\;6b+3v+j}=\frac14B_{ab}A_{ij}.
}
$$

Every row offset from 0 to 11 is represented exactly once by the allowed $(a,u,i)$ values, and similarly for the columns. So this formula specifies all 144 kernel entries.

**A visual way to arrange them:** repeat $A$ in a $2\times2$ block to form a $6\times6$ matrix $T$, then scale four copies using the entries of $B$:

$$
T=\begin{bmatrix}A&A\\A&A\end{bmatrix},
\qquad
K=\frac14\begin{bmatrix}
B_{00}T&B_{01}T\\
B_{10}T&B_{11}T
\end{bmatrix}.
$$

For example, if every entry of $A$ and $B$ is 1, every entry of $K$ is $1/4$. A $12\times12$ patch of ones then produces $144/4=36$: the original layers give $9\to9\to36$, so the two calculations agree.

## 4. Include channels and biases when present

**Reminder:** a channel is one grid of values, such as an RGB color plane or a learned feature map. A convolution bias adds the same learned constant at every position of an output channel.

With input channel $c$, intermediate channel $m$, and output channel $o$, sum over the intermediate paths:

$$
\boxed{
K_{o,c,6a+3u+i,\;6b+3v+j}
=\frac14\sum_m B_{o,m,a,b}A_{m,c,i,j}.
}
$$

The fused convolution retains the original input-channel count and the last convolution's output-channel count.

If the convolution biases are $b^{(1)}_m$ and $b^{(2)}_o$, averaging four copies of $b^{(1)}_m$ leaves it unchanged. The last convolution then weights that constant, giving

$$
b^{\mathrm{eff}}_o
=b^{(2)}_o+\sum_m b^{(1)}_m\sum_{a,b=0}^1 B_{o,m,a,b}.
$$

For one channel, this reduces to $b^{\mathrm{eff}}=b^{(2)}+b^{(1)}\sum_{a,b}B_{ab}$. With no biases, set them to zero.

## What to submit later

State the effective layer, show how the input index yields kernel size 12 and stride 12, and give the coefficient formula. The fused weights are computed from the original kernels; their 144 entries are not all independent parameters of the original network.
