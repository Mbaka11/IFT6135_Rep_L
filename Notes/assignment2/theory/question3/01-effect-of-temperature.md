# Assignment 2 — Question 3.1: effect of temperature

**Goal:** compute attention weights and limiting context vectors, then derive how attention weights and the context respond to changes in a score. This note covers 3.1(a)–(c), not 3.2 or 3.3.

- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Next: 3.2 — Scaling the scores](02-scaling-scores.md)

**Status:** notes only; the Question 3 LaTeX answer is blank.

## 1. The setup

At one fixed decoder step, write the input states as $h_1,\ldots,h_n$ with $n=T_x$. Each $e_j$ measures the alignment with $h_j$. Temperature $\tau>0$ changes the softmax weights:

$$
\alpha_j(\tau)=\frac{\exp(e_j/\tau)}{\sum_{r=1}^n\exp(e_r/\tau)},
\qquad c=\sum_{j=1}^n\alpha_j h_j.
$$

Each $\alpha_j>0$ for finite scores and temperature, and $\sum_j\alpha_j=1$. Thus $c$ is a weighted average of the states, which may be vectors. For the derivatives below, hold $\tau$, the other scores and all $h_j$ fixed; the dependence of $c$ on a score is through the attention weights.

## 2. Part (a): evaluate $\alpha(1)$

For $e=[2,1,0]$ and $\tau=1$, the unnormalized weights are $[e^2,e,1]$, where $e$ here is Euler's number, not an alignment score. Hence

$$
\alpha(1)=\frac{[e^2,e,1]}{e^2+e+1}
\approx[0.6652,\;0.2447,\;0.0900].
$$

The weights sum to one (up to rounding); the first state contributes most because its score is largest. Its context at this temperature is

$$
c(1)=\frac{e^2h_1+eh_2+h_3}{e^2+e+1}.
$$

## 3. Temperature limits and context

### As $\tau\to0^+$

Divide numerator and denominator by $\exp(2/\tau)$:

$$
\alpha(\tau)=\frac{[1,\exp(-1/\tau),\exp(-2/\tau)]}{1+\exp(-1/\tau)+\exp(-2/\tau)}.
$$

Both exponentials with negative powers tend to zero. Therefore

$$
\lim_{\tau\to0^+}\alpha(\tau)=[1,0,0],
\qquad \lim_{\tau\to0^+}c(\tau)=h_1.
$$

The maximum score is uniquely at position 1. More generally, if $m$ scores tie for the maximum, the limit distributes weight $1/m$ over those positions, not just one; the limiting context is their average.

### As $\tau\to\infty$

For each fixed finite score, $e_j/\tau\to0$, so $\exp(e_j/\tau)\to1$. Therefore

$$
\lim_{\tau\to\infty}\alpha(\tau)=\left[\frac13,\frac13,\frac13\right],
\qquad \lim_{\tau\to\infty}c(\tau)=\frac{h_1+h_2+h_3}{3}.
$$

For fixed scores, a lower temperature sharpens the distribution and a higher temperature flattens it. The actual weights also depend on the score differences, not on temperature alone; see [3.2](02-scaling-scores.md).

## 4. Part (b): derivative of a weight with respect to a score

Let $Z=\sum_r\exp(e_r/\tau)$. For the Kronecker delta, $\delta_{jk}=1$ if $j=k$ and $0$ otherwise. Holding $\tau$ fixed,

$$
\frac{\partial}{\partial e_k}\exp(e_j/\tau)
=\frac{\delta_{jk}}{\tau}\exp(e_j/\tau),
\qquad
\frac{\partial Z}{\partial e_k}=\frac{1}{\tau}\exp(e_k/\tau).
$$

The quotient rule gives

$$
\begin{aligned}
\frac{\partial\alpha_j}{\partial e_k}
&=\frac{1}{\tau}\left(
\frac{\delta_{jk}\exp(e_j/\tau)}{Z}
-\frac{\exp(e_j/\tau)\exp(e_k/\tau)}{Z^2}\right)\\
&=\boxed{\frac{1}{\tau}\alpha_j(\delta_{jk}-\alpha_k)}.
\end{aligned}
$$

For $j=k$ this is $\alpha_k(1-\alpha_k)/\tau\ge0$: increasing $e_k$ increases its own weight. For $j\ne k$ it is $-\alpha_j\alpha_k/\tau\le0$: the other weights decrease. These changes sum to zero, consistent with $\sum_j\alpha_j=1$.

## 5. Part (c): derivative of the context

Differentiate $c=\sum_j\alpha_jh_j$ componentwise, keeping the states fixed:

$$
\begin{aligned}
\frac{\partial c}{\partial e_k}
&=\sum_j h_j\frac{\partial\alpha_j}{\partial e_k}\\
&=\frac{1}{\tau}\sum_j h_j\alpha_j(\delta_{jk}-\alpha_k)\\
&=\frac{1}{\tau}\left(\alpha_kh_k-\alpha_k\sum_j\alpha_jh_j\right)\\
&=\boxed{\frac{\alpha_k}{\tau}(h_k-c)}.
\end{aligned}
$$

For a small positive change $\Delta e_k$,

$$
\Delta c\approx\frac{\alpha_k}{\tau}(h_k-c)\,\Delta e_k.
$$

So the context moves locally **toward $h_k$**, along the vector from the current context to that state. If $h_k=c$, the first-order change is zero. The factor $\alpha_k/\tau$ is not by itself a guarantee that lower temperatures cause larger changes, because $\alpha_k$ and $c$ also depend on temperature.

## What to retain for a concise answer

Give the exact and numerical weights at $\tau=1$, both limiting weights and contexts, and the quotient-rule steps. For (c), substitute the derivative into the context sum and interpret $h_k-c$ as a direction.
