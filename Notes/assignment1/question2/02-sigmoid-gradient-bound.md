# Assignment 1 — Question 2.2: sigmoid derivatives and vanishing gradients

**Scope:** main Question 2, part (2). This single note contains all three sub-subquestions:

- **(a)** Find the maximum absolute sigmoid derivative.
- **(b)** Derive a bound on the gradient at the first hidden pre-activation.
- **(c)** Identify when that bound decreases exponentially with depth.

- [Previous: 2.1 — Identity activations](01-linear-activations.md)
- [Next: 2.3 — ReLU, LeakyReLU, and Maxout](03-relu-and-variants.md)
- [Earlier sigmoid derivative steps: Question 1.2](../question1/02-cross-entropy-derivative.md)
- [Assignment statement and answer template](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)

These are study notes. **LaTeX status: Question 2(2)(a), (b), and (c) are all filled.** This question uses the standard mathematical sigmoid; the numerical approximation $e\approx3$ was a convention specifically stated for Question 1.

## 1. Notation and the depth convention

Use $\mathcal L$ for the loss and $L$ for the number of learned layers. The statement sometimes writes the loss as $L$ in a derivative, but that is not the depth variable.

There are $L-1$ hidden layers and one final logit layer. The first hidden pre-activation is scalar even when the input is a vector:

$$
a^{(1)}=(\mathbf w^{(1)})^\top\mathbf x+b^{(1)},
\qquad h^{(1)}=\sigma(a^{(1)}).
$$

After that, the width-one hidden transitions are scalar:

$$
a^{(l)}=w^{(l)}h^{(l-1)}+b^{(l)},
\qquad h^{(l)}=\sigma(a^{(l)}),
\qquad 2\leq l\leq L-1,
$$

while the final layer produces the output logit:

$$
a^{(L)}=w^{(L)}h^{(L-1)}+b^{(L)}.
$$

There is **no hidden sigmoid after the output logit** in the question's architecture. The output softmax and cross-entropy are already accounted for in the incoming output-loss gradient.

Define the scalar error signals

$$
\delta^{(l)}:=\frac{\partial\mathcal L}{\partial a^{(l)}},
\qquad G:=\left|\frac{\partial\mathcal L}{\partial a^{(L)}}\right|.
$$

The multiclass head has vector logits in the actual classifier. The scalar notation above makes the chain factors explicit; Section 3's final remark explains the corresponding vector-head bound without treating the classifier as a one-class softmax.

## 2. Part (a) — Maximum absolute derivative

### Step 1: recall the derivative

For

$$
g(a)=\sigma(a)=\frac{1}{1+e^{-a}},
$$

the derivative is

$$
g'(a)=\frac{e^{-a}}{(1+e^{-a})^2}
=\sigma(a)(1-\sigma(a)).
$$

The reciprocal/exponential chain-rule steps were worked out in Question 1.2. Here we need to find the maximum, not differentiate the loss again.

### Step 2: express the derivative as a quadratic

Set $p=\sigma(a)$, so $0<p<1$. Then

$$
g'(a)=p(1-p)=p-p^2.
$$

To rewrite this expression, we **complete the square**. Start by expanding:

$$
\begin{aligned}
\left(p-\frac12\right)^2
&=p^2-2p\left(\frac12\right)+\frac14\\
&=p^2-p+\frac14.
\end{aligned}
$$

We choose $1/2$ because the middle term $-2p(1/2)$ is exactly $-p$.

Subtract $1/4$ from both sides:

$$
p^2-p=\left(p-\frac12\right)^2-\frac14.
$$

Now negate both sides. This changes $p^2-p$ into $p-p^2$:

$$
\begin{aligned}
g'(a)=p-p^2
&=-\left[\left(p-\frac12\right)^2-\frac14\right]\\
&=\boxed{\frac14-\left(p-\frac12\right)^2}.
\end{aligned}
$$

A square is nonnegative, so

$$
0<g'(a)\leq\frac14
$$

for finite $a$. The derivative is positive, hence taking its absolute value does not change it.

### Step 3: find where equality holds

The square is zero precisely when $p=1/2$, which occurs at $a=0$. Therefore,

$$
\boxed{\max_{a\in\mathbb R}|g'(a)|=\frac14,
\qquad\text{attained at }a=0.}
$$

This is a **maximum**, not the derivative at every input. When the sigmoid saturates near 0 or 1, its derivative approaches zero and can be much smaller than $1/4$.

## 3. Part (b) — Expand the chain to the first layer

### Step 1: derive one backward transition

For any hidden layer $l$,

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial a^{(l)}}
&=\frac{\partial\mathcal L}{\partial a^{(l+1)}}
  \frac{\partial a^{(l+1)}}{\partial h^{(l)}}
  \frac{\partial h^{(l)}}{\partial a^{(l)}}\\
&=\frac{\partial\mathcal L}{\partial a^{(l+1)}}
  w^{(l+1)}\sigma'(a^{(l)}).
\end{aligned}
$$

This is the explicit chain-rule presentation: incoming derivative, local affine derivative, then local activation derivative.

The bias contributes no multiplicative factor here because $\partial a^{(l+1)}/\partial h^{(l)}=w^{(l+1)}$.

### Step 2: see the pattern in a three-layer example

With two sigmoid hidden layers and one output-logit layer,

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial a^{(1)}}
&=\frac{\partial\mathcal L}{\partial a^{(3)}}
  \frac{\partial a^{(3)}}{\partial h^{(2)}}
  \frac{\partial h^{(2)}}{\partial a^{(2)}}
  \frac{\partial a^{(2)}}{\partial h^{(1)}}
  \frac{\partial h^{(1)}}{\partial a^{(1)}}\\
&=\frac{\partial\mathcal L}{\partial a^{(3)}}
  w^{(3)}\sigma'(a^{(2)})
  w^{(2)}\sigma'(a^{(1)}).
\end{aligned}
$$

Notice what is absent:

- No $w^{(1)}$: we stop at $a^{(1)}$, rather than continuing back to the input.
- No $\sigma'(a^{(3)})$: the output layer is a logit layer, not a sigmoid hidden layer.
- No extra softmax derivative: it is already included in the starting loss gradient.

### Step 3: write the general product

Repeating the backward transition yields

$$
\boxed{
\frac{\partial\mathcal L}{\partial a^{(1)}}
=\frac{\partial\mathcal L}{\partial a^{(L)}}
 \prod_{l=1}^{L-1}\left[w^{(l+1)}\sigma'(a^{(l)})\right].
}
$$

There are $L-1$ weight factors, $w^{(2)},\ldots,w^{(L)}$, and $L-1$ sigmoid-derivative factors. These factors are scalars, so their multiplication order does not introduce matrix-dimension issues.

### Step 4: take absolute values and bound each factor

Using the assumption $|w^{(l)}|\leq W_{\max}$ and part (a),

$$
\begin{aligned}
\left|\frac{\partial\mathcal L}{\partial a^{(1)}}\right|
&=G\prod_{l=1}^{L-1}
 \left(|w^{(l+1)}|\,|\sigma'(a^{(l)})|\right)\\
&\leq G\prod_{l=1}^{L-1}\left(W_{\max}\cdot\frac14\right)\\
&=\boxed{G\left(\frac{W_{\max}}4\right)^{L-1}}.
\end{aligned}
$$

This is an **upper bound**, not generally the exact gradient magnitude. Different weights can be smaller than $W_{\max}$, and different sigmoid derivatives can be smaller than $1/4$.

### Why the exponent is $L-1$, not $L$

The statement counts $L$ learned layers **including the output-logit layer**. The backward path from the output logits to the first hidden pre-activation crosses $L-1$ hidden activations.

If another source instead defines $L$ as the number of hidden layers, with an additional output layer, the exponent would be $L$. That is a different indexing convention, not a different chain rule.

### Multiclass-head clarification

With hidden width one, the last hidden activation is scalar, but the actual output logits form a vector $\mathbf z\in\mathbb R^K$:

$$
\mathbf z=\mathbf v h^{(L-1)}+\mathbf b^{(L)},
\qquad \mathbf v\in\mathbb R^K.
$$

Let $\mathbf g_{\mathrm{out}}=\nabla_{\mathbf z}\mathcal L$. At the last hidden neuron,

$$
\frac{\partial\mathcal L}{\partial a^{(L-1)}}
=\sigma'(a^{(L-1)})\mathbf v^\top\mathbf g_{\mathrm{out}}.
$$

Cauchy–Schwarz gives

$$
\left|\frac{\partial\mathcal L}{\partial a^{(L-1)}}\right|
\leq\frac14\|\mathbf v\|_2\|\mathbf g_{\mathrm{out}}\|_2.
$$

Using $\|\mathbf v\|_2\leq W_{\max}$ for the output weight matrix and the scalar bounds for the earlier layers leads to the same form:

$$
\boxed{
\left|\frac{\partial\mathcal L}{\partial a^{(1)}}\right|
\leq\|\nabla_{\mathbf z}\mathcal L\|_2
\left(\frac{W_{\max}}4\right)^{L-1}.
}
$$

The output-layer factor must be a bound on the appropriate matrix/vector norm, not merely on each output entry without accounting for the number of classes. For a $K\times1$ matrix, its Euclidean operator norm is $\|\mathbf v\|_2$.

This explains how the requested scalar-chain argument fits the multiclass setup. It does not require adding a hidden sigmoid to the output.

## 4. Part (c) — When does the bound vanish exponentially?

Set

$$
q:=\frac{W_{\max}}4.
$$

The bound has the form $Gq^{L-1}$. For a bounded incoming output gradient, it tends to zero exponentially when $0\leq q<1$, equivalently,

$$
\boxed{W_{\max}<4.}
$$

Here $W_{\max}\geq0$ because it bounds absolute weights. If $W_{\max}=0$, the backward gradient is already zero for a network with at least one hidden layer.

For the actual softmax/cross-entropy head with a one-hot target,

$$
\nabla_{\mathbf z}\mathcal L=\hat{\mathbf y}-\mathbf y,
\qquad \|\hat{\mathbf y}-\mathbf y\|_2\leq\sqrt2.
$$

Thus the output gradient is uniformly bounded for a single-sample loss, so it does not cancel the exponential decay by growing arbitrarily with depth.

### A concrete scale example

If $W_{\max}=2$, then $q=1/2$:

$$
L=6:\quad |\delta^{(1)}|\leq\frac{G}{32},
\qquad
L=21:\quad |\delta^{(1)}|\leq\frac{G}{2^{20}}
\approx9.54\times10^{-7}G.
$$

Each additional layer multiplies the worst-case bound by another $1/2$. That repeated multiplication is what “exponential with depth” means.

### What the bound does and does not prove

| Weight bound | Consequence of this upper bound |
| --- | --- |
| $W_{\max}<4$ | Forces exponential decay for bounded output gradients. |
| $W_{\max}=4$ | Gives a depth-independent upper bound; does not force decay. |
| $W_{\max}>4$ | Gives a growing upper bound; does not prove that the actual gradient grows. |

The condition $W_{\max}<4$ is a **sufficient uniform condition**, not a necessary condition for every network to have small gradients. Saturated sigmoids or small actual weights can still cause vanishing gradients when $W_{\max}\geq4$.

For example, a particular network with $w^{(l)}=5$ and hidden sigmoid derivatives below $0.01$ has backward factors below $0.05$, even though the weight bound exceeds 4. The products can still shrink.

## 5. Target variable and common mistakes

- The question asks for $\partial\mathcal L/\partial a^{(1)}$, not directly for an input or weight gradient.
- An input gradient would involve $w^{(1)}$ as an additional factor in the scalar case.
- A first-layer weight gradient is $\delta^{(1)}h^{(0)}$; it introduces the input, not another sigmoid derivative.
- Use absolute values before multiplying the magnitude bounds; weights can be negative.
- Do not replace every sigmoid derivative by $1/4$ and call that the exact gradient.
- Keep the output softmax/cross-entropy inside the incoming output-loss gradient.
- A growing upper bound is not a lower bound and does not guarantee exploding gradients.

## 6. What belongs in the eventual LaTeX answer?

- **(a):** $g'(a)=\sigma(a)(1-\sigma(a))\leq1/4$, with equality at $a=0$.
- **(b):** show the local chain rule, its $L-1$-factor product, and the bound $G(W_{\max}/4)^{L-1}$, identifying the output-loss gradient. Use a norm for the multiclass head.
- **(c):** state $W_{\max}<4$ and explain that the bound then decays exponentially for bounded output gradients.

The examples, indexing reminders, and detailed multiclass clarification are learning material. **Question 2(2)(a), (b), and (c) are all filled in LaTeX.** The maximum-derivative answer includes the expanded completing-the-square steps.
