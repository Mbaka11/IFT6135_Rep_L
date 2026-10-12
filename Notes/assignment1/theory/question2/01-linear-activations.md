# Assignment 1 — Question 2.1: identity activations and affine composition

**Scope:** main Question 2, part (1). This single note contains both sub-subquestions:

- **(a)** Show that two consecutive affine layers collapse into one affine layer.
- **(b)** Explain why adding depth without nonlinear hidden activations does not increase representational capacity.

- [Next: 2.2 — Sigmoid and the gradient bound](02-sigmoid-gradient-bound.md)
- [Assignment statement and answer template](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)

These are study notes. **All three parts of Question 2 are fully filled in LaTeX.**

## 1. Setup: what does an identity activation do?

Write the input as $\mathbf h^{(0)}=\mathbf x$. A hidden layer first computes

$$
\mathbf a^{(l)}=W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)}
$$

and then applies the activation. With $g(a)=a$, the activation changes nothing:

$$
\mathbf h^{(l)}=g(\mathbf a^{(l)})
=\mathbf a^{(l)}
=W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)}.
$$

A useful terminology distinction:

- $W\mathbf x$ is a **linear** map.
- $W\mathbf x+\mathbf b$ is an **affine** map; it need not send zero to zero.
- The identity activation is linear, but the whole layer is generally affine because it includes a bias.

The final layer produces logits, not another hidden activation. Softmax is applied afterward. The input is not counted as a learned layer: $L$ includes the output-logit layer.

## 2. Part (a) — Compose two consecutive layers

### Step 1: write both layers separately

For layers $l$ and $l+1$,

$$
\mathbf h^{(l)}
=W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)},
$$

$$
\mathbf h^{(l+1)}
=W^{(l+1)}\mathbf h^{(l)}+\mathbf b^{(l+1)}.
$$

If the second layer is the output layer, call its result the logit vector instead; the same algebra applies.

### Step 2: substitute the first expression into the second

$$
\mathbf h^{(l+1)}
=W^{(l+1)}\left(W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)}\right)
+\mathbf b^{(l+1)}.
$$

### Step 3: distribute the second weight matrix

$$
\mathbf h^{(l+1)}
=W^{(l+1)}W^{(l)}\mathbf h^{(l-1)}
+W^{(l+1)}\mathbf b^{(l)}+\mathbf b^{(l+1)}.
$$

The result has the form “one matrix times the input, plus one bias.” Define

$$
\boxed{
\widetilde W=W^{(l+1)}W^{(l)},
\qquad
\widetilde{\mathbf b}=W^{(l+1)}\mathbf b^{(l)}+\mathbf b^{(l+1)}.
}
$$

Then

$$
\boxed{\mathbf h^{(l+1)}
=\widetilde W\mathbf h^{(l-1)}+\widetilde{\mathbf b}.}
$$

That proves the composition is a single affine transformation.

### Why do we multiply the earlier bias by the later matrix?

The first bias is added **before** the second layer. Everything produced by the first layer, including its bias contribution, passes through $W^{(l+1)}$.

It is therefore generally incorrect to write $\widetilde{\mathbf b}=\mathbf b^{(l)}+\mathbf b^{(l+1)}$. Those two biases may not even have the same dimension.

### Dimension check

Let the three relevant widths be $d_{l-1}$, $d_l$, and $d_{l+1}$.

| Quantity | Shape |
| --- | --- |
| $\mathbf h^{(l-1)}$ | $d_{l-1}\times1$ |
| $W^{(l)}$ | $d_l\times d_{l-1}$ |
| $\mathbf b^{(l)}$ | $d_l\times1$ |
| $W^{(l+1)}$ | $d_{l+1}\times d_l$ |
| $\mathbf b^{(l+1)}$ | $d_{l+1}\times1$ |
| $\widetilde W$ | $d_{l+1}\times d_{l-1}$ |
| $\widetilde{\mathbf b}$ | $d_{l+1}\times1$ |

The matrix order is important: $W^{(l+1)}W^{(l)}$ follows the order in which the input travels through the layers. Reversing the product may be dimensionally impossible and is not generally equivalent.

### Tiny numerical example

Take two scalar layers:

$$
h^{(1)}=2x+1,
\qquad h^{(2)}=3h^{(1)}-4.
$$

Substituting,

$$
h^{(2)}=3(2x+1)-4=6x-1.
$$

The effective weight is $3\cdot2=6$ and the effective bias is $3\cdot1-4=-1$. At $x=2$, both the two-layer calculation and the collapsed layer give 11.

## 3. Part (b) — Collapse the whole network

### Step 1: repeat the result from part (a)

After merging two layers, we still have an affine map. Merge that map with the next affine layer, then keep going until the output logits.

For example, three layers give

$$
\begin{aligned}
\mathbf z
&=W^{(3)}\left(W^{(2)}(W^{(1)}\mathbf x+\mathbf b^{(1)})
+\mathbf b^{(2)}\right)+\mathbf b^{(3)}\\
&=W^{(3)}W^{(2)}W^{(1)}\mathbf x
+W^{(3)}W^{(2)}\mathbf b^{(1)}
+W^{(3)}\mathbf b^{(2)}+\mathbf b^{(3)}.
\end{aligned}
$$

For $L$ learned layers, define

$$
\boxed{W_{\mathrm{eff}}=W^{(L)}W^{(L-1)}\cdots W^{(1)}}
$$

and

$$
\boxed{
\mathbf b_{\mathrm{eff}}
=\sum_{k=1}^{L}
\left(W^{(L)}W^{(L-1)}\cdots W^{(k+1)}\right)\mathbf b^{(k)}.
}
$$

For $k=L$, the product in parentheses is the identity, so the final term is simply $\mathbf b^{(L)}$. Every earlier bias is multiplied by all the weight matrices that come after it.

Consequently,

$$
\boxed{\mathbf z=W_{\mathrm{eff}}\mathbf x+\mathbf b_{\mathrm{eff}}.}
$$

If there are $K$ classes and $d$ input features, $W_{\mathrm{eff}}$ has shape $K\times d$ and $\mathbf b_{\mathrm{eff}}$ has shape $K\times1$.

### Step 2: keep the same softmax head

The deep network's predicted probabilities are

$$
\hat{\mathbf y}
=\operatorname{softmax}(\mathbf z)
=\operatorname{softmax}(W_{\mathrm{eff}}\mathbf x+\mathbf b_{\mathrm{eff}}).
$$

A single affine layer followed by softmax produces **exactly the same probabilities**. This is the usual multiclass linear classifier, also called multinomial logistic regression.

Softmax is nonlinear, but it is already present in both models. We are not claiming that the probability vector itself is a linear function of the input; we are saying that the hidden depth adds no extra nonlinearity to the logits.

### Step 3: state the capacity conclusion precisely

Every function produced by the deep identity-activation network can be produced by a single affine-logit classifier with the same softmax head. Therefore the deep model has **no more representational capacity**.

It need not have strictly the same family of attainable logit maps: a narrow hidden bottleneck can restrict the effective matrix's rank,

$$
\operatorname{rank}(W_{\mathrm{eff}})
\leq\min_{1\leq l\leq L}\operatorname{rank}(W^{(l)}).
$$

Such restrictions can reduce the set of representable affine logit maps, not enlarge it. Depth can also change optimization and parameterization; that is different from adding new representable functions.

### Optional intuition: decision boundaries remain affine

Write class $i$'s logit as $z_i=\mathbf w_i^\top\mathbf x+b_i$. Softmax preserves the ordering of logits, so comparing classes $i$ and $j$ gives the boundary

$$
(\mathbf w_i-\mathbf w_j)^\top\mathbf x+(b_i-b_j)=0.
$$

That is an affine hyperplane. Repeated affine hidden layers do not introduce curved pairwise decision boundaries.

## 4. Common mistakes and self-checks

- **Calling the bias-containing layer strictly linear.** The correct term is affine.
- **Adding the two biases directly**, without passing the earlier one through the later matrix.
- **Reversing the weight-matrix product.** Matrix multiplication is not commutative.
- **Removing softmax.** The collapsed model retains the same output head.
- **Claiming that the whole probability map is linear.** The logits are affine; softmax probabilities are generally nonlinear.
- **Claiming that every hidden-width choice has identical capacity.** Bottlenecks can constrain the effective matrix.

Check that you can explain why inserting ReLU between the two layers would invalidate the affine-collapse proof: one can no longer distribute the second matrix through the intervening activation as if that activation were the identity.

## 5. What belongs in the eventual LaTeX answer?

- **(a):** substitute the first affine map into the second, then identify $\widetilde W$ and $\widetilde{\mathbf b}$.
- **(b):** repeat the composition argument through all layers and retain softmax to obtain $\operatorname{softmax}(W_{\mathrm{eff}}\mathbf x+\mathbf b_{\mathrm{eff}})$.

The tiny example, detailed dimension table, and extra interpretation remain study material. **Question 2(1)(a) and (b) are filled in LaTeX.** The submission keeps (b) brief: the affine collapse supplies the proof, followed by the softmax/capacity conclusion.
