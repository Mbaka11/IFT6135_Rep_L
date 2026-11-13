# Assignment 1 — Question 1.3: analytical gradients and backpropagation

**Scope:** Question 1, part (3): derive formulas for the gradients with respect to $b_2$, $W_2$, $\mathbf{h}^{(1)}$, $\mathbf{b}_1$, and $W_1$. This is not the separate main Question 3 on a quadratic loss.

- [1.1 — Forward pass and notation](01-forward-pass.md)
- [Previous: 1.2 — Binary cross-entropy derivative](02-cross-entropy-derivative.md)
- [Next: 1.4 — Numerical gradients](04-numerical-gradients.md)
- [Assignment statement and answer template](../../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)

These are study notes. The LaTeX answer for part (3) now shows explicit component-wise chain-rule products followed by the vector/matrix formulas. **Substituting the assignment's numbers into all parameter gradients belongs to part (4).**

## 1. What “analytical formula” means

An analytical formula describes the derivative using symbols such as the input, weights, activations, and prediction. It shows how to compute the derivative for a suitable sample and parameter setting, rather than giving only the values at this initialization.

A gradient measures the loss's local sensitivity to a variable. When differentiating with respect to a parameter, hold the input, label, and other independent parameters fixed, but account for every intermediate quantity that depends on that parameter.

For example, $\mathbf{h}^{(1)}$ does not depend on $W_2$, but it does depend on $W_1$. That dependency determines which chain-rule path we follow.

## 2. The chain rule: one small example

For a scalar intermediate value $u=u(\theta)$:

$$
\frac{\partial L}{\partial\theta}
=\frac{\partial L}{\partial u}\frac{\partial u}{\partial\theta}.
$$

If the same variable affects several intermediate components, **add** the contributions of those paths.

As a separate toy example, let:

$$
u=wx+b,\qquad L=\frac12u^2.
$$

The local derivatives are:

$$
\frac{\partial L}{\partial u}=u,
\qquad
\frac{\partial u}{\partial w}=x,
\qquad
\frac{\partial u}{\partial b}=1.
$$

Therefore:

$$
\frac{\partial L}{\partial w}=ux,
\qquad
\frac{\partial L}{\partial b}=u.
$$

With the toy values $x=2$, $w=3$, and $b=1$, we get $u=7$, a weight gradient of $14$, and a bias gradient of $7$. These are **not** the assignment's values or loss; this example only demonstrates multiplying an incoming loss derivative by a local derivative.

## 3. The assignment's computation graph

The forward dependencies are:

$$
\mathbf{z}^{(1)}=W_1\mathbf{x}+\mathbf{b}_1,
\qquad
\mathbf{h}^{(1)}=\operatorname{ReLU}(\mathbf{z}^{(1)}),
$$

$$
z^{(2)}=W_2^\top\mathbf{h}^{(1)}+b_2,
\qquad
\hat{y}=\sigma(z^{(2)}),
\qquad
L=\operatorname{BCE}(\hat{y},y).
$$

The backward calculation begins at the loss and follows these dependencies in reverse. At each weighted-sum operation, it branches to the weights, bias, and inputs of that operation.

From part (2), define the scalar output error signal:

$$
\delta^{(2)}:=\frac{\partial L}{\partial z^{(2)}}=\hat{y}-y.
$$

This already includes the effects of **both binary cross-entropy and sigmoid**. Do not multiply it by another sigmoid derivative.

### Gradient shapes and conventions

We use the usual machine-learning convention: the gradient of a scalar loss has the **same shape as the variable being differentiated**. Gradients with respect to column vectors are column vectors.

| Variable | Shape | Gradient shape |
| --- | --- | --- |
| $b_2$ | Scalar | Scalar |
| $W_2$ | $3\times1$ | $3\times1$ |
| $\mathbf{h}^{(1)}$ | $3\times1$ | $3\times1$ |
| $\mathbf{b}_1$ | $3\times1$ | $3\times1$ |
| $W_1$ | $3\times2$ | $3\times2$ |

The statement writes $W_2$ as a column vector, which is why the forward pass uses $W_2^\top$.

## 4. Gradient with respect to the output bias

Expand the output weighted sum:

$$
z^{(2)}=\sum_{i=1}^3W_{2,i}h_i^{(1)}+b_2.
$$

Changing $b_2$ changes $z^{(2)}$ by the same amount, so:

$$
\frac{\partial z^{(2)}}{\partial b_2}=1.
$$

Apply the chain rule:

$$
\boxed{\frac{\partial L}{\partial b_2}
=\delta^{(2)}=\hat{y}-y}.
$$

This is a scalar, matching the scalar bias.

## 5. Gradient with respect to the output weights

Consider one output weight $W_{2,i}$. Its contribution to the output score is $W_{2,i}h_i^{(1)}$, and the hidden activation is independent of that output weight. Therefore:

$$
\frac{\partial z^{(2)}}{\partial W_{2,i}}=h_i^{(1)}.
$$

The chain rule gives:

$$
\frac{\partial L}{\partial W_{2,i}}
=\delta^{(2)}h_i^{(1)}.
$$

Stack the three components into a column vector:

$$
\frac{\partial L}{\partial W_2}
=\begin{bmatrix}
\delta^{(2)}h_1^{(1)}\\
\delta^{(2)}h_2^{(1)}\\
\delta^{(2)}h_3^{(1)}
\end{bmatrix}
=\boxed{\mathbf{h}^{(1)}\delta^{(2)}}.
$$

Equivalently:

$$
\boxed{\frac{\partial L}{\partial W_2}
=(\hat{y}-y)\mathbf{h}^{(1)}}.
$$

**Interpretation:** each weight's gradient is the output error signal multiplied by the activation feeding into that weight.

## 6. Gradient with respect to the hidden activations

Now differentiate the same output score with respect to $h_i^{(1)}$. Its coefficient is $W_{2,i}$:

$$
\frac{\partial z^{(2)}}{\partial h_i^{(1)}}=W_{2,i}.
$$

Thus:

$$
\frac{\partial L}{\partial h_i^{(1)}}
=\delta^{(2)}W_{2,i}.
$$

Stacking the components gives:

$$
\boxed{\frac{\partial L}{\partial\mathbf{h}^{(1)}}
=W_2\delta^{(2)}
=(\hat{y}-y)W_2}.
$$

The gradient is a $3\times1$ column vector. The forward pass uses $W_2^\top$, but this backward gradient uses **$W_2$**, not its transpose.

**Interpretation:** the output error is distributed to the hidden neurons according to their connections to the output. The hidden-activation gradient is not yet the hidden-pre-activation gradient: ReLU still lies between those two quantities.

## 7. Passing the gradient through ReLU

For each hidden neuron:

$$
h_i^{(1)}=\operatorname{ReLU}(z_i^{(1)}).
$$

Using the assignment's convention at zero:

$$
\operatorname{ReLU}'(a)
=\begin{cases}
1,&a>0,\\
0,&a\leq0.
\end{cases}
$$

Define the vector of these local derivatives:

$$
\mathbf{r}
=\begin{bmatrix}
\operatorname{ReLU}'(z_1^{(1)})\\
\operatorname{ReLU}'(z_2^{(1)})\\
\operatorname{ReLU}'(z_3^{(1)})
\end{bmatrix}.
$$

The hidden error signal is:

$$
\boldsymbol{\delta}^{(1)}
:=\frac{\partial L}{\partial\mathbf{z}^{(1)}}
=\frac{\partial L}{\partial\mathbf{h}^{(1)}}\odot\mathbf{r}
=\boxed{\left(W_2\delta^{(2)}\right)\odot\mathbf{r}}.
$$

The symbol $\odot$ means **element-wise multiplication**: multiply matching vector entries. This is neither a dot product nor ordinary matrix multiplication.

Component-wise:

$$
\delta_i^{(1)}
=\delta^{(2)}W_{2,i}\operatorname{ReLU}'(z_i^{(1)}).
$$

A positive pre-activation passes its incoming gradient through unchanged. A nonpositive pre-activation blocks the gradient to that neuron's pre-activation and its first-layer parameters under the stated convention.

For the initial forward pass, $\mathbf{z}^{(1)}=[1,2,1]^\top$, so $\mathbf{r}=[1,1,1]^\top$. Notice that the derivative at the middle neuron's input is **1**, not 2: do not confuse the ReLU activation with its derivative.

An inactive neuron can still have a nonzero gradient with respect to its **activation** $h_i^{(1)}$. ReLU's mask is what makes the corresponding gradient with respect to **$z_i^{(1)}$** zero.

## 8. Gradient with respect to the hidden biases

The first-layer affine calculation is:

$$
z_i^{(1)}=\sum_{j=1}^2W_{1,ij}x_j+b_{1,i}.
$$

Bias $b_{1,i}$ affects $z_i^{(1)}$ with derivative 1 and does not directly affect another hidden neuron's pre-activation. Consequently:

$$
\frac{\partial L}{\partial b_{1,i}}=\delta_i^{(1)}.
$$

In vector form:

$$
\boxed{\frac{\partial L}{\partial\mathbf{b}_1}
=\boldsymbol{\delta}^{(1)}
=\left(W_2(\hat{y}-y)\right)\odot\mathbf{r}}.
$$

**Interpretation:** a bias gradient equals that neuron's pre-activation error signal, after accounting for its activation derivative.

## 9. Gradient with respect to the first-layer weights

For weight $W_{1,ij}$:

- The first index $i$ chooses the hidden neuron: a row of $W_1$.
- The second index $j$ chooses the input feature: a column of $W_1$.

Its contribution to $z_i^{(1)}$ is $W_{1,ij}x_j$, so:

$$
\frac{\partial z_i^{(1)}}{\partial W_{1,ij}}=x_j.
$$

It does not directly affect $z_k^{(1)}$ for $k\ne i$. The sum of chain-rule paths therefore reduces to:

$$
\frac{\partial L}{\partial W_{1,ij}}
=\sum_{k=1}^3\delta_k^{(1)}
\frac{\partial z_k^{(1)}}{\partial W_{1,ij}}
=\boxed{\delta_i^{(1)}x_j}.
$$

Arrange these six entries into a matrix:

$$
\frac{\partial L}{\partial W_1}
=\begin{bmatrix}
\delta_1^{(1)}x_1&\delta_1^{(1)}x_2\\
\delta_2^{(1)}x_1&\delta_2^{(1)}x_2\\
\delta_3^{(1)}x_1&\delta_3^{(1)}x_2
\end{bmatrix}
=\boxed{\boldsymbol{\delta}^{(1)}\mathbf{x}^\top}.
$$

This is an **outer product**:

$$
(3\times1)(1\times2)=3\times2.
$$

It produces a gradient for every connection from an input feature to a hidden neuron. In terms of the original quantities:

$$
\boxed{\frac{\partial L}{\partial W_1}
=\left[\left(W_2(\hat{y}-y)\right)\odot\mathbf{r}\right]\mathbf{x}^\top}.
$$

## 10. The five requested formulas in one place

Define:

$$
\delta^{(2)}=\hat{y}-y,
\qquad
\mathbf{r}=\operatorname{ReLU}'(\mathbf{z}^{(1)}),
\qquad
\boldsymbol{\delta}^{(1)}=(W_2\delta^{(2)})\odot\mathbf{r},
$$

where the ReLU derivative is applied component-wise.

| Requested gradient | Formula | Shape |
| --- | --- | --- |
| $\partial L/\partial b_2$ | $\delta^{(2)}$ | Scalar |
| $\partial L/\partial W_2$ | $\mathbf{h}^{(1)}\delta^{(2)}$ | $3\times1$ |
| $\partial L/\partial\mathbf{h}^{(1)}$ | $W_2\delta^{(2)}$ | $3\times1$ |
| $\partial L/\partial\mathbf{b}_1$ | $\boldsymbol{\delta}^{(1)}$ | $3\times1$ |
| $\partial L/\partial W_1$ | $\boldsymbol{\delta}^{(1)}\mathbf{x}^\top$ | $3\times2$ |

These are gradients for the **single-sample loss in the statement**. There is no additional batch-average factor in that loss.

## 11. Common mistakes and self-checks

- **Forgetting the ReLU derivative:** gradients of the first-layer biases and weights must use $\boldsymbol{\delta}^{(1)}$, not simply $\partial L/\partial\mathbf{h}^{(1)}$.
- **Using the forward ReLU values as the mask:** the backward mask contains 0 or 1, not arbitrary positive activation values.
- **Transposing the wrong vector:** both $W_2$ and its gradient are columns; the first-layer weight gradient is $\boldsymbol{\delta}^{(1)}\mathbf{x}^\top$.
- **Confusing an outer product with a dot product:** the $W_1$ gradient must be a $3\times2$ matrix, not a scalar.
- **Applying sigmoid's derivative again:** it is already included in $\delta^{(2)}=\hat{y}-y$.
- **Updating parameters while deriving gradients:** use one consistent forward pass. Updating the parameters is part (5).

Check that you can explain:

1. Why $\partial L/\partial b_2$ is just the output error signal.
2. Why an output weight gradient includes a hidden activation.
3. Why a hidden-activation gradient includes an output weight instead.
4. Where ReLU's derivative enters the calculation.
5. Why the first-layer weight gradient is an outer product.

## 12. What goes into the eventual LaTeX answer?

The LaTeX answer uses the expanded-chain presentation: for each requested derivative, it writes the local derivative factors, substitutes their formulas, and then collects the scalar results into the vector or matrix gradient. For the first-layer parameters, it explicitly expands the path through the hidden activation and hidden pre-activation. This is equivalent to the compact error-signal notation above; both use the same chain rule.

**LaTeX status: parts (3) and (4) are filled.** Part (4) uses these formulas with the initial forward-pass values to obtain the numerical parameter gradients; see the [next note](04-numerical-gradients.md).
