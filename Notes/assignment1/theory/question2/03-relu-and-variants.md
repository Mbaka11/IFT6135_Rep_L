# Assignment 1 — Question 2.3: ReLU, inactive neurons, LeakyReLU, and Maxout

**Scope:** main Question 2, part (3). This single note contains all three sub-subquestions:

- **(a)** State the ReLU derivative, including the convention at zero.
- **(b)** Compute the incoming weight-gradient vector for an inactive neuron.
- **(c)** Compare LeakyReLU and Maxout as ways to avoid a fixed flat negative branch.

- [Previous: 2.2 — Sigmoid and the gradient bound](02-sigmoid-gradient-bound.md)
- [Assignment statement and answer template](../../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)
- [Course Maxout definition: artificial-neuron slides, physical PDF page 20](../../../../Slides/01_artificial_neuron.pdf#page=20)

These are study notes. **Question 2(3)(a), (b), and (c) are filled in LaTeX. Question 2 is complete.** This is not the separate main Question 3 on gradient descent.

## 1. Setup: one hidden neuron and its incoming weights

For neuron $i$ in hidden layer $l$, write

$$
a_i^{(l)}=\mathbf w_i^\top\mathbf h^{(l-1)}+b_i^{(l)},
\qquad h_i^{(l)}=g(a_i^{(l)}).
$$

Here $\mathbf w_i\in\mathbb R^{d_{l-1}}$ is the column version of row $i$ of $W^{(l)}$. It contains the weights **feeding into** neuron $i$.

Define the incoming loss derivative

$$
c_i:=\frac{\partial\mathcal L}{\partial h_i^{(l)}}.
$$

The remainder of the network determines $c_i$. To reach this neuron's weights, we still need to differentiate its activation and its affine pre-activation.

## 2. Part (a) — ReLU's derivative

ReLU is

$$
g(a)=\max(0,a)
=\begin{cases}
a,&a>0,\\
0,&a\leq0.
\end{cases}
$$

For strictly positive inputs, it is the identity and has slope 1. For strictly negative inputs, it is constant and has slope 0.

At $a=0$, the left and right slopes differ, so there is **no classical derivative**. For backpropagation, choose the conventional value $g'(0)=0$, consistent with the convention used earlier in the assignment. Under that stated convention,

$$
\boxed{
g'(a)=\begin{cases}
1,&a>0,\\
0,&a\leq0.
\end{cases}
}
$$

The value at zero is a chosen subgradient/backpropagation convention, not a proof that ReLU is differentiable there. It matters because part (b) includes equality in $a_i^{(l)}\leq0$.

## 3. Part (b) — Weight gradient for an inactive neuron

### Step 1: expand the scalar chain for an entry

For the weight connecting input component $j$ to neuron $i$,

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial w_{i,j}}
&=\frac{\partial\mathcal L}{\partial h_i^{(l)}}
  \frac{\partial h_i^{(l)}}{\partial a_i^{(l)}}
  \frac{\partial a_i^{(l)}}{\partial w_{i,j}}\\
&=c_i\,g'(a_i^{(l)})\,h_j^{(l-1)}.
\end{aligned}
$$

The last factor is the corresponding input to the neuron, just as in the first-layer weight derivation from Question 1.

### Step 2: collect the entries into a vector

$$
\boxed{
\nabla_{\mathbf w_i}\mathcal L
=c_i\,g'(a_i^{(l)})\,\mathbf h^{(l-1)}.
}
$$

The shape is $d_{l-1}\times1$, matching the column parameter vector $\mathbf w_i$.

### Step 3: apply the inactive-neuron condition

If $a_i^{(l)}\leq0$, then $g'(a_i^{(l)})=0$ under the stated convention. Hence

$$
\boxed{
\nabla_{\mathbf w_i}\mathcal L
=c_i\cdot0\cdot\mathbf h^{(l-1)}
=\mathbf0_{d_{l-1}}.
}
$$

Equivalently, every entry in **row $i$** of the matrix gradient $\partial\mathcal L/\partial W^{(l)}$ is zero. This does not imply that the entire layer's weight-gradient matrix is zero.

The bias gradient is also zero for this sample:

$$
\frac{\partial\mathcal L}{\partial b_i^{(l)}}
=c_i\,g'(a_i^{(l)})\cdot1=0.
$$

### Tiny example

Suppose $\mathbf h^{(l-1)}=[1,2]^\top$, $a_i^{(l)}=-2$, and $c_i=3$. Then

$$
\nabla_{\mathbf w_i}\mathcal L
=3\cdot0\cdot\begin{bmatrix}1\\2\end{bmatrix}
=\begin{bmatrix}0\\0\end{bmatrix}.
$$

A nonzero incoming loss gradient cannot pass through a local activation derivative of zero.

### Inactive on one sample versus a “dead” neuron

An inactive neuron on one example can still activate on another example. A dead ReLU usually means that it stays on the zero branch for all relevant training inputs, receiving no data-loss gradient through its own weights and bias.

With fixed incoming representations and no other update mechanism, those zero gradients prevent its own ordinary gradient update from bringing it back. Changes in upstream representations or other parameter-update terms can change that situation. One zero-gradient sample alone does not prove permanent death.

## 4. Part (c) — Two ways to modify the activation block

The issue is the **flat negative branch**, not simply that the output is negative. Both alternatives can pass learning signals where an ordinary ReLU would return a zero local derivative, but they do so differently.

### 4.1 LeakyReLU: keep a small negative-side slope

For a fixed $0<\alpha<1$, define

$$
g_{\mathrm{leaky}}(a)=\begin{cases}
a,&a\geq0,\\
\alpha a,&a<0.
\end{cases}
$$

Away from the kink,

$$
g_{\mathrm{leaky}}'(a)=\begin{cases}
1,&a>0,\\
\alpha,&a<0.
\end{cases}
$$

A convention is still needed at zero. The important change is that the entire strictly negative region now has slope $\alpha$, rather than slope zero.

For a negative pre-activation, the incoming weight gradient becomes

$$
\boxed{\nabla_{\mathbf w_i}\mathcal L
=c_i\alpha\mathbf h^{(l-1)}.}
$$

Using the earlier example with $\alpha=0.01$,

$$
g_{\mathrm{leaky}}(-2)=-0.02,
\qquad
\nabla_{\mathbf w_i}\mathcal L
=3(0.01)\begin{bmatrix}1\\2\end{bmatrix}
=\begin{bmatrix}0.03\\0.06\end{bmatrix}.
$$

The negative branch no longer kills this gradient outright. The small slope can still attenuate gradients; nonzero $\alpha$ does not guarantee a nonzero full-network gradient if, for example, the incoming derivative is already zero.

With fixed $\alpha$, LeakyReLU adds no trainable parameters to the original affine layer and is inexpensive to evaluate.

### 4.2 Maxout: choose among learned affine pieces

A Maxout unit computes several affine pre-activations, then takes their maximum. With $m$ pieces,

$$
s_{i,r}=\mathbf u_{i,r}^\top\mathbf h^{(l-1)}+c_{i,r},
\qquad 1\leq r\leq m,
$$

$$
\boxed{h_i^{(l)}=\max_{1\leq r\leq m}s_{i,r}.}
$$

Here $c_{i,r}$ denotes a piece's bias, not the incoming loss derivative $c_i$ used earlier.

This modifies the affine-plus-activation block: it is not merely a different scalar function applied after the same single pre-activation. Each unit needs multiple learned affine pieces.

Suppose one piece $r_*$ is the unique winner. Locally,

$$
h_i^{(l)}=s_{i,r_*},
\qquad
\frac{\partial h_i^{(l)}}{\partial s_{i,r_*}}=1,
\qquad
\frac{\partial h_i^{(l)}}{\partial s_{i,r}}=0\quad(r\ne r_*).
$$

Thus the winning piece's parameters receive

$$
\nabla_{\mathbf u_{i,r_*}}\mathcal L
=c_i\mathbf h^{(l-1)},
\qquad
\frac{\partial\mathcal L}{\partial c_{i,r_*}}=c_i,
$$

while losing pieces receive zero gradient for that example. The activation's input derivative is

$$
\nabla_{\mathbf h^{(l-1)}}h_i^{(l)}=\mathbf u_{i,r_*}.
$$

At ties, the function can be nondifferentiable and needs a winner/subgradient convention.

### Why this can avoid the ReLU zero branch

ReLU itself can be written as a maximum of two affine pieces:

$$
\max(0,\mathbf w_i^\top\mathbf h^{(l-1)}+b_i).
$$

One of those pieces is fixed at zero. When it wins, it provides a flat activation branch.

Maxout instead allows the competing pieces to be learned affine functions. The winning piece can have a useful slope even when its value is negative. There is no mandatory rule “negative pre-activation means zero derivative.”

For example,

$$
f(x)=\max(x,2x-1).
$$

At $x=-2$, the candidates are $-2$ and $-5$, so

$$
f(-2)=-2,
\qquad f'(-2)=1.
$$

Both candidates are negative, yet the winning branch has nonzero slope. At $x=2$, the second branch wins and the slope is 2. At $x=1$, the two pieces tie and a convention is needed.

**Caveat:** Maxout does not guarantee every gradient is nonzero. The incoming loss gradient can be zero, the selected input-slope vector can be zero, and some pieces may rarely or never win. Its advantage is the absence of a *fixed* flat negative region, not a universal guarantee against all training failures.

### 4.3 Comparison at a glance

| Aspect | LeakyReLU | Maxout |
| --- | --- | --- |
| Construction | One affine pre-activation followed by a fixed two-slope function. | Maximum of several learned affine pre-activations. |
| Negative-region behavior | Slope $\alpha>0$ for strictly negative input. | A winning affine piece can have nonzero slope even at a negative output value. |
| How gradients pass | Through the same neuron with factor 1 or $\alpha$. | Through the winning piece; losing pieces get zero gradient for that example. |
| Additional parameters | None when $\alpha$ is fixed. | $m$ sets of weights and biases per unit instead of one. |
| Cost and flexibility | Simple and inexpensive; fixed slopes. | More computation and parameters; learned piecewise-affine function. |
| Limitation | A small $\alpha$ can still shrink gradients; zero upstream gradients remain zero. | Flat selected pieces and rarely selected pieces remain possible. |

With input width $d_{l-1}$, a standard affine-plus-LeakyReLU neuron uses $d_{l-1}+1$ trainable weights/biases. A Maxout neuron with $m$ pieces uses $m(d_{l-1}+1)$.

The course slide describes Maxout gradients as sparse: for one example, only the selected piece receives a gradient. This sparsity is across pieces; it does not mean that every input-coordinate derivative must be sparse.

## 5. Common mistakes and self-checks

- **Calling ReLU differentiable at zero.** State the chosen backpropagation value there.
- **Concluding that every neuron in the layer has a zero gradient** because one neuron is inactive.
- **Confusing a zero forward output with a zero incoming loss derivative.** It is the local ReLU derivative that blocks the chain here.
- **Calling a neuron permanently dead after one inactive sample.** Consider its behavior across the training inputs.
- **Saying LeakyReLU always has derivative 1.** Its negative-side derivative is $\alpha$.
- **Treating Maxout as a maximum with a mandatory zero branch.** Its affine pieces are learned.
- **Claiming either alternative guarantees nonzero gradients everywhere.** Both depend on the rest of the network and the selected local path.

Check that you can explain why the LeakyReLU example uses the extra factor $0.01$, whereas the winning Maxout piece receives derivative 1 with respect to its own affine output.

## 6. What belongs in the eventual LaTeX answer?

- **(a):** give the two derivative branches and explicitly state the convention $g'(0)=0$.
- **(b):** write the chain-rule vector $c_i g'(a_i^{(l)})\mathbf h^{(l-1)}$ and conclude that it is zero when the neuron is inactive.
- **(c):** define LeakyReLU and Maxout, explain their distinct gradient paths, and contrast their parameter/computation costs without claiming universal guarantees.

The examples and detailed caveats remain learning material. **Question 2(3)(a), (b), and (c) are all filled in LaTeX.**
