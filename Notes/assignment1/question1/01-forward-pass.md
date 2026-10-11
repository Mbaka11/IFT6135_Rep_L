# Assignment 1 — Question 1.1: forward pass

**Scope:** Question 1, part (1), before any training or parameter update. Parts (2)–(6) are not solved here.

[Assignment statement and answer template](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)

These are study notes, not the text of a final submission. A concise calculation and the completed table are now included in Answer 1, part (1), of the LaTeX template; the detailed reminders and toy example stay here.

## 1. The network at a glance

The assignment uses a **2 → 3 → 1** network:

- **Input layer: 2 inputs.** It supplies the two components of the input vector; it does not apply a learned transformation here.
- **Hidden layer: 3 neurons.** Each neuron forms a weighted sum of the inputs, adds its bias, and applies ReLU.
- **Output layer: 1 neuron.** It forms a weighted sum of the three hidden outputs, adds its bias, and applies sigmoid.

It is called a **two-layer MLP** because the two layers with learned parameters are the hidden layer and the output layer. The input layer is not counted in this convention.

The computation follows this order:

$$
\mathbf{x}
\longrightarrow \mathbf{z}^{(1)}
\longrightarrow \mathbf{h}^{(1)}
\longrightarrow z^{(2)}
\longrightarrow \hat{y}.
$$

## 2. Quick reminders: ReLU versus sigmoid

### ReLU: used in the hidden layer

$$
\operatorname{ReLU}(a)=\max(0,a).
$$

It keeps positive values and replaces negative values with zero. For a vector, apply it **separately to each component**:

$$
\operatorname{ReLU}\!\left(\begin{bmatrix}-2\\0\\3\end{bmatrix}\right)
=\begin{bmatrix}0\\0\\3\end{bmatrix}.
$$

ReLU outputs are nonnegative, but they are **not restricted to a maximum of 1** and are not probabilities.

### Sigmoid: used in the output layer

$$
\sigma(a)=\frac{1}{1+e^{-a}}.
$$

It maps a scalar to a value strictly between 0 and 1. In this binary-classification setup, that value is interpreted as the model's predicted probability of class 1:

- A very negative input gives a value close to 0.
- A zero input gives exactly $\sigma(0)=1/2$.
- A very positive input gives a value close to 1.

**Main distinction here:** ReLU produces the hidden features; sigmoid turns the final scalar score into a predicted probability. Use the activation specified for each layer, not the same activation everywhere.

## 3. What the symbols mean

The superscripts $(1)$ and $(2)$ identify layers; they are **not powers**. Bold symbols below are vectors.

| Symbol | Meaning | Shape in this question |
| --- | --- | --- |
| $\mathbf{x}$ | Input sample, as a column vector | $2\times1$ |
| $W_1$ | Weights connecting the input to the hidden layer | $3\times2$ |
| $\mathbf{b}_1$ | One bias for each hidden neuron | $3\times1$ |
| $\mathbf{z}^{(1)}$ | Hidden-layer weighted sums **before** ReLU; the pre-activations | $3\times1$ |
| $\mathbf{h}^{(1)}$ | Hidden-layer outputs **after** ReLU; the activations | $3\times1$ |
| $W_2$ | Output weights, written as a column vector in the statement | $3\times1$ |
| $W_2^\top$ | The same output weights, transposed into a row | $1\times3$ |
| $b_2$ | Bias of the single output neuron | Scalar |
| $z^{(2)}$ | Output weighted sum **before** sigmoid; also called a logit | Scalar |
| $\hat{y}$ | Prediction **after** sigmoid, pronounced “y-hat” | Scalar |
| $y$ | True label supplied with the training sample | Scalar |

The hat is on **$y$**, not $h$. Distinguish $\hat{y}$, the prediction, from $y$, the known label.

The equations are:

$$
\mathbf{z}^{(1)}=W_1\mathbf{x}+\mathbf{b}_1,
\qquad
\mathbf{h}^{(1)}=\operatorname{ReLU}(\mathbf{z}^{(1)}),
$$

$$
z^{(2)}=W_2^\top\mathbf{h}^{(1)}+b_2,
\qquad
\hat{y}=\sigma(z^{(2)}).
$$

A useful dimension check is:

$$
(3\times2)(2\times1)+(3\times1)=(3\times1),
$$

$$
(1\times3)(3\times1)+\text{scalar}=\text{scalar}.
$$

This explains why the transpose on $W_2$ matters: we need a dot product that returns one number.

## 4. What “component-wise notation” means

It means expanding a vector or matrix expression into its individual scalar calculations.

For hidden neuron $i$, the corresponding **row** of $W_1$ contains its two input weights:

$$
z_i^{(1)}=W_{1,i1}x_1+W_{1,i2}x_2+b_{1,i},
\qquad i\in\{1,2,3\}.
$$

For the single output neuron, the statement writes the dot product explicitly:

$$
z^{(2)}
=W_{2,1}h_1^{(1)}+W_{2,2}h_2^{(1)}+W_{2,3}h_3^{(1)}+b_2.
$$

Here $h_i^{(1)}$ is hidden neuron $i$'s output, and $W_{2,i}$ is the weight connecting that neuron to the output. Multiply each hidden output by its matching weight, then add the contributions and the bias. This is the **same operation** as $W_2^\top\mathbf{h}^{(1)}+b_2$, not an additional step.

## 5. The given parameters, sample, and conventions

### Initial parameters

$$
W_1=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix},
\qquad
\mathbf{b}_1=\begin{bmatrix}0\\0\\0\end{bmatrix},
$$

$$
W_2=\begin{bmatrix}1\\-1\\1\end{bmatrix},
\qquad b_2=0.
$$

These are the current weights and biases. **Keep them fixed** throughout this forward pass. No learning-rate or gradient-descent calculation is needed yet.

### Training sample

$$
\mathbf{x}=\begin{bmatrix}1\\1\end{bmatrix},
\qquad y=1.
$$

Feed $\mathbf{x}$ into the network. The label $y$ is not an input to these forward-pass equations; it tells us the desired class and will be used in the loss later. Knowing $y=1$ does not force the prediction to be 1.

### Conventions

- **$e\approx3$:** use this approximation if evaluating an exponential requires it. In our final computation, the exponent will be zero, so the approximation will not change the result.
- **$\operatorname{ReLU}'(0)=0$:** this is a derivative convention for the later backward pass. It is not needed to calculate the forward pass. The forward value $\operatorname{ReLU}(0)=0$ already follows from $\max(0,a)$.

## 6. What a forward pass does

A forward pass evaluates the network from input to output using its **current parameters**:

1. Compute the hidden pre-activations.
2. Apply the hidden activation.
3. Compute the output pre-activation.
4. Apply the output activation to obtain the prediction.

There are no parameter updates in these steps. A forward pass is used during both training and prediction; by itself, it is not the learning/update step.

### Tiny example, separate from the assignment

To see the sequence without matrices, consider **one input, one hidden neuron, and one output neuron**:

- Input: $x=2$.
- Hidden weight: $w_1=1$; hidden bias: $b_1=-1$; activation: ReLU.
- Output weight: $w_2=1$; output bias: $b_2=-1$; activation: sigmoid.

Then:

$$
z^{(1)}=1\cdot2-1=1,
\qquad h^{(1)}=\max(0,1)=1,
$$

$$
z^{(2)}=1\cdot1-1=0,
\qquad\hat{y}=\frac{1}{1+e^0}=\frac12.
$$

The important pattern is **weighted sum plus bias → activation → weighted sum plus bias → activation**. This toy example has a different size from the assignment; only the calculation pattern is shared.

## 7. Worked solution to Question 1.1

We now use the assignment's actual **2 → 3 → 1** network and its given initial parameters.

### Step 1 — Compute the hidden pre-activations

Substitute the input and parameters into $\mathbf{z}^{(1)}=W_1\mathbf{x}+\mathbf{b}_1$:

$$
\mathbf{z}^{(1)}
=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix}
\begin{bmatrix}1\\1\end{bmatrix}
+\begin{bmatrix}0\\0\\0\end{bmatrix}.
$$

Compute one row at a time:

$$
z_1^{(1)}=1\cdot1+0\cdot1+0=1,
$$

$$
z_2^{(1)}=1\cdot1+1\cdot1+0=2,
$$

$$
z_3^{(1)}=0\cdot1+1\cdot1+0=1.
$$

Therefore:

$$
\boxed{\mathbf{z}^{(1)}=\begin{bmatrix}1\\2\\1\end{bmatrix}}.
$$

Intuitively, with these particular weights, the first hidden neuron selects $x_1$, the second adds $x_1+x_2$, and the third selects $x_2$. All hidden biases are zero.

### Step 2 — Apply ReLU to each hidden pre-activation

$$
\mathbf{h}^{(1)}
=\begin{bmatrix}\max(0,1)\\\max(0,2)\\\max(0,1)\end{bmatrix}
=\boxed{\begin{bmatrix}1\\2\\1\end{bmatrix}}.
$$

All three pre-activations are positive, so ReLU leaves them unchanged. ReLU does **not** cap the value 2 at 1.

### Step 3 — Compute the output pre-activation

Transpose the output-weight vector and take the dot product:

$$
z^{(2)}
=\begin{bmatrix}1&-1&1\end{bmatrix}
\begin{bmatrix}1\\2\\1\end{bmatrix}+0.
$$

In component-wise form:

$$
z^{(2)}=1\cdot1+(-1)\cdot2+1\cdot1+0
=1-2+1=\boxed{0}.
$$

The first and third neurons contribute $+1$ each; the second contributes $-2$. These contributions cancel.

### Step 4 — Apply sigmoid to obtain the prediction

$$
\hat{y}=\sigma(0)=\frac{1}{1+e^{-0}}
=\frac{1}{1+1}=\boxed{\frac12=0.5}.
$$

Because $e^0=1$ exactly, using the convention $e\approx3$ gives the same result: $3^0=1$.

### Completed answer table

Vectors below are written in column-vector form using the transpose notation.

| Quantity | Value for $\mathbf{x}=[1,1]^\top$, $y=1$ |
| --- | --- |
| $\mathbf{z}^{(1)}$ | $[1,2,1]^\top$ |
| $\mathbf{h}^{(1)}$ | $[1,2,1]^\top$ |
| $z^{(2)}$ | $0$ |
| $\hat{y}$ | $1/2=0.5$ |

**Interpretation:** before training, the model assigns probability 0.5 to class 1 for this sample, although the true label is 1. That is not a calculation error: the given initial parameters need not already classify the sample confidently. No hard class label or tie-breaking rule is requested for this table; report the sigmoid output.

## 8. What belongs in Notes versus the submission?

- **In Notes:** keep the reminders, symbol meanings, toy example, expanded arithmetic, and interpretation above. They are here to support understanding.
- **In the LaTeX answer:** a sensible concise version would contain the completed table and a few calculation lines showing $W_1\mathbf{x}+\mathbf{b}_1$, the ReLU result, the output dot product, and the sigmoid evaluation. There is no need to copy the whole study note or toy example into the submission.

The wording of part (1) specifically asks you to **fill in the table**; it does not explicitly ask for this long explanation. The template also tells you to follow the course's assignment policy. Check that policy or ask a TA if it requires a particular amount of working. Brief, checkable mathematical steps are a safer choice than unexplained numbers, but this note cannot establish the grader's requirements.

**LaTeX status: all six parts of Answer 1 are filled.**

## 9. Quick self-checks

Before moving on, check that you can explain:

1. Why $W_1$ has three rows and two columns.
2. Why the hidden pre-activation vector has three entries.
3. Why ReLU leaves the hidden vector unchanged in this example.
4. Why $W_2$ is transposed in the output equation.
5. Why a zero output logit produces 0.5, not 0.
6. Why the true label $y=1$ is not substituted into the forward-pass prediction.
