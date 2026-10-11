# Assignment 1 — Question 1.2: binary cross-entropy derivative

**Scope:** Question 1, part (2): derive and evaluate the loss derivative with respect to the output pre-activation $z^{(2)}$. This is the unnumbered prompt between parts (1) and (3), not the separate main Question 2 on activation functions.

- [Previous: 1.1 — Forward pass](01-forward-pass.md)
- [Next: 1.3 — Analytical gradients](03-analytical-gradients.md)
- [Assignment statement and answer template](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)

These are study notes. Answer 1, part (2), contains the derivation and numerical result in LaTeX; part (3) now contains the expanded analytical chain-rule derivations.

## 1. What exactly are we differentiating?

The question asks for:

$$
\frac{\partial L}{\partial z^{(2)}}.
$$

This measures how the loss changes when we change the **output score before sigmoid**. It is not the derivative with respect to the prediction $\hat{y}$, although that derivative will be an intermediate step.

The relevant dependency is:

$$
z^{(2)}\longrightarrow\hat{y}=\sigma(z^{(2)})\longrightarrow L.
$$

For the calculations below, write $z=z^{(2)}$ and $p=\hat{y}$ to keep the notation short. The true label $y$ is fixed when differentiating.

From part (1), the given sample has:

$$
z=0,\qquad p=\frac12,\qquad y=1.
$$

## 2. Quick reminder: what binary cross-entropy measures

The loss is:

$$
L=-\left[y\ln p+(1-y)\ln(1-p)\right].
$$

Here $\ln$ is the natural logarithm. The two possible labels select the relevant probability:

- If $y=1$, then $L=-\ln p$. Assigning a larger probability to class 1 reduces the loss.
- If $y=0$, then $L=-\ln(1-p)$. Assigning a smaller probability to class 1 reduces the loss.

For a finite sigmoid input, $0<p<1$, so the expressions below are well-defined mathematically.

## 3. The chain rule we need

For a composition $L(p(z))$, multiply the two local derivatives:

$$
\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial p}\frac{\partial p}{\partial z}.
$$

We will therefore calculate:

1. How the loss changes with the predicted probability.
2. How the predicted probability changes with the output score.
3. Their product.

## 4. First local derivative: loss with respect to prediction

Use:

$$
\frac{d}{dp}\ln p=\frac1p,
\qquad
\frac{d}{dp}\ln(1-p)=-\frac{1}{1-p}.
$$

The minus sign in the second derivative comes from differentiating the inner expression $1-p$.

Applying these rules to both loss terms gives:

$$
\frac{\partial L}{\partial p}
=-\frac{y}{p}+\frac{1-y}{1-p}.
$$

Notice that the second term is **positive**: the loss's outside minus sign and the derivative's minus sign cancel.

For the actual label $y=1$, this simplifies to $-1/p$. However, keeping $y$ symbolic first gives a formula that works for both labels.

## 5. Second local derivative: sigmoid with respect to its input

Start with:

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}=(1+e^{-z})^{-1}.
$$

Differentiate the outer reciprocal and the inner exponential:

$$
\frac{\partial p}{\partial z}
=-(1+e^{-z})^{-2}(-e^{-z})
=\frac{e^{-z}}{(1+e^{-z})^2}.
$$

Since:

$$
p=\frac{1}{1+e^{-z}},
\qquad
1-p=\frac{e^{-z}}{1+e^{-z}},
$$

we can rewrite the derivative as:

$$
\boxed{\frac{\partial p}{\partial z}=p(1-p)}.
$$

This is the standard sigmoid derivative. Derive the usual sigmoid analytically; the assignment's $e\approx3$ convention is a numerical approximation, not a request to redefine sigmoid as a base-3 function.

## 6. Combine the derivatives and simplify

Substitute the two local derivatives into the chain rule:

$$
\frac{\partial L}{\partial z}
=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p).
$$

Distribute and cancel the matching factors:

$$
\frac{\partial L}{\partial z}
=-y(1-p)+(1-y)p.
$$

Expand:

$$
-y+yp+p-yp=p-y.
$$

Returning to the assignment's notation:

$$
\boxed{\frac{\partial L}{\partial z^{(2)}}=\hat{y}-y}.
$$

The simplification comes from combining **this binary cross-entropy loss with sigmoid**. It is not a universal derivative for every loss or output activation.

## 7. Evaluate the requested value

Use the prediction from the initial forward pass and the given label:

$$
\frac{\partial L}{\partial z^{(2)}}
=\frac12-1
=\boxed{-\frac12=-0.5}.
$$

We can check the same result without using the simplified formula:

$$
\frac{\partial L}{\partial p}=-\frac{1}{1/2}=-2,
$$

$$
\frac{\partial p}{\partial z}=\frac12\left(1-\frac12\right)=\frac14,
$$

$$
\frac{\partial L}{\partial z}=(-2)\left(\frac14\right)=-\frac12.
$$

**Answer to 1.2:** the derivative with respect to the output pre-activation is $-1/2$.

## 8. What does the negative sign tell us?

A derivative is a local slope, not a probability. It can be negative even though the loss is nonnegative.

Here $y=1$, but the current prediction is only 0.5. A small increase in $z^{(2)}$ increases the sigmoid output. That makes the model assign more probability to the correct class, so the loss decreases: a negative loss slope is exactly what we expect.

As a separate illustration, if $p=0.8$:

- With $y=1$, the derivative is $0.8-1=-0.2$.
- With $y=0$, the derivative is $0.8-0=0.8$.

The true label changes which direction reduces the loss. We do not update any network parameter in this question yet.

## 9. Common mistakes and self-checks

- **Stopping at $\partial L/\partial\hat{y}$.** The question asks for $\partial L/\partial z^{(2)}$, so include the sigmoid derivative.
- **Multiplying by the sigmoid derivative twice.** Once the expression simplifies to $\hat{y}-y$, that factor has already been accounted for.
- **Reporting $y-\hat{y}$.** That reverses the derivative's sign. The gradient is $\hat{y}-y$; gradient descent later subtracts the gradient.
- **Plugging in an updated prediction.** Use the initial forward pass from part (1).
- **Replacing the exponential base before differentiating.** Work with the mathematical sigmoid as defined in the statement.

Before moving on, check that you can explain why $\partial L/\partial p=-2$ but $\partial L/\partial z=-1/2$ for this sample. They differentiate with respect to different quantities.

## 10. Connection to the next question

Define the output error signal:

$$
\delta^{(2)}:=\frac{\partial L}{\partial z^{(2)}}=\hat{y}-y.
$$

Part (3) uses this quantity as the starting point for propagating derivatives to the output weights, hidden activations, and first-layer parameters. The next note derives those formulas symbolically; their numerical evaluation belongs to part (4).

The concise LaTeX answer now shows the chain-rule simplification and the final value. The extended reminders and examples remain in these notes.
