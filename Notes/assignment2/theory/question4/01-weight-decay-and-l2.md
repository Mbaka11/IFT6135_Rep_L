# Assignment 2 — Question 4.1: weight decay versus $L_2$

**Goal:** compare the assignment's explicit weight-decay update with adding an $L_2$ penalty to the loss, for SGD, RMSProp and Adam.

- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Course optimization slides — RMSProp and Adam](../../../../Slides/12_firstorder_optimization.pdf#page=13)
- [Next: 4.2 — Squared-error level sets](02-squared-error-level-sets.md)

**Status:** notes only; the Question 4 LaTeX answer remains blank.

## 1. Two different places to add shrinkage

Write $g_t=\nabla_\theta\mathcal L(\theta_t)$, and let $\eta>0$ be the learning rate. The question writes **decoupled weight decay** as

$$
\theta_{t+1}=\theta_t-\Delta\theta_t-\lambda\theta_t,
$$

where $\Delta\theta_t$ is the optimizer's usual update computed from the unregularized loss. The shrinkage term is applied directly to the parameter.

For **$L_2$ regularization**, the objective is $\mathcal L_{\rm reg}=\mathcal L+\gamma\|\theta\|_2^2$. Its gradient is

$$
\widetilde g_t=g_t+2\gamma\theta_t.
$$

The optimizer uses $\widetilde g_t$ in place of $g_t$, including in its moving averages. The factor is **$2\gamma$** because the assignment does not put a $1/2$ in front of the penalty.

The symbol $\lambda$ here is the coefficient of the assignment's explicit term $-\lambda\theta_t$. Some libraries instead define weight decay with a factor of $\eta$ built in; compare actual update equations rather than coefficient names. The assignment's $L_2$ penalty is on all of $\theta$; in practice, implementations may exclude biases or other parameters.

## 2. SGD: equivalent after matching coefficients

For plain SGD without momentum,

$$
\begin{aligned}
\text{decay:}\quad \theta_{t+1}&=\theta_t-\eta g_t-\lambda\theta_t,\\
L_2:\quad \theta_{t+1}&=\theta_t-\eta(g_t+2\gamma\theta_t)
=\theta_t-\eta g_t-2\eta\gamma\theta_t.
\end{aligned}
$$

The updates agree when

$$
\boxed{\lambda=2\eta\gamma}.
$$

With the same initialization and minibatch sequence, this equality holds at every step for constant $\eta$. If $\eta_t$ varies, matching fixed $\gamma$ requires $\lambda_t=2\eta_t\gamma$. This result is for **plain** SGD; if momentum also processes the regularization gradient, the equivalence with a separately applied decay term does not generally carry over.

## 3. RMSProp: not generally equivalent

The course version keeps an elementwise average of squared gradients. Denote it by $r_t$ and use $\delta>0$ for stabilization (the slides place it inside the square root):

$$
r_t=\rho r_{t-1}+(1-\rho)g_t\odot g_t,
\qquad
\theta_{t+1}^{\rm decay}=\theta_t-\eta\frac{g_t}{\sqrt{\delta+r_t}}-\lambda\theta_t.
$$

All divisions, square roots and products are elementwise. With an $L_2$ penalty instead,

$$
\widetilde r_t=\rho\widetilde r_{t-1}+(1-\rho)\widetilde g_t\odot\widetilde g_t,
\qquad
\theta_{t+1}^{L_2}=\theta_t-\eta\frac{\widetilde g_t}{\sqrt{\delta+\widetilde r_t}}.
$$

The penalty gradient goes **through** the adaptive denominator and changes the running second moment. Decoupled decay is applied **after** the unregularized optimizer update and does not enter that moment. Their shrinkage therefore depends on different quantities; no single fixed scalar mapping generally makes these trajectories equal. A special setting or one particular step can coincide, but that does not establish general equivalence.

### Concrete first-step check

Take one coordinate with $g_1=0$, $\theta_1=1$, $\eta=0.1$, $\gamma=0.5$, $\rho=0$, $\delta=1$, and zero initial accumulator. Then $\widetilde g_1=1$ and $\widetilde r_1=1$. If we used the SGD matching coefficient $\lambda=2\eta\gamma=0.1$:

$$
\theta_2^{\rm decay}=0.9,
\qquad
\theta_2^{L_2}=1-\frac{0.1}{\sqrt2}\approx0.9293.
$$

The updates differ even on this simple step. More generally, the accumulator and coefficient interactions rule out a universal fixed scalar equivalence.

## 4. Adam: not generally equivalent

Adam forms moving averages of the gradient and its elementwise square:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t\odot g_t.
$$

After bias correction, the unregularized update is $\eta\widehat m_t/(\sqrt{\widehat v_t}+\epsilon)$, using elementwise operations. (The precise placement of the stabilizer varies by convention; it does not change the comparison.) With decoupled decay, $-\lambda\theta_t$ is added separately. With $L_2$, Adam instead computes its moment estimates using $\widetilde g_t=g_t+2\gamma\theta_t$.

Thus $L_2$ changes **both** moments and their future values, and its contribution is adaptively scaled. Decoupled weight decay changes the parameter directly and does not insert the penalty into those moment estimates. The two are not generally equivalent. The decoupled Adam variant is often called **AdamW**; check a library's specific update and coefficient convention before comparing it to the assignment.

## What to retain for the short answer

Plain SGD: yes, with $\lambda=2\eta\gamma$. RMSProp and Adam: no in general, because the penalty gradient enters their adaptive statistics and scaling, while the question's decay term is applied separately. Give the equations to support the distinction.
