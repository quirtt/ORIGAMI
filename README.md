# Ideal Abstractions for Decisions-Focused Learning

- [Decision-Theoretic Context](#decision-theoretic-context)
  - [Fold a Simplex](#fold-a-simplex)
  - [Computing Decision Losses on
    Sets](#computing-decision-losses-on-sets)
- [Objective](#objective)
  - [Integral Objective](#integral-objective)
  - [Max-Increase Objective](#max-increase-objective)
  - [Vertex Objective](#vertex-objective)
- [End.](#end.)

This paper is in the context of Reinforcement Learning. The goal of the
paper is to perform **ideal** simplifying abstractions by realizing the
utility structure of the decisions.

Modern machine learning systems deal with vast amounts of complex data,
like extremely detailed images or graphs with billions of nodes. How can
machine learning methods effectively align with pythonreal-world
decision-making based on such large-scale data? Moreover, how can one
manage domains where the problem’s complexity makes it challenging to
collect sufficient data for predictive model to comprehend the full
scope?

An agent observes ![x](https://latex.codecogs.com/svg.latex?x "x"), and
given a model
![p\_\theta(x, z)](https://latex.codecogs.com/svg.latex?p_%5Ctheta%28x%2C%20z%29 "p_\theta(x, z)")
of
![p(x,z)](https://latex.codecogs.com/svg.latex?p%28x%2Cz%29 "p(x,z)"),
returns an action ![a](https://latex.codecogs.com/svg.latex?a "a")
following the policy
![\pi(\mathcal{Z})](https://latex.codecogs.com/svg.latex?%5Cpi%28%5Cmathcal%7BZ%7D%29 "\pi(\mathcal{Z})").
Domain knowledge about the task is represented as a loss function
![l:\mathcal{Z} \times A \to \mathbb{R}](https://latex.codecogs.com/svg.latex?l%3A%5Cmathcal%7BZ%7D%20%5Ctimes%20A%20%5Cto%20%5Cmathbb%7BR%7D "l:\mathcal{Z} \times A \to \mathbb{R}")
measuring the cost of performing action
![a](https://latex.codecogs.com/svg.latex?a "a") when the outcome is
![x](https://latex.codecogs.com/svg.latex?x "x"). The loss function can
be represented as
![L\_{ij} = l(z_i, a_j)](https://latex.codecogs.com/svg.latex?L_%7Bij%7D%20%3D%20l%28z_i%2C%20a_j%29 "L_{ij} = l(z_i, a_j)").
Further,
![p = \\p(z_i\|x)\\\_i](https://latex.codecogs.com/svg.latex?p%20%3D%20%5C%7Bp%28z_i%7Cx%29%5C%7D_i "p = \{p(z_i|x)\}_i")
is a vector in
![\mathbb{R}^C](https://latex.codecogs.com/svg.latex?%5Cmathbb%7BR%7D%5EC "\mathbb{R}^C")
taking values in the probability simplex
![\Delta^C](https://latex.codecogs.com/svg.latex?%5CDelta%5EC "\Delta^C").

# Decision-Theoretic Context

Our goal is to find the optimal partition of the outcome set
![\mathcal{Z}](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BZ%7D "\mathcal{Z}")
such that the net loss of quality of decision-making is counterbalanced
by the decrease in the computational resources required. We aim to hide
information that is not required in the decision making process. We
consider the **H-entropy**


<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?H_l%28p%29%20%3D%20%5Cinf_%7Ba%20%5Cin%20A%7D%20%5Cmathbb%7BE%7D_%7Bp%28z%7Cx%29%7Dl%28%5Cmathcal%7BZ%7D%2C%20a%29%20%3D%20%5Cmin_%7Ba%20%5Cin%20A%7D%28Lp%29">
    <img src="https://latex.codecogs.com/svg.latex?H_l%28p%29%20%3D%20%5Cinf_%7Ba%20%5Cin%20A%7D%20%5Cmathbb%7BE%7D_%7Bp%28z%7Cx%29%7Dl%28%5Cmathcal%7BZ%7D%2C%20a%29%20%3D%20%5Cmin_%7Ba%20%5Cin%20A%7D%28Lp%29" alt="H_l(p)">
  </a>
</p>

This is the “least possible expected loss” aka “Bayes optimal loss”. We
can quantify the increase in the H-entropy (suboptimality gap) caused by
partitioning the support
![\mathcal{Z}](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BZ%7D "\mathcal{Z}"):

<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?%5Cdelta%28q%2Cp%29%20%3D%20H_%7B%5Ctilde%7Bl%7D%7D%28q%29%20-%20H_l%28p%29">
    <img src="https://latex.codecogs.com/svg.latex?%5Cdelta%28q%2Cp%29%20%3D%20H_%7B%5Ctilde%7Bl%7D%7D%28q%29%20-%20H_l%28p%29" alt="\delta(q,p)">
  </a>
</p>
We achieve this by noticing that every partition is a culmination of
*simplex folds*.

## Fold a Simplex

Fold
![f\_{i \to j}](https://latex.codecogs.com/svg.latex?f_%7Bi%20%5Cto%20j%7D "f_{i \to j}")
“buckets” ![z_i](https://latex.codecogs.com/svg.latex?z_i "z_i") and
![z_j](https://latex.codecogs.com/svg.latex?z_j "z_j") together reducing
the dimension of the support
![\mathcal{Z}](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BZ%7D "\mathcal{Z}").

## Computing Decision Losses on Sets

How is the loss function calculated after we consider the abstraction?
Well, naturally, we consider the worst-case extension!

<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?%5Ctilde%7Bl%7D%28S%2C%20a%29%20%3D%20%5Cmax_%7Bz%20%5Cin%20S%7D%20l%28z%2C%20a%29%2C%20%5C%20S%20%5Csubset%20%5Cmathcal%7BZ%7D">
    <img src="https://latex.codecogs.com/svg.latex?%5Ctilde%7Bl%7D%28S%2C%20a%29%20%3D%20%5Cmax_%7Bz%20%5Cin%20S%7D%20l%28z%2C%20a%29%2C%20%5C%20S%20%5Csubset%20%5Cmathcal%7BZ%7D" alt="\tilde{l}(S, a)">
  </a>
</p>

It is clear that folding increases H-entropy.

# Objective

Now our goal is to, at each step, find the optimal simplex fold. We find

<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?i%5E%7B%5Cstar%7D%2C%20j%5E%7B%5Cstar%7D%20%3D%20%5Carg%20%5Cmin_%7Bi%2Cj%2C%20i%5Cneq%20j%7D%20L%28i%2C%20j%2C%20L%29">
    <img src="https://latex.codecogs.com/svg.latex?i%5E%7B%5Cstar%7D%2C%20j%5E%7B%5Cstar%7D%20%3D%20%5Carg%20%5Cmin_%7Bi%2Cj%2C%20i%5Cneq%20j%7D%20L%28i%2C%20j%2C%20L%29" alt="i^{\star}, j^{\star}">
  </a>
</p>

We can achieve this by three ways:

- Integral Objective

- Max-Increase Objective

- Vertex Objective

## Integral Objective

Goal: Use the average amount of suboptimality gap over the probability
simplex as the Objective to find ideal abstractions.

<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?L%20%3D%20%5Cfrac%201%20%5Clambda%20%5Cint_%7B%5CDelta%5EC%7D%20%5BH_%7B%5Ctilde%7Bl%7D%7D%28f_%7Bi%20%5Cto%20j%7D%28%5Cmathbf%7Bp%7D%29%29%20-%20H_l%28%5Cmathbf%7Bp%7D%29%20%5Ctext%7Bd%7D%5Cmathbf%7Bp%7D%5D">
    <img src="https://latex.codecogs.com/svg.latex?L%20%3D%20%5Cfrac%201%20%5Clambda%20%5Cint_%7B%5CDelta%5EC%7D%20%5BH_%7B%5Ctilde%7Bl%7D%7D%28f_%7Bi%20%5Cto%20j%7D%28%5Cmathbf%7Bp%7D%29%29%20-%20H_l%28%5Cmathbf%7Bp%7D%29%20%5Ctext%7Bd%7D%5Cmathbf%7Bp%7D%5D" alt="L = \frac 1 \lambda \int_{\Delta^C} [H_{\tilde{l}}(f_{i \to j}(\mathbf{p})) - H_l(\mathbf{p}) \text{d}\mathbf{p}]">
  </a>
</p>

To calculate this computationally efficiently, we perform a monte-carlo
estimate. We pick ![N](https://latex.codecogs.com/svg.latex?N "N")
points on the probability simplex and find the average of the
suboptimality gap over those
![N](https://latex.codecogs.com/svg.latex?N "N") points.

<table>
<colgroup>
<col style="width: 23%" />
<col style="width: 76%" />
</colgroup>
<thead>
<tr class="header">
<th>Code</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><code>generate_points_in_simplex(N,c)</code></td>
<td>Generates <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20N" alt="N"
title="N" class="math inline" /> random points on the probability
simplex <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20%5CDelta%5Ec"
alt="\Delta^c" title="\Delta^c" class="math inline" />.</td>
</tr>
<tr class="even">
<td><code>fold(p, L, i, j)</code></td>
<td>Folds <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20%5Cmathbf%7Bp%7D%20%5Cto%20%5Cmathbf%7Bq%7D"
alt="\mathbf{p} \to \mathbf{q}" title="\mathbf{p} \to \mathbf{q}"
class="math inline" /> and <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20%5Cmathbf%7BL%7D%20%5Cto%20%5Cmathbf%7B%5Ctilde%7BL%7D%7D"
alt="\mathbf{L} \to \mathbf{\tilde{L}}"
title="\mathbf{L} \to \mathbf{\tilde{L}}" class="math inline" /> by
deleting <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20%5Cmax%28i%2Cj%29"
alt="\max(i,j)" title="\max(i,j)" class="math inline" />, column in <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20L" alt="L"
title="L" class="math inline" /> and row in <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20p" alt="p"
title="p" class="math inline" />, and assimilating it in <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20%5Cmin%28i%2Cj%29"
alt="\min(i,j)" title="\min(i,j)" class="math inline" /> row in <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20L" alt="L"
title="L" class="math inline" /> and column <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20p" alt="p"
title="p" class="math inline" />.</td>
</tr>
<tr class="odd">
<td><p><code>H_p = np.einsum("ac,cb-&gt;ab", L, p).min(axis = 0)</code></p>
<p><code>H_q = np.einsum("ac,cb-&gt;ab", L, q).min(axis = 0)</code></p></td>
<td><strong>einsum(“ac, cb-&gt;ab”)</strong> performs <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20Lp" alt="Lp"
title="Lp" class="math inline" /> and <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20Lq" alt="Lq"
title="Lq" class="math inline" /> respectively, and finds the minimum of
all the rows.</td>
</tr>
<tr class="even">
<td><code>i_fold, j_fold = np.unravel_index(np.argmin(M), M.shape</code></td>
<td><strong>unravel_index</strong> “opens” the array and argmin finds
the index with the associate minimum value.</td>
</tr>
</tbody>
</table>

## Max-Increase Objective

Goal: Use the max-increase suboptimality gap over the probability
simplex as the Objective to find ideal abstractions.

<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?L%20%3D%20%5Csup_%7Bp%20%5Cin%20%5CDelta%5EC%7D%5BH_%7B%5Ctilde%7Bl%7D%28f_%7Bi%20%5Cto%20j%7D%28%5Cmathbf%7Bp%7D%29%7D%20-%20H_l%28%5Cmathbf%7Bp%7D%29%5D%20%3D%20%5Cmax_%7Bp%20%5Cin%20%5CDelta%5EC%7D%20%5BH_%7B%5Ctilde%7Bl%7D%28f_%7Bi%20%5Cto%20j%7D%28%5Cmathbf%7Bp%7D%29%7D%20-%20H_l%28%5Cmathbf%7Bp%7D%29%5D">
    <img src="https://latex.codecogs.com/svg.latex?L%20%3D%20%5Csup_%7Bp%20%5Cin%20%5CDelta%5EC%7D%5BH_%7B%5Ctilde%7Bl%7D%28f_%7Bi%20%5Cto%20j%7D%28%5Cmathbf%7Bp%7D%29%7D%20-%20H_l%28%5Cmathbf%7Bp%7D%29%5D%20%3D%20%5Cmax_%7Bp%20%5Cin%20%5CDelta%5EC%7D%20%5BH_%7B%5Ctilde%7Bl%7D%28f_%7Bi%20%5Cto%20j%7D%28%5Cmathbf%7Bp%7D%29%7D%20-%20H_l%28%5Cmathbf%7Bp%7D%29%5D" alt="L = \sup_{p \in \Delta^C}[H_{\tilde{l}(f_{i \to j}(\mathbf{p})} - H_l(\mathbf{p})] = \max_{p \in \Delta^C} [H_{\tilde{l}(f_{i \to j}(\mathbf{p})} - H_l(\mathbf{p})]">
  </a>
</p>

To calculate this, we notice that

<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?L%20%3D%20%5Cmax_%7Bp%20%5Cin%20%5CDelta%5EC%7D%20%5Cleft%5B%5Cunderbrace%7B%5Cmin_%7Ba%7D%28%5Ctilde%7BL%7Dq%29%7D_%7B%5Ctext%7Bconcave%7D%7D%20-%20%5Cunderbrace%7B%5Cmin_%7Ba%7D%28Lp%29%7D_%7B%5Ctext%7Bconcave%7D%7D%5Cright%5D%20%3D%20%5Cmax_%7Bp%20%5Cin%20%5CDelta%5EC%7D%20%5Cleft%5BQ%28q%29%20-%20P%28p%29%5Cright%5D">
    <img src="https://latex.codecogs.com/svg.latex?L%20%3D%20%5Cmax_%7Bp%20%5Cin%20%5CDelta%5EC%7D%20%5Cleft%5B%5Cunderbrace%7B%5Cmin_%7Ba%7D%28%5Ctilde%7BL%7Dq%29%7D_%7B%5Ctext%7Bconcave%7D%7D%20-%20%5Cunderbrace%7B%5Cmin_%7Ba%7D%28Lp%29%7D_%7B%5Ctext%7Bconcave%7D%7D%5Cright%5D%20%3D%20%5Cmax_%7Bp%20%5Cin%20%5CDelta%5EC%7D%20%5Cleft%5BQ%28q%29%20-%20P%28p%29%5Cright%5D" alt="L = \max_{p \in \Delta^C} \left[\underbrace{\min_{a}(\tilde{L}q)}_{\text{concave}} - \underbrace{\min_{a}(Lp)}_{\text{concave}}\right] = \max_{p \in \Delta^C} \left[Q(q) - P(p)\right]">
  </a>
</p>

![L](https://latex.codecogs.com/svg.latex?L "L") is a difference of
concave function, and thus we use an algorithm to solve problems in the
class *difference of convex or concave problems.*

What do we do? At every point in the algorithm, we start with a
![p^k](https://latex.codecogs.com/svg.latex?p%5Ek "p^k") (starting with
![0](https://latex.codecogs.com/svg.latex?0 "0")), linearize
![P](https://latex.codecogs.com/svg.latex?P "P") around
![p^k](https://latex.codecogs.com/svg.latex?p%5Ek "p^k") and then
optimize ![L](https://latex.codecogs.com/svg.latex?L "L") around that
linearization of ![P](https://latex.codecogs.com/svg.latex?P "P"). Using
this we find the next point
![p^{k+1}](https://latex.codecogs.com/svg.latex?p%5E%7Bk%2B1%7D "p^{k+1}"),
and keep repeating this process until the absolute change in the
suboptimality gap is below the convergence threshold
![\delta](https://latex.codecogs.com/svg.latex?%5Cdelta "\delta")
(`self.delta`).

How is ![P](https://latex.codecogs.com/svg.latex?P "P") linearized?
Well, for a ![p^k](https://latex.codecogs.com/svg.latex?p%5Ek "p^k") we
find the subgradient
![g_k](https://latex.codecogs.com/svg.latex?g_k "g_k") of
![\min\_{a}(Lp)](https://latex.codecogs.com/svg.latex?%5Cmin_%7Ba%7D%28Lp%29 "\min_{a}(Lp)").

<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?g_k%20%3D%20%5Cfrac%7B1%7D%7B%5Cgamma%7D%20L_m%5ET">
    <img src="https://latex.codecogs.com/svg.latex?g_k%20%3D%20%5Cfrac%7B1%7D%7B%5Cgamma%7D%20L_m%5ET" alt="g_k = \frac{1}{\gamma} L_m^T">
  </a>
</p>

where
![m = \arg \min\_{a} (Lp)](https://latex.codecogs.com/svg.latex?m%20%3D%20%5Carg%20%5Cmin_%7Ba%7D%20%28Lp%29 "m = \arg \min_{a} (Lp)")
and ![\gamma](https://latex.codecogs.com/svg.latex?%5Cgamma "\gamma") is
the slowdown parameter of the objective subgradient (`self.gamma`).

<table>
<colgroup>
<col style="width: 27%" />
<col style="width: 72%" />
</colgroup>
<thead>
<tr class="header">
<th>Code</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><code>X(self,i,j)</code></td>
<td>This is an alternate implementation to generate <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20%5Cmathbf%7Bq%7D"
alt="\mathbf{q}" title="\mathbf{q}" class="math inline" /> as <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20%5Cmathbf%7Bq%7D%20%3D%20X%5Cmathbf%7Bp%7D"
alt="\mathbf{q} = X\mathbf{p}" title="\mathbf{q} = X\mathbf{p}"
class="math inline" />. Here, <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20X%20%3D%20I%20%2B%20E_%7Bij%7D%20-%20E_%7Bjj%7D"
alt="X = I + E_{ij} - E_{jj}" title="X = I + E_{ij} - E_{jj}"
class="math inline" />. We use this because this is more efficient and
reusable.</td>
</tr>
<tr class="even">
<td><code>Lt(self,i,j)</code></td>
<td>To generate <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20%5Ctilde%7BL%7D"
alt="\tilde{L}" title="\tilde{L}" class="math inline" /> from <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20L" alt="L"
title="L" class="math inline" /> . Largely the same as the
<strong>Integral Objective</strong> but we take care of the dimensions
of <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20q" alt="q"
title="q" class="math inline" /> as that is implemented slightly
differently.</td>
</tr>
<tr class="odd">
<td><code>objectiveFunction(self, p, pk, i, j)</code></td>
<td>Defines the suboptimality gap objective function after the
linearization of <img style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20P" alt="P"
title="P" class="math inline" /> around <img
style="vertical-align:middle"
src="https://latex.codecogs.com/svg.latex?%5Ctextstyle%20p%5Ek"
alt="p^k" title="p^k" class="math inline" />.</td>
</tr>
<tr class="even">
<td><code>suboptimalityGap(self, p, i, j)</code></td>
<td>Defines the suboptimality gap objective function.</td>
</tr>
<tr class="odd">
<td><code>linear_optimizer(self, pk, i, j)</code></td>
<td>This function performs step-by-step optimization of the linearized
<code>objectiveFunction</code> .</td>
</tr>
<tr class="even">
<td><code>DCOptimizer(self, i, j)</code></td>
<td>Performs the complete optimization and stops when change in the
suboptimality gap dips below the convergence threshold
<code>self.delta</code> .</td>
</tr>
<tr class="odd">
<td><p><code>probability_constraint(self, p)</code></p>
<p><code>bounds = [(0,1) for _ in range(self.c)]</code></p>
<p><code>constraints = ({'type':'eq', 'fun': self.probability_constraint})</code></p></td>
<td>Defines the constraints for <code>scipy.optimize</code> .</td>
</tr>
</tbody>
</table>

## Vertex Objective

Goal: Only care about the vertices of
![\Delta^c](https://latex.codecogs.com/svg.latex?%5CDelta%5Ec "\Delta^c")
to find suboptimality gap.

<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D_%7Bi%2Cj%7D%20%3D%20%5Cleft%5BH_l%28p%5E%7B%28i%29%7D%29%20-%20H_l%28p%5E%7B%28j%29%7D%29%5Cright%5D%20%3D%20%5B%5Cmin_a%28L_i%29%20-%20%5Cmin_a%28L_i%29%5D">
    <img src="https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D_%7Bi%2Cj%7D%20%3D%20%5Cleft%5BH_l%28p%5E%7B%28i%29%7D%29%20-%20H_l%28p%5E%7B%28j%29%7D%29%5Cright%5D%20%3D%20%5B%5Cmin_a%28L_i%29%20-%20%5Cmin_a%28L_i%29%5D" alt="\mathcal{L}_{i,j} = \left[H_l(p^{(i)}) - H_l(p^{(j)})\right] = [\min_a(L_i) - \min_a(L_i)]">
  </a>
</p>


<p align="center">
  <a href="https://latex.codecogs.com/svg.latex?i%5E%7B%5Cstar%7D%2C%20j%5E%7B%5Cstar%7D%20%3D%20%5Cmin_%7Bi%20%5Cneq%20j%7D%20L_%7Bi%2Cj%7D">
    <img src="https://latex.codecogs.com/svg.latex?i%5E%7B%5Cstar%7D%2C%20j%5E%7B%5Cstar%7D%20%3D%20%5Cmin_%7Bi%20%5Cneq%20j%7D%20L_%7Bi%2Cj%7D" alt="i^{\star}, j^{\star} = \min_{i \neq j} L_{i,j}">
  </a>
</p>

The implementation is fairly trivial to understand.

# End.
