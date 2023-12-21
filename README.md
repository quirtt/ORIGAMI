---
editor: visual
title: Ideal Abstractions for Decisions-Focused Learning
toc-title: Table of contents
---

-   [Decision-Theoretic
    Context](#decision-theoretic-context){#toc-decision-theoretic-context}
    -   [Fold a Simplex](#fold-a-simplex){#toc-fold-a-simplex}
    -   [Computing Decision Losses on
        Sets](#computing-decision-losses-on-sets){#toc-computing-decision-losses-on-sets}
-   [Objective](#objective){#toc-objective}
    -   [Integral
        Objective](#integral-objective){#toc-integral-objective}
    -   [Max-Increase
        Objective](#max-increase-objective){#toc-max-increase-objective}
    -   [Vertex Objective](#vertex-objective){#toc-vertex-objective}
-   [End.](#end.){#toc-end.}

This paper is in the context of Reinforcement Learning. The goal of the
paper is to perform **ideal** simplifying abstractions by realizing the
utility structure of the decisions.

Modern machine learning systems deal with vast amounts of complex data,
like extremely detailed images or graphs with billions of nodes. How can
machine learning methods effectively align with pythonreal-world
decision-making based on such large-scale data? Moreover, how can one
manage domains where the problem's complexity makes it challenging to
collect sufficient data for predictive model to comprehend the full
scope?

An agent observes $x$, and given a model $p_\theta(x, z)$ of $p(x,z)$,
returns an action $a$ following the policy $\pi(\mathcal{Z})$. Domain
knowledge about the task is represented as a loss function
$l:\mathcal{Z} \times A \to \mathbb{R}$ measuring the cost of performing
action $a$ when the outcome is $x$. The loss function can be represented
as $L_{ij} = l(z_i, a_j)$. Further, $p = \{p(z_i|x)\}_i$ is a vector in
$\mathbb{R}^C$ taking values in the probability simplex $\Delta^C$.

# Decision-Theoretic Context

Our goal is to find the optimal partition of the outcome set
$\mathcal{Z}$ such that the net loss of quality of decision-making is
counterbalanced by the decrease in the computational resources required.
We aim to hide information that is not required in the decision making
process. We consider the **H-entropy**

$$
H_l(p) = \inf_{a \in A} \mathbb{E}_{p(z|x)}l(\mathcal{Z}, a) = \min_{a \in A}(Lp)
$$

This is the "least possible expected loss" aka "Bayes optimal loss". We
can quantify the increase in the H-entropy (suboptimality gap) caused by
partitioning the support $\mathcal{Z}$:

$$
\delta(q,p) = H_{\tilde{l}}(q) - H_l(p) 
$$

We achieve this by noticing that every partition is a culmination of
*simplex folds*.

## Fold a Simplex

Fold $f_{i \to j}$ "buckets" $z_i$ and $z_j$ together reducing the
dimension of the support $\mathcal{Z}$.

## Computing Decision Losses on Sets

How is the loss function calculated after we consider the abstraction?
Well, naturally, we consider the worst-case extension!

$$
\tilde{l}(S, a) = \max_{z \in S} l(z, a), \ S \subset \mathcal{Z}
$$

It is clear that folding increases H-entropy.

# Objective

Now our goal is to, at each step, find the optimal simplex fold. We find

$$
i^{\star}, j^{\star} = \arg \min_{i,j, i\neq j} L(i, j, L)
$$

We can achieve this by three ways:

-   Integral Objective

-   Max-Increase Objective

-   Vertex Objective

## Integral Objective

Goal: Use the average amount of suboptimality gap over the probability
simplex as the Objective to find ideal abstractions.

$$L = \frac 1 \lambda \int_{\Delta^C} [H_{\tilde{l}}(f_{i \to j}(\mathbf{p})) - H_l(p) \text{d}\mathbf{p}]$$

To calculate this computationally efficiently, we perform a monte-carlo
estimate. We pick $N$ points on the probability simplex and find the
average of the suboptimality gap over those $N$ points.

+----------------+-----------------------------------------------------+
| Code           | Purpose                                             |
+================+=====================================================+
| `gene          | Generates $N$ random points on the probability      |
| rate_points_in | simplex $\Delta^c$.                                 |
| _simplex(N,c)` |                                                     |
+----------------+-----------------------------------------------------+
| `fol           | Folds $\mathbf{p} \to \mathbf{q}$ and               |
| d(p, L, i, j)` | $\mathbf{L} \to \mathbf{\tilde{L}}$ by deleting     |
|                | $\max(i,j)$, column in $L$ and row in $p$, and      |
|                | assimilating it in $\min(i,j)$ row in $L$ and       |
|                | column $p$.                                         |
+----------------+-----------------------------------------------------+
| `H_p = n       | **einsum("ac, cb-\>ab")** performs $Lp$ and $Lq$    |
| p.einsum("ac,c | respectively, and finds the minimum of all the      |
| b->ab", L, p). | rows.                                               |
| min(axis = 0)` |                                                     |
|                |                                                     |
| `H_q = n       |                                                     |
| p.einsum("ac,c |                                                     |
| b->ab", L, q). |                                                     |
| min(axis = 0)` |                                                     |
+----------------+-----------------------------------------------------+
| `              | **unravel_index** "opens" the array and argmin      |
| i_fold, j_fold | finds the index with the associate minimum value.   |
|  = np.unravel_ |                                                     |
| index(np.argmi |                                                     |
| n(M), M.shape` |                                                     |
+----------------+-----------------------------------------------------+

## Max-Increase Objective

Goal: Use the max-increase suboptimality gap over the probability
simplex as the Objective to find ideal abstractions.

$$
L = \sup_{p \in \Delta^C}[H_{\tilde{l}(f_{i \to j}(\mathbf{p})} - H_l(\mathbf{p})] = \max_{p \in \Delta^C} [H_{\tilde{l}(f_{i \to j}(\mathbf{p})} - H_l(\mathbf{p})]
$$

To calculate this, we notice that

$$
L = \max_{p \in \Delta^C} \left[\underbrace{\min_{a}(\tilde{L}q)}_{\text{concave}} - \underbrace{\min_{a}(Lp)}_{\text{concave}}\right] = \max_{p \in \Delta^C} \left[Q(q) - P(p)\right]
$$

$L$ is a difference of concave function, and thus we use an algorithm to
solve problems in the class *difference of convex or concave problems.*

What do we do? At every point in the algorithm, we start with a $p^k$
(starting with $0$), linearize $P$ around $p^k$ and then optimize $L$
around that linearization of $P$. Using this we find the next point
$p^{k+1}$, and keep repeating this process until the absolute change in
the suboptimality gap is below the convergence threshold $\delta$
(`self.delta`).

How is $P$ linearized? Well, for a $p^k$ we find the subgradient $g_k$
of $\min_{a}(Lp)$.

$$
g_k = \frac{1}{\gamma} L_m^T
$$

where $m = \arg \min_{a} (Lp)$ and $\gamma$ is the slowdown parameter of
the objective subgradient (`self.gamma`).

+------------------+---------------------------------------------------+
| Code             | Purpose                                           |
+==================+===================================================+
| `X(self,i,j)`    | This is an alternate implementation to generate   |
|                  | $\mathbf{q}$ as $\mathbf{q} = X\mathbf{p}$. Here, |
|                  | $X = I + E_{ij} - E_{jj}$. We use this because    |
|                  | this is more efficient and reusable.              |
+------------------+---------------------------------------------------+
| `Lt(self,i,j)`   | To generate $\tilde{L}$ from $L$ . Largely the    |
|                  | same as the **Integral Objective** but we take    |
|                  | care of the dimensions of $q$ as that is          |
|                  | implemented slightly differently.                 |
+------------------+---------------------------------------------------+
| `objec           | Defines the suboptimality gap objective function  |
| tiveFunction(sel | after the linearization of $P$ around $p^k$.      |
| f, p, pk, i, j)` |                                                   |
+------------------+---------------------------------------------------+
| `                | Defines the suboptimality gap objective function. |
| suboptimalityGap |                                                   |
| (self, p, i, j)` |                                                   |
+------------------+---------------------------------------------------+
| `l               | This function performs step-by-step optimization  |
| inear_optimizer( | of the linearized `objectiveFunction` .           |
| self, pk, i, j)` |                                                   |
+------------------+---------------------------------------------------+
| `DCOptimi        | Performs the complete optimization and stops when |
| zer(self, i, j)` | change in the suboptimality gap dips below the    |
|                  | convergence threshold `self.delta` .              |
+------------------+---------------------------------------------------+
| `                | Defines the constraints for `scipy.optimize` .    |
| probability_cons |                                                   |
| traint(self, p)` |                                                   |
|                  |                                                   |
| `bounds =        |                                                   |
|  [(0,1) for _ in |                                                   |
|  range(self.c)]` |                                                   |
|                  |                                                   |
| `co              |                                                   |
| nstraints = ({'t |                                                   |
| ype':'eq', 'fun' |                                                   |
| : self.probabili |                                                   |
| ty_constraint})` |                                                   |
+------------------+---------------------------------------------------+

## Vertex Objective

Goal: Only care about the vertices of $\Delta^c$ to find suboptimality
gap.

$$
\mathcal{L_{i,j}} = \left[H_l(p^{(i)}) - H_l(p^{(j)})\right] = [\min_a(L_i) - \min_a(L_i)]
$$

$$
i^{\star}, j^{\star} = \min_{i \neq j} L_{i,j}
$$

The implementation is fairly trivial to understand.

# End.
