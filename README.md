
# Ideal Abstractions for Decision-Focused Learning

This paper is in the context of Reinforcement Learning. The goal of the
paper is to perform **ideal** simplifying abstractions by realizing the
utility structure of the decisions.

Modern machine learning systems deal with vast amounts of complex data,
like extremely detailed images or graphs with billions of nodes. How can
machine learning methods effectively align with real-world
decision-making based on such large-scale data? Moreover, how can one
manage domains where the problem’s complexity makes it challenging to
collect sufficient data for predictive model to comprehend the full
scope?

An agent observes *x*, and given a model *p*<sub>*θ*</sub>(*x*,*z*) of
*p*(*x*,*z*), returns an action *a* following the policy *π*(𝒵). Domain
knowledge about the task is represented as a loss function
*l* : 𝒵 × *A* → ℝ measuring the cost of performing action *a* when the
outcome is *x*. The loss function can be represented as
*L*<sub>*i**j*</sub> = 𝓁(*z*<sub>*i*</sub>,*a*<sub>*j*</sub>). Further,
*p* = {*p*(*z*<sub>*i*</sub>|*x*)}<sub>*i*</sub> is a vector in
ℝ<sup>*C*</sup> taking values in the probability simplex
*Δ*<sup>*C*</sup>.

# Decision-Theoretic Context

Our goal is to find the optimal partition of the outcome set 𝒵 such that
the net loss of quality of decision-making is counterbalanced by the
decrease in the computational resources required. We aim to hide
information that is not required in the decision making process. We
consider the **H-entropy**

*H*<sub>*l*</sub>(*p*) = inf<sub>*a* ∈ *A*</sub>𝔼<sub>*p*(*z*|*x*)</sub>𝓁(𝒵,*a*) = min<sub>*a* ∈ *A*</sub>(*L**p*)

This is the “least possible expected loss” aka “Bayes optimal loss”. We
can quantify the increase in the H-entropy (suboptimality gap) caused by
partitioning the support 𝒵:

*δ*(*q*,*p*) = *H*<sub>*l̃*</sub>(*q*) − *H*<sub>*l*</sub>(*p*)

We achieve this by noticing that every partition is a culmination of
*simplex folds*.

## Fold a Simplex

Fold *f*<sub>*i* → *j*</sub> “buckets” *z*<sub>*i*</sub> and
*z*<sub>*j*</sub> together reducing the dimension of the support 𝒵.

## Computing Decision Losses on Sets

How is the loss function calculated after we consider the abstraction?
Well, naturally, we consider the worst-case extension!

$$
\tilde{\mathcal{l}}(S, a) = \max\_{z \in S} \mathcal{l}(z, a), \\S \subset \mathcal{Z}
$$

It is clear that folding increases H-entropy.

# Objective

Now our goal is to, at each step, find the optimal simplex fold. We find

*i*<sup>\*</sup>, *j*<sup>\*</sup> = arg min<sub>*i*, *j*, *i* ≠ *j*</sub>ℒ(*i*,*j*,*L*)

We can achieve this by three ways:

-   Integral Objective

-   Max-Increase Objective

-   Vertex Objective

## Integral Objective

Goal: Use the average amount of suboptimality gap over the probability
simplex as the Objective to find ideal abstractions.

$$\mathcal{L} = \frac 1 \lambda \int\_{\Delta^C} \[H\_{\tilde{l}}(f\_{i \to j}(\mathbf{p})) - H_l(p) \text{d}\mathbf{p}\]$$

To calculate this computationally efficiently, we perform a monte-carlo
estimate. We pick *N* points on the probability simplex and find the
average of the suboptimality gap over those *N* points.

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
<td><div class="sourceCode" id="cb1"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a>generate_points_in_simplex(N,c)</span></code></pre></div></td>
<td>Generates <span class="math inline"><em>N</em></span> random points
on the probability simplex <span
class="math inline"><em>Δ</em><sup><em>c</em></sup></span>.</td>
</tr>
<tr class="even">
<td><div class="sourceCode" id="cb2"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb2-1"><a href="#cb2-1" aria-hidden="true" tabindex="-1"></a>fold(p, L, i, j)</span></code></pre></div></td>
<td>Folds <span
class="math inline"><strong>p</strong> → <strong>q</strong></span> and
<span class="math inline"><strong>L</strong> → <strong>L̃</strong></span>
by deleting <span
class="math inline">max (<em>i</em>,<em>j</em>)</span>, column in <span
class="math inline"><em>L</em></span> and row in <span
class="math inline"><em>p</em></span>, and assimilating it in <span
class="math inline">min (<em>i</em>,<em>j</em>)</span> row in <span
class="math inline"><em>L</em></span> and column <span
class="math inline"><em>p</em></span>.</td>
</tr>
<tr class="odd">
<td><div class="sourceCode" id="cb3"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb3-1"><a href="#cb3-1" aria-hidden="true" tabindex="-1"></a>H_p <span class="op">=</span> np.einsum(<span class="st">&quot;ac,cb-&gt;ab&quot;</span>, L, p).<span class="bu">min</span>(axis <span class="op">=</span> <span class="dv">0</span>)</span></code></pre></div>
<div class="sourceCode" id="cb4"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb4-1"><a href="#cb4-1" aria-hidden="true" tabindex="-1"></a>H_q <span class="op">=</span> np.einsum(<span class="st">&quot;ac,cb-&gt;ab&quot;</span>, L, q).<span class="bu">min</span>(axis <span class="op">=</span> <span class="dv">0</span>)</span></code></pre></div></td>
<td><strong>einsum(“ac, cb-&gt;ab”)</strong> performs <span
class="math inline"><em>L</em><em>p</em></span> and <span
class="math inline"><em>L</em><em>q</em></span> respectively, and finds
the minimum of all the rows.</td>
</tr>
<tr class="even">
<td><div class="sourceCode" id="cb5"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb5-1"><a href="#cb5-1" aria-hidden="true" tabindex="-1"></a>i_fold, j_fold <span class="op">=</span> np.unravel_index(np.argmin(M), M.shape)</span></code></pre></div></td>
<td><strong>unravel_index</strong> “opens” the array and argmin finds
the index with the associate minimum value.</td>
</tr>
</tbody>
</table>

## Max-Increase Objective

Goal: Use the max-increase suboptimality gap over the probability
simplex as the Objective to find ideal abstractions.

ℒ = sup<sub>*p* ∈ *Δ*<sup>*C*</sup></sub>\[*H*<sub>*l̃*(*f*<sub>*i* → *j*</sub>(**p**)</sub>−*H*<sub>*l*</sub>(**p**)\] = max<sub>*p* ∈ *Δ*<sup>*C*</sup></sub>\[*H*<sub>*l̃*(*f*<sub>*i* → *j*</sub>(**p**)</sub>−*H*<sub>*l*</sub>(**p**)\]

To calculate this, we notice that

$$
\mathcal{L} = \max\_{p \in \Delta^C} \left\[\underbrace{\min\_{a}(\tilde{L}q)}\_{\text{concave}} - \underbrace{\min\_{a}(Lp)}\_{\text{concave}}\right\] = \max\_{p \in \Delta^C} \left\[Q(q) - P(p)\right\]
$$

ℒ is a difference of concave function, and thus we use an algorithm to
solve problems in the class *difference of convex or concave problems.*

What do we do? At every point in the algorithm, we start with a
*p*<sup>*k*</sup> (starting with 0), linearize *P* around
*p*<sup>*k*</sup> and then optimize ℒ around that linearization of *P*.
Using this we find the next point *p*<sup>*k* + 1</sup>, and keep
repeating this process until the absolute change in the suboptimality
gap is below the convergence threshold *δ* (`self.delta`).

How is *P* linearized? Well, for a *p*<sup>*k*</sup> we find the
subgradient *g*<sub>*k*</sub> of min<sub>*a*</sub>(*L**p*).

$$
g_k = \frac{1}{\gamma} L_m^T
$$

where *m* = arg min<sub>*a*</sub>(*L**p*) and *γ* is the slowdown
parameter of the objective subgradient (`self.gamma`).

<table>
<colgroup>
<col style="width: 26%" />
<col style="width: 73%" />
</colgroup>
<thead>
<tr class="header">
<th>Code</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><div class="sourceCode" id="cb1"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a>X(<span class="va">self</span>,i,j)</span></code></pre></div></td>
<td>This is an alternate implementation to generate <span
class="math inline"><strong>q</strong></span> as <span
class="math inline"><strong>q</strong> = <em>X</em><strong>p</strong></span>.
Here, <span
class="math inline"><em>X</em> = <em>I</em> + <em>E</em><sub><em>i</em><em>j</em></sub> − <em>E</em><sub><em>j</em><em>j</em></sub></span>.
We use this because this is more efficient and reusable.</td>
</tr>
<tr class="even">
<td><div class="sourceCode" id="cb2"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb2-1"><a href="#cb2-1" aria-hidden="true" tabindex="-1"></a>Lt(<span class="va">self</span>,i,j)</span></code></pre></div></td>
<td>To generate <span class="math inline"><em>L̃</em></span> from <span
class="math inline"><em>L</em></span> . Largely the same as the
<strong>Integral Objective</strong> but we take care of the dimensions
of <span class="math inline"><em>q</em></span> as that is implemented
slightly differently.</td>
</tr>
<tr class="odd">
<td><div class="sourceCode" id="cb3"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb3-1"><a href="#cb3-1" aria-hidden="true" tabindex="-1"></a>objectiveFunction(<span class="va">self</span>, p, pk, i, j)</span></code></pre></div></td>
<td>Defines the suboptimality gap objective function after the
linearization of <span class="math inline"><em>P</em></span> around
<span class="math inline"><em>p</em><sup><em>k</em></sup></span>.</td>
</tr>
<tr class="even">
<td><div class="sourceCode" id="cb4"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb4-1"><a href="#cb4-1" aria-hidden="true" tabindex="-1"></a>suboptimalityGap(<span class="va">self</span>, p, i, j)</span></code></pre></div></td>
<td>Defines the suboptimality gap objective function.</td>
</tr>
<tr class="odd">
<td><div class="sourceCode" id="cb5"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb5-1"><a href="#cb5-1" aria-hidden="true" tabindex="-1"></a>linear_optimizer(<span class="va">self</span>, pk, i, j)</span></code></pre></div></td>
<td>This function performs step-by-step optimization of the linearized
<code>objectiveFunction</code> .</td>
</tr>
<tr class="even">
<td><div class="sourceCode" id="cb6"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb6-1"><a href="#cb6-1" aria-hidden="true" tabindex="-1"></a>DCOptimizer(<span class="va">self</span>, i, j)</span></code></pre></div></td>
<td>Performs the complete optimization and stops when change in the
suboptimality gap dips below the convergence threshold
<code>self.delta</code> .</td>
</tr>
<tr class="odd">
<td><div class="sourceCode" id="cb7"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb7-1"><a href="#cb7-1" aria-hidden="true" tabindex="-1"></a>probability_constraint(<span class="va">self</span>, p)</span></code></pre></div>
<div class="sourceCode" id="cb8"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb8-1"><a href="#cb8-1" aria-hidden="true" tabindex="-1"></a>bounds <span class="op">=</span> [(<span class="dv">0</span>,<span class="dv">1</span>) <span class="cf">for</span> _ <span class="kw">in</span> <span class="bu">range</span>(<span class="va">self</span>.c)]</span>
<span id="cb8-2"><a href="#cb8-2" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb8-3"><a href="#cb8-3" aria-hidden="true" tabindex="-1"></a>constraints <span class="op">=</span> ({<span class="st">&#39;type&#39;</span>:<span class="st">&#39;eq&#39;</span>, <span class="st">&#39;fun&#39;</span>: <span class="va">self</span>.probability_constraint})</span></code></pre></div></td>
<td>Defines the constraints for <code>scipy.optimize</code> .</td>
</tr>
</tbody>
</table>

## Vertex Objective

Goal: Only care about the vertices of *Δ*<sup>*c*</sup> to find
suboptimality gap.

ℒ<sub>𝒾, 𝒿</sub> = \[*H*<sub>*l*</sub>(*p*<sup>(*i*)</sup>)−*H*<sub>*l*</sub>(*p*<sup>(*j*)</sup>)\] = \[min<sub>*a*</sub>(*L*<sub>*i*</sub>)−min<sub>*a*</sub>(*L*<sub>*i*</sub>)\]

*i*<sup>\*</sup>, *j*<sup>\*</sup> = min<sub>*i* ≠ *j*</sub>ℒ<sub>*i*, *j*</sub>

The implementation is fairly trivial to understand.

# End.
