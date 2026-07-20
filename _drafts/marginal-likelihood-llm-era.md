---
layout: post
title: "The Marginal Likelihood is the Area Under Your Loss Curve"
description: >-
  The marginal likelihood, the cross-validation score, and the conditional (log) marginal likelihood are three functionals of the same object: an idealized one-pass loss curve. Seen this way, the classic debates about Bayesian model selection map directly onto things LLM researchers stare at every day — pretraining curves, scaling-law crossovers, and in-context learning curves — and the classic failure modes (misspecification, prior-data misfit) become statements about where curves plateau and how fast they fall.
date: 2026-07-20
author: Andreas Kirsch
---

<!--
NOTES FOR MOVING THIS DRAFT TO THE PERSONAL BLOG:
- Figures are referenced from /assets/img/2024-05-07-clml/ (this repo): four PNGs and one SVG.
  Copy that folder (responsive -480/-800/-1400 webp variants also exist there if wanted) and
  adjust paths; absolute /assets/... paths assume the site is served from the domain root —
  prepend a baseurl or use relative_url if not.
- Math uses kramdown-style $$...$$ delimiters (inline and display); figure captions avoid math
  so they render regardless of MathJax config.
- Footnotes use kramdown [^n] syntax.
- This is a rewrite of the ICLR 2024 blog post:
  https://iclr-blogposts.github.io/2024/blog/clml/
-->

**TL;DR.** By the chain rule, the negative log marginal likelihood — the log of the Bayesian evidence, negated — is exactly the *area under the loss curve* of an ideal Bayesian learner making a single pass over your dataset. Validation loss is the *final height* of that curve; the conditional log marginal likelihood (CLML) corresponds to the *area under its tail*. One curve, three functionals — and they can only rank two models differently when their loss curves *cross*, which happens when inductive-bias alignment (how fast a curve falls) is anti-correlated with expressivity (where it plateaus). That is the scaling-law crossover phenomenon, and it reappears at inference time in in-context learning curves. No functional is universally right; match it to the question: comparing priors and testing hypotheses → whole area; deploying a frozen model → final height; sequential prediction, compression, in-context learning → tail area.

## Two Runs, One Decision

You have two pretraining runs on the same data stream, one pass each. Run A gets off to a great start — its loss drops fast — but flattens out early. Run B spends the first chunk of training looking embarrassing, then keeps grinding and ends lower.

Which model is better?

If you answered "B, obviously — I ship the final checkpoint, not the training history," you have, perhaps without noticing, picked a *criterion*: the final height of the loss curve. It is not the criterion classical Bayesian model selection uses. The marginal likelihood — the quantity behind Bayes factors and "automatic Occam's razor" arguments — scores A against B by the **area under the loss curve**, early mistakes included. That is an identity, not an analogy, and it's the subject of this post. (On this particular pair of runs, the area might still favor B — if the late advantage is large enough to repay the early deficit — but it need not, and when the two criteria diverge, it matters which question each one answers.)

Once you see the criteria as curve functionals, the menu of Bayesian model-selection quantities — marginal likelihood, cross-validation, and the more recently proposed conditional marginal likelihood — snaps into one picture, and the classic debates become concrete questions about curve geometry. When can the rankings disagree? (Only when curves cross.) What makes curves cross? (A model's inductive-bias alignment — how fast its curve falls — trading off against its expressivity — where its curve plateaus.) Which functional should you use? (Depends on the question you're asking — and the answer changes if you care about in-context learning.)

This post is a substantially rewritten version of an [ICLR 2024 blog post](https://iclr-blogposts.github.io/2024/blog/clml/) of mine that made these points in heavier information-theoretic clothing; this version is written for people who spend their days looking at loss curves rather than posteriors.

## The Identity

Take a probabilistic model $$\mathcal{M}$$: a prior $$p(\theta \mid \mathcal{M})$$ over parameters plus a likelihood $$p(x \mid \theta, \mathcal{M})$$. The **marginal likelihood** (or **evidence**) of a dataset $$x_{1:N} = (x_1, \ldots, x_N)$$ integrates the parameters out:

$$
p(x_{1:N} \mid \mathcal{M}) = \int p(x_{1:N} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta.
$$

This is the quantity behind Bayes factors (a Bayes factor is the ratio of two models' marginal likelihoods — Bayesian statistics' analogue of a likelihood-ratio test), behind type-II maximum likelihood (maximizing the marginal likelihood over hyperparameters rather than parameters, a.k.a. empirical Bayes), and behind "automatic Occam's razor" arguments. It looks like a static score for how well the model explains the data. But apply the chain rule of probability:

$$
\log p(x_{1:N} \mid \mathcal{M}) = \sum_{n=1}^{N} \log p(x_n \mid x_{<n}, \mathcal{M}),
$$

and notice what each term is. Assuming, as usual, that data points are independent given the parameters, $$p(x_n \mid x_{<n}, \mathcal{M}) = \int p(x_n \mid \theta, \mathcal{M}) \, p(\theta \mid x_{<n}, \mathcal{M}) \, d\theta$$ — the same integral as the evidence above, with the prior replaced by the posterior after $$n-1$$ points. That is all "posterior predictive" means: the prediction of a learner that has *already conditioned on the first $$n-1$$ points*. Write the per-step loss as $$\ell_n = -\log p(x_n \mid x_{<n}, \mathcal{M})$$, and the negative log evidence becomes

$$
-\log p(x_{1:N} \mid \mathcal{M}) = \sum_{n=1}^{N} \ell_n :
$$

the summed one-step-ahead losses of an ideal Bayesian learner doing online learning over your dataset in a single pass — updating by exact conditioning rather than by gradient steps. Plot $$\ell_n$$ against $$n$$ and you have a loss curve; the negative log evidence is the area under it.

<figure>
  <img src="/assets/img/2024-05-07-clml/area_under_curve_1.00.png" alt="Left: a smooth decreasing loss curve with the area under it shaded. Right: the same with noisy per-step losses." style="max-width: 100%;">
  <figcaption><strong>The marginal likelihood is an area.</strong> Left: the expected one-step-ahead loss of an ideal Bayesian learner as a function of the number of observed points; the shaded area under the curve is the negative log marginal likelihood (in expectation). Right: the realized per-step losses on one dataset trace a noisy version of the same curve; their sum is the negative log evidence of that dataset.</figcaption>
</figure>

Three things follow immediately.

First, **the marginal likelihood is not a property of the trained model.** It is a property of the entire learning trajectory. The very first term charges the model for predicting $$x_1$$ from the bare prior. A model class that will eventually be excellent, but starts from a poorly aligned prior, pays for that start in every comparison — no matter how much data comes later, the early-curve area stays on the books.

Second, this is a **compression statement**. Feeding the one-step-ahead predictions into an arithmetic coder compresses the dataset to $$\sum_n \ell_n$$ nats (logs are natural throughout; the toy figures near the end use bits). This is Dawid's *prequential* view of model assessment and the [minimum-description-length](https://en.wikipedia.org/wiki/Minimum_description_length) view of the evidence. It's the same accounting that makes a language model a lossless compressor ([Delétang et al., 2023](https://arxiv.org/abs/2309.10668)), and prequential codes built from actual training runs are among the best practical description lengths we know for deep learning ([Blier & Ollivier, 2018](https://arxiv.org/abs/1802.07044)).

Third, the SGD analogy is closer than it might look. In the (near-)single-epoch regime of pretraining — where most tokens are seen once, even if curated subsets are repeated for a few epochs ([Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)) — the loss you log on each batch is computed *before* you update on that batch: every fresh batch is held-out data for the current parameters. The summed first-pass training loss is therefore the prequential code length of the corpus under your training pipeline as the learning algorithm: the practical stand-in for a negative log marginal likelihood, with an SGD step in place of Bayesian conditioning. (Second-epoch losses are not code lengths of anything new — you already paid for those tokens.) This correspondence is exactly what training-speed-based model selection exploits ([Lyle et al., 2020](https://arxiv.org/abs/2010.14499); [Ru et al., 2021](https://arxiv.org/abs/2006.04492)): "sum of training losses" is a marginal-likelihood surrogate, and "area under the training curve" is not a metaphor but the actual estimator.

Two honest caveats before the analogy gets taken further than it deserves. In the Bayesian correspondence, only the initialization and the architecture play the role of the *prior*; the optimizer and the learning-rate schedule play the role of the *inference mechanism* — the stand-in for conditioning. Areas therefore compare **learners, not just models**: two runs of the same architecture with different warmup or annealing schedules have very different areas and nearly identical final checkpoints, so area comparisons are only meaningful across matched training recipes. This sensitivity is not a bug in the framing but its message — the marginal likelihood scores the trajectory, not the artifact. (Final height is not fully innocent either: where you stop and how hard you anneal move it too.) Relatedly, the area is **order-sensitive** in a way the final height mostly is not: for non-exchangeable data streams — curricula, annealed mixtures, and SGD itself — different orderings give different areas but roughly the same end state.[^order] If you compare learners by area, your data ordering is part of what is being scored.

[^order]: A model is *exchangeable* if permuting the data leaves the joint probability unchanged — true for i.i.d.-given-$$\theta$$ Bayesian models, false for curricula and for SGD. For an exchangeable model, every ordering gives the same total area (the intermediate curves differ; the area doesn't), so the marginal likelihood is well-defined without reference to an order. Without exchangeability, you can average over orderings to get an order-free quantity. The [original post](https://iclr-blogposts.github.io/2024/blog/clml/) treats this carefully; here I just flag it.

## Three Functionals, One Curve

With the curve in hand, the model-selection zoo reduces to geometry:

| Functional of the curve | Bayesian name | What you'd call it |
|---|---|---|
| **Final height:** $$\ell_N$$, the loss after conditioning on (nearly) all $$N$$ points | negative log posterior predictive; averaged over orderings, the leave-one-out cross-validation score | validation loss, held-out perplexity |
| **Total area:** $$\sum_{n=1}^{N} \ell_n$$ | negative log marginal likelihood; negative log evidence | summed one-pass training loss; training-speed estimator |
| **Tail area:** $$\sum_{n=N-k+1}^{N} \ell_n$$ | negative **conditional** log marginal likelihood (CLML) | loss summed over the last $$k$$ steps of the pass; what sliding-window perplexity evaluation does within each window |

The **conditional log marginal likelihood** is the newest entry, proposed by [Lotfi et al. (2022)](https://arxiv.org/abs/2202.11678) — an ICML 2022 Outstanding Paper — as a repair for the marginal likelihood's pathologies: condition on the first $$N-k$$ points (let the posterior warm up), then score only the joint likelihood of the remaining $$k$$:

$$
\log p(x_{N-k+1:N} \mid x_{1:N-k}, \mathcal{M}) = \sum_{n=N-k+1}^{N} \log p(x_n \mid x_{<n}, \mathcal{M}).
$$

In curve terms: cut off the head of the curve, where the prior dominates, and keep the area under the tail. The split is a dial. At $$k = N$$ you recover the full marginal likelihood. At $$k = 1$$ you recover the final height — the last one-step-ahead loss; and since, for an exchangeable model, predicting one held-out point from the other $$N-1$$ is the same problem no matter which point is held out, this equals leave-one-out cross-validation once you average over the choice. The CLML therefore *interpolates between the marginal likelihood and cross-validation*. Averaged over data orderings, it coincides with the cumulative leave-$$p$$-out cross-validation score that [Fong & Holmes (2020)](https://arxiv.org/abs/1905.08737) had already studied for exchangeable models — the two literatures arrived at the same functional from opposite ends.[^expectation]

Where to set the dial has only heuristic guidance — condition until the posterior has stabilized; Lotfi et al. use an 80/20 split, Fong & Holmes discuss holding back 10–50% — and the selected model can depend on the setting, as a phase diagram will show at the end.

One more piece of bookkeeping: to compare the functionals in the same units, divide each by the number of points it sums over. You get, respectively, the *final* loss, the *average loss over the whole run*, and the *average loss over the tail of the run*. Phrased that way, it is almost obvious that they can rank two models differently — and almost obvious when they can't.

[^expectation]: Throughout, there's a quiet distinction between the *realized* quantity on your particular dataset and ordering (the sum of the $$\ell_n$$'s you actually observed) and its *expectation* over data draws and orderings. The literature's terms — marginal likelihood, cross-validation score — mix these freely, and per-token averages over large corpora concentrate anyway. The original post keeps the two rigorously apart, at some cost in terminology; here I won't.

## Where They Must Agree — Mostly

As $$n \to \infty$$, the posterior concentrates on the parameters whose predictions are closest (in KL divergence) to the data distribution, and the one-step-ahead predictive converges to the best predictor in the model class.[^bvm] The loss curve flattens onto an asymptote $$L_\infty(\mathcal{M})$$: the irreducible entropy of the data plus the model's **misspecification gap** — how far even the best member of the class is from the data distribution.

In per-token units, all three functionals converge to $$L_\infty(\mathcal{M})$$. So *if two models' asymptotes differ*, then given enough data all three criteria settle on the same ranking, and the differences between them live in the transient: the excess area between the curve and its asymptote grows only logarithmically — roughly $$\tfrac{d}{2}\log n$$ for a well-specified regular model with $$d$$ parameters, and $$\lambda \log n$$ with a learning coefficient $$\lambda \le d/2$$ (typically much smaller) for singular models like neural networks ([Watanabe](https://www.cambridge.org/core/books/algebraic-geometry-and-statistical-learning-theory/9C8FD1BDC817E2FC79117C7F41544A3A)).[^singular] The entire "Occam factor" — the complexity penalty the marginal likelihood is celebrated for — lives in that logarithmic sliver above the asymptote.

But the tied case is not a corner case — it is the marginal likelihood's home turf. Suppose two asymptotes coincide: nested model classes that both contain the truth, a symmetry or constraint that holds exactly, a redundant block of parameters. Then held-out loss goes silent — the final-height difference shrinks to zero and drowns in noise, which is the well-known inconsistency of cross-validation for selecting between nested models — while the logarithmic Occam term keeps growing and settles the comparison *permanently* in favor of the simpler model. That is Bayes-factor consistency, and it is why the evidence can recover the true dimensionality in Bayesian PCA and learn exact symmetries and constraints where validation loss has nothing to say. If your question is about ties, the area is not one option among three; it is the only functional with any signal left.

So: asymptotes differ → all criteria agree eventually, and disagreements are pre-asymptotic; asymptotes tie → only the area still discriminates. Either way, "eventually" is doing heroic work. For expressive models the transient is the only regime we ever see: we pretrain for roughly one epoch because fresh tokens beat repeated ones per unit of compute, and the curve is still falling when the data runs out. Deep learning lives permanently in the pre-asymptotic regime — which is exactly the regime where the criteria can disagree.

[^bvm]: For well-specified regular models this is Bernstein–von Mises territory; under misspecification, concentration at the KL-projection is a separate (Berk / Kleijn–van der Vaart-type) result; and for *singular* models — non-identifiable parameters, degenerate Fisher information, i.e., neural networks — Bernstein–von Mises fails outright, which is precisely why $$d/2$$ gets replaced by Watanabe's learning coefficient in the next sentence. The cash value, if you want to skip the theory: the curve flattens onto a floor, and neural networks pay their complexity penalty more slowly than parameter counting suggests.

[^singular]: Strictly, the clean $$\lambda \log n$$ expansion is for the realizable (truth-in-class) case; under misspecification the second-order term is more delicate. Treat this paragraph as a blog-level gloss.

## Where They Disagree: Crossing Curves

When can two curves rank differently under different functionals? Decompose model quality into two independent axes:

**Misspecification sets the plateau.** No parameter setting closes the gap; the curve's asymptote is bounded away from the entropy of the data. Every real model is misspecified somewhere — a bounded context window, a fixed-size recurrent state that cannot do arbitrary retrieval, a tokenizer that makes some structure expensive to represent. Model selection among misspecified models is asking: whose *floor* is lowest?

**Prior-data misfit sets the descent.** Two models with the same floor can approach it at very different speeds, depending on how much prior probability mass sits near the good parameter configurations. This axis covers a spectrum. At the benign end, mere *vagueness*: a diffuse prior wastes mass on parameters the data will rule out, and pays for it over the early curve — the Occam penalty operating exactly as designed. At the pathological end, genuine **prior-data conflict** in the technical sense (Evans & Moshonov, 2006): the prior concentrates in the *wrong* place, and the data arrives as a surprise. For a deep-learning pipeline, read "prior" as the initialization and the architecture's inductive biases — the optimizer and schedule belong to the inference mechanism, as discussed above.

<figure>
  <img src="/assets/img/2024-05-07-clml/prior_conflict_and_model_misspecification_0.67.png" alt="Two panels of loss curves. Left: three curves descend on a similar timescale but plateau at different heights (misspecification). Right: three curves plateau at the same height but descend at different speeds (prior-data misfit)." style="max-width: 100%;">
  <figcaption><strong>The two axes, separated.</strong> Left: three models with different plateaus — different degrees of misspecification. Right: three models with the same plateau but different descent speeds — different degrees of prior-data misfit. Real models differ along both axes at once.</figcaption>
</figure>

If the two axes are *correlated* — the fastest starter also has the lowest floor — every functional picks the same winner and there is nothing to argue about. The interesting case is **anti-correlation**: the model that adapts fastest from small data has a worse floor, and the eventual winner starts slow. Then the curves cross, and the functionals come apart:

- **Final height** flips at the crossing point: past it, the strong-floor model is simply better *now*.
- **Total area** (the marginal likelihood) flips much later, if at all. The area deficit accumulated before the crossing has to be repaid before the LML changes its mind — it keeps books on early mistakes with no forgiveness. This hysteresis is also why the LML's correlation with generalization is *non-monotonic* in dataset size, a pattern Lotfi et al. document: at tiny $$N$$ the prior is all there is, and measuring it is fine; at intermediate $$N$$ the stale early terms actively mislead; at large $$N$$ everything reconverges.
- **Tail area** (the CLML) flips somewhere in between, at a point controlled by the split — a hyperparameter of the *criterion*, not of the model.

<figure>
  <img src="/assets/img/2024-05-07-clml/anticorrelated_prior_conflict_and_model_misspecification_1.30.png" alt="Three loss curves that intersect each other as dataset size grows; the model best at small n is worst asymptotically and vice versa." style="max-width: 100%;">
  <figcaption><strong>Anti-correlated misspecification and prior-data misfit.</strong> The model that wins at small dataset sizes (fast descent, high floor) is the worst asymptotically; the eventual winner starts slowest. The ranking by final height changes multiple times as data grows, and rankings by area change at different — later — points than rankings by height.</figcaption>
</figure>

If this looks like a scaling-law plot, that is because — with one substitution — it is one. The height of the curve at step $$n$$ is the loss of a learner that has seen $$n$$ points: the curve's x-axis *is* dataset scale, and reading the one-pass curve left to right is (exactly for the ideal Bayesian learner, approximately for SGD) sweeping out $$L(D)$$. Data-scaling fits make the two axes quantitative: in the standard parametric form $$L(D) = E + A \cdot D^{-\alpha}$$, the fitted floor $$E$$ *is* the misspecification axis and the fitted descent terms $$(A, \alpha)$$ *are* the misfit axis. Plateau and descent are not metaphors here; they are parameters you already estimate, and "does a crossing lie before my target scale?" is a checkable property of two fits. Compute-scaling comparisons — where parameters and data grow together — put the same geometry on a compound axis, and ranking inversions there are a documented, recurring phenomenon ([Tay et al., 2022](https://arxiv.org/abs/2207.10551)). Every "architecture X beats transformers" claim staked at 350M parameters is implicitly a bet that the curves don't cross before the scale you care about. A live example: fixed-state-size architectures can match transformers on average perplexity while being fundamentally limited at copying and retrieval from long contexts ([Jelassi et al., 2024](https://arxiv.org/abs/2402.01032)) — matched descent, different floors, in exactly the dimension long-context deployments care about.

When are you actually in the crossing regime? A field heuristic: matched-recipe ablations within one architecture family usually produce near-parallel curves — vertical shifts, no crossings — and then all three functionals agree and none of this matters. Crossings concentrate where inductive bias is genuinely being traded against expressivity: across architecture families, and under regularization, data-mixture, or curriculum changes. Which suggests the right use of this whole framing when stakes are high: don't commit to any single functional evaluated at proxy scale — fit both curves, estimate whether a crossing lies between proxy and target scale, and report *that*.

## In-Context Learning: The Same Picture, Frozen Weights

Everything so far marginalized over parameters. But the chain-rule identity applies to any joint predictive distribution — including that of a trained LLM with frozen weights, where conditioning happens in activations rather than in weights or posteriors.

Score a sequence of task examples under a frozen model: the per-example loss as a function of position is an **in-context learning curve**, and the log-probability of the whole demonstration sequence is — same algebra as before — the area under it. These curves have been visible from the start: [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) already plotted per-token loss against context position and found LSTM curves plateauing after a few hundred tokens while transformer curves kept falling — the plateau-versus-descent geometry, measured within the context window instead of across training. Under the reading of in-context learning as implicit Bayesian inference ([Ortega et al., 2019](https://arxiv.org/abs/1905.03030); [Xie et al., 2021](https://arxiv.org/abs/2111.02080); and by construction in prior-data fitted networks, [Müller et al., 2022](https://arxiv.org/abs/2112.10510)), the area is a task-level marginal likelihood: the pretrained model plays the role of the prior over tasks, and each in-context example plays the role of a data point.

The menu of functionals reappears, and now the choice is a live eval-design question rather than Bayesian arcana:

- **Few-shot evaluation** — scoring the query given a fixed set of demonstrations — is a *final height* criterion.
- **Scoring the whole demonstration sequence** — as whole-prompt log-likelihood ranking and compression-style analyses do — is an *area* criterion, and it charges the model for its zero-shot and one-shot predictions even if you will always deploy with a full prompt.
- **Discarding the early positions before averaging** — standard practice in sliding-window perplexity evaluation, where only the final tokens of each window are scored given the overlap — is exactly the CLML move, performed daily by people who have never heard of it.

And the failure mode transfers: a model with strong zero-shot priors but sluggish in-context adaptation, versus a model that starts poorly and adapts steeply, gives you crossing in-context curves — the same anti-correlation geometry, at inference time. Which one is "better at in-context learning" depends on the functional, i.e., on the deployment: fixed-shot prompting cares about the height at the deployment shot count; a long-context assistant that keeps learning over a session cares about the tail area.

## Which Functional Answers Which Question

There is no winner. There are different questions:

**"Which fixed hypothesis explains this data?" → total area (LML).** For comparing *priors as priors* — hypothesis testing, Bayes factors, MacKay-style arguments — charging the prior is not a bug; it is the entire point, and the marginal likelihood has a coherence argument the other functionals lack: it is the probability the model assigns to the data, the one score that composes correctly under further Bayesian updating. Add the tied-asymptote argument from above — constraint and symmetry learning, dimensionality recovery — and this also covers the evidence's genuine practical successes, like type-II maximum likelihood for GP kernel hyperparameters (differentiable, no validation split needed). For the classic exposition, read [MacKay's chapter 28](https://www.inference.org.uk/itprnn/book.pdf#page=355).

**"Which trained model generalizes best, frozen, at the current data size?" → final height (validation loss / cross-validation).** This directly measures the deployment quantity. It is also the cheapest of the three to estimate honestly, which is not a coincidence: the field converged on held-out evaluation because it answers the question the field usually asks.

**"Which learner will predict best while continuing to learn?" → area, usually tail area.** Online learning, continual learning, sequential decision-making, compression, and in-context learning over long sessions all integrate performance across the curve, so an area is the honest objective — with the head of the curve cut off if deployment always starts from a warm state.

**"I can't afford held-out evaluation" → use the curve you already have, carefully.** In the single-epoch regime, batch losses are computed on not-yet-seen data, so a running average of recent training losses is a free held-out-loss estimate — *on the training mixture*. It is not comparable across runs with different data mixtures or orderings, and it does not replace the fixed per-domain validation sets used for distribution control. Training-speed estimators operationalize the area version — inheriting, by this post's own argument, the LML's books on early mistakes, so selecting by training speed is itself a bet that the curves don't cross. Scaling-law fits go one step further and model the curve rather than evaluating a single functional of it.

One caveat deserves its own paragraph, because in practice it dominates everything above: **for deep networks, none of the Bayesian quantities are computable exactly, and the estimation error can exceed all the conceptual differences.** Estimating the LML from prior samples is hopeless — an astronomically small fraction of prior samples lands anywhere useful — and Laplace or variational posteriors capture one mode of a many-moded landscape. In Lotfi et al.'s DNN experiments, the Laplace-estimated CLML ended up nearly indistinguishable from a plain validation loss (for the CNN models, per a difference overlay in the original post's appendix); indeed, the first arXiv version accidentally computed the Bayesian-model-averaged validation loss in place of the CLML, and fixing the bug left the qualitative take-aways unchanged — which is itself evidence of how close the quantities are in that regime.[^bug] When your estimator of the sophisticated criterion degenerates into the cheap criterion you already had, use the cheap criterion and keep the honesty.

[^bug]: I found the bug; the exchange with the authors, their response — which reads the corrected results more favorably for the CLML than I do — and a detailed code review are in the [appendix of the original post](https://iclr-blogposts.github.io/2024/blog/clml/#appendix). To be fair to the CLML: most of that paper's experiments (density models, Fourier features, GPs, deep kernel learning) use exact or well-behaved likelihoods and stand independently of the DNN Laplace experiments.

## Occam's Razor Won't Settle It

The marginal likelihood's traditional selling point is that it "automatically implements Occam's razor": complex models spread their prior mass over more possible datasets, so they assign less evidence to any particular one. This is true — as a statement about fixed priors, i.e., about areas. But Occam's razor, taken as "prefer the shortest description," doesn't single out the area. The same razor:

- with a uniform prior over parameters, hands you maximum likelihood;
- with a joint code for parameters and data, hands you MAP estimation;
- applied to the tail of the curve, hands you the CLML;
- applied to the final predictive, hands you cross-validation.

All of these are compression-flavored criteria, and the razor endorses the whole menu.[^longcow] Meanwhile, standard model selection quietly assumes a *uniform prior over the models themselves*, ignoring that architectures and priors have description lengths too. The razor is a menu, not a verdict; you still have to order by asking what you'll do with the model.

[^longcow]: The razor's dependence on framing is old news in picture form: MacKay's classic how-many-boxes-behind-the-tree example ([p. 343](https://www.inference.org.uk/itprnn/book.pdf#page=355)) — and its internet-era rebuttal, the [long cow](https://www.reddit.com/r/confusing_perspective/comments/atvu6s/long_cow/) (h/t Freddie Bickford Smith): two ordinary cows, or one very long cow behind a tree? Simplicity is prior-dependent.

## A Toy Where Every Criterion Fails

None of this requires deep networks or approximation error to bite. Take Bayesian linear regression — exact posteriors, exact evidence, no estimation excuses — with 64 features and three hyperparameter settings that mix the two axes: a tight prior with the lowest assumed noise ($$\sigma_w = 0.1$$, $$\sigma_\text{noise} = 0.8$$), a diffuse prior with middling noise ($$100$$, $$1.0$$), and a moderate prior with the highest noise ($$1$$, $$1.2$$). The synthetic targets are noiseless, so every setting is misspecified — the assumed noise level is never right — and the prior scales span three orders of magnitude, so the degree of prior-data misfit varies wildly too.

<figure>
  <img src="/assets/img/2024-05-07-clml/binary_regression_information_metrics.png" alt="Six panels of information metrics versus dataset size for three Bayesian linear regression models." style="max-width: 100%;">
  <figcaption><strong>Exact criteria, exactly disagreeing.</strong> Six panels: held-out loss and training loss, total area (negative log marginal likelihood), tail area (negative CLML, conditioning on half the data), the per-token area rate, and a training-speed proxy for the area — for the three models as a function of dataset size (five trials; reported in bits). The area criterion never picks the eventual held-out winner anywhere in this range, and the condition-on-half tail criterion only finds it after roughly 80% of the data.</figcaption>
</figure>

The punchline is sharper in the model-selection view: sweep both the dataset size and the CLML's split point, and record which model wins.

<figure>
  <img src="/assets/img/2024-05-07-clml/binary_regression_conditional_joint_marginal_information_decision_boundary.svg" alt="A phase diagram of which of three models is selected, as a function of dataset size and held-back size." style="max-width: 60%;">
  <figcaption><strong>The selected model depends on the criterion's own settings.</strong> Which of the three models has the best tail-area score (CLML), as a function of the dataset size (x-axis) and how much of it is held back for scoring (y-axis) — the rest is conditioned on. The white line marks the hold-back-half split. All three models are "the best" somewhere on this plane.</figcaption>
</figure>

Every criterion here is computed *exactly*, in a linear model, and the selected model still depends on the dataset size and on the criterion's own settings. Whatever difficulties large models add, they are on top of this — the ambiguity is intrinsic to the functionals, not an artifact of approximate inference. (Code: [toy experiment](https://colab.research.google.com/drive/1rUnOvkFIxVrIJACxyjcQiGHo3nA77T4T?usp=sharing), [visualizations](https://colab.research.google.com/drive/1q0esvQGSqd7d6zJfjbFcz-DGSKYi_WpC?usp=sharing).)

## Takeaways

- The negative log marginal likelihood is the area under the loss curve of an ideal Bayesian online learner; validation loss is the curve's final height; the CLML is the area under its tail, interpolating between the two. One curve, three functionals.
- In per-token units all three converge to the model's asymptote, so when asymptotes differ they can only disagree pre-asymptotically — which, for expressive models trained for roughly one epoch, is always.
- Pre-asymptotic disagreement requires crossing curves, and curves cross when prior-data misfit (descent speed) is anti-correlated with misspecification (plateau height). In scaling-law terms: the fitted floor and the fitted descent terms can rank two models oppositely, and scaling-law crossovers are this phenomenon at industrial scale.
- When asymptotes *tie* — nested models, exact constraints and symmetries — held-out loss goes silent and the evidence's logarithmic Occam term is the only signal left: Bayes-factor consistency is the marginal likelihood's irreplaceable use case.
- Match the functional to the question: comparing priors or testing hypotheses → area; deploying a frozen model → height; sequential prediction, compression, and long-context in-context learning → tail area.
- In-context learning curves are the frozen-weights version of the same geometry, so eval-design choices — score at the deployment shot count? average over the whole prompt? discard early positions? — are choices among these functionals, whether made deliberately or not.
- Two different things make criteria disagree in practice: curve geometry (crossings) and estimator error (approximate posteriors). For deep networks the second can dwarf the first — and when the fancy estimator collapses into the validation loss, use the validation loss.

## Appendix: Terminology Map

For readers coming from the [original post](https://iclr-blogposts.github.io/2024/blog/clml/) or the Bayesian literature (the entropy notation $$\mathrm{H}[\cdot]$$ below is used loosely for both realized and expected quantities, per the earlier footnote):

| This post | Literature / original post |
|---|---|
| one-step-ahead loss $$\ell_n$$ | negative log posterior predictive; conditional marginal information $$\mathrm{H}[x_n \mid x_{<n}, \mathcal{M}]$$ |
| final height (expected) | conditional marginal cross-entropy; expected leave-one-out CV loss; held-out NLL |
| total area | negative log marginal likelihood; joint marginal information; prequential code length |
| tail area | negative conditional log marginal likelihood (CLML, [Lotfi et al., 2022](https://arxiv.org/abs/2202.11678)); conditional joint marginal information |
| tail area, averaged over orderings | cumulative leave-$$p$$-out CV score ([Fong & Holmes, 2020](https://arxiv.org/abs/1905.08737)); conditional joint marginal cross-entropy |

*Acknowledgements: This post is a rewrite of my ICLR 2024 blog post, which grew out of an exchange with the authors of Lotfi et al. (2022) — my thanks to them for their engagement, and to Freddie Bickford Smith for comments on the original. LLM assistance was used in drafting and editing both versions.*
