---
layout: post
title: "The Marginal Likelihood is the Area Under Your Loss Curve"
description: >-
  Bayesian model selection, cross-validation, and the conditional (log) marginal likelihood are three functionals of the same object: an idealized one-pass loss curve. Seen this way, the classic debates about the marginal likelihood map directly onto things LLM researchers stare at every day — pretraining curves, scaling-law crossovers, and in-context learning curves — and the classic failure modes (misspecification, prior-data misfit) become statements about where curves plateau and how fast they fall.
date: 2026-07-20
author: Andreas Kirsch
---

<!--
NOTES FOR MOVING THIS DRAFT TO THE PERSONAL BLOG:
- Figures are referenced from /assets/img/2024-05-07-clml/ (this repo). Copy that folder
  (or at least the five SVGs referenced below) and adjust paths.
- Math uses kramdown-style $$...$$ delimiters (inline and display), matching the ICLR blog setup.
- Footnotes use kramdown [^n] syntax.
- This is a rewrite of the ICLR 2024 blog post:
  https://iclr-blogposts.github.io/2024/blog/clml/
-->

**TL;DR.** For any probabilistic model, the chain rule turns the log marginal likelihood — the Bayesian evidence — into a sum of one-step-ahead prediction losses. The evidence is, exactly, the negative *area under the loss curve* of an ideal Bayesian learner making a single pass over your dataset. Held-out loss (cross-validation, perplexity evals) is the *final height* of that curve. The conditional log marginal likelihood (CLML) is the *area under its tail*. All the model-selection criteria that the Bayesian literature debates are functionals of one curve, and they can only disagree when loss curves *cross*. Curves cross when a model's inductive-bias alignment (how fast its curve falls) trades off against its expressivity limits (where its curve plateaus). None of the functionals is universally right: whole-curve area answers "which prior explains this data?"; final height answers "which trained model generalizes best right now?"; tail area answers "which learner predicts best after a warm-up?". The same geometry reappears, frozen-weights edition, in in-context learning curves — which is why this old Bayesian debate is worth knowing about even if you never intend to compute a posterior.

## Two Runs, One Decision

You have two pretraining runs on the same data stream, one pass each. Run A gets off to a great start — its loss drops fast — but flattens out early. Run B spends the first chunk of training looking embarrassing, then keeps grinding and ends lower.

Which model is better?

If you answered "B, obviously — I ship the final checkpoint, not the training history," you have taken a side in a long-running argument in Bayesian statistics, and you have taken it against the marginal likelihood. Because — and this is an identity, not an analogy — classical Bayesian model selection scores models by the **area under that loss curve**, early mistakes included.

Once you see the criteria as curve functionals, the whole menu of Bayesian model-selection quantities — marginal likelihood, cross-validation, and the more recently proposed conditional marginal likelihood — snaps into a single picture, and the classic debates about them become concrete questions about curve geometry: When can the rankings disagree? (Only when curves cross.) What makes curves cross? (Misspecification trading off against prior-data misfit.) Which functional should you use? (Depends on which question you're asking — and the answer changes if you care about in-context learning.)

This post walks through that picture. It is a substantially rewritten version of an [ICLR 2024 blog post](https://iclr-blogposts.github.io/2024/blog/clml/) of mine that made these points in heavier information-theoretic clothing; this version is written for people who spend their days looking at loss curves rather than posteriors.

## The Identity

Take a probabilistic model $$\mathcal{M}$$: a prior $$p(\theta \mid \mathcal{M})$$ over parameters plus a likelihood $$p(x \mid \theta, \mathcal{M})$$. The **marginal likelihood** (or **evidence**) of a dataset $$x_{1:N} = (x_1, \ldots, x_N)$$ integrates the parameters out:

$$
p(x_{1:N} \mid \mathcal{M}) = \int p(x_{1:N} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta.
$$

This is the quantity behind Bayes factors, type-II maximum likelihood, and "automatic Occam's razor" arguments. It looks like a static score for how well the model explains the data. But apply the chain rule of probability:

$$
\log p(x_{1:N} \mid \mathcal{M}) = \sum_{n=1}^{N} \log p(x_n \mid x_{<n}, \mathcal{M}),
$$

and notice what each term is: $$p(x_n \mid x_{<n}, \mathcal{M}) = \int p(x_n \mid \theta, \mathcal{M}) \, p(\theta \mid x_{<n}, \mathcal{M}) \, d\theta$$ is the **posterior predictive** of a learner that has *already conditioned on the first $$n-1$$ points*. Write the per-step loss as $$\ell_n = -\log p(x_n \mid x_{<n}, \mathcal{M})$$, and the negative log evidence becomes

$$
-\log p(x_{1:N} \mid \mathcal{M}) = \sum_{n=1}^{N} \ell_n,
$$

the summed one-step-ahead losses of an ideal Bayesian learner doing online learning over your dataset in a single pass — updating by exact conditioning rather than by gradient steps.[^order] Plot $$\ell_n$$ against $$n$$ and you have a loss curve; the negative log evidence is the area under it.

<figure>
  <img src="/assets/img/2024-05-07-clml/area_under_curve_1.00.svg" alt="Left: a smooth decreasing loss curve with the area under it shaded. Right: the same with noisy per-step losses." style="max-width: 100%;">
  <figcaption><strong>The marginal likelihood is an area.</strong> Left: the expected one-step-ahead loss of an ideal Bayesian learner as a function of the number of observed points; the shaded area under the curve is the negative log marginal likelihood (in expectation). Right: the realized per-step losses \(\ell_n\) on one dataset are a noisy version of the same curve; their sum is the negative log evidence of that dataset.</figcaption>
</figure>

Two things follow immediately.

First, **the marginal likelihood is not a property of the trained model.** It is a property of the entire learning trajectory. The very first term charges the model for predicting $$x_1$$ from the bare prior. A model class that will eventually be excellent, but starts from a poorly aligned prior, pays for that start in every comparison — no matter how much data comes later, the early-curve area stays on the books.

Second, this is a **compression statement**. Feeding the one-step-ahead predictions into an arithmetic coder compresses the dataset to $$\sum_n \ell_n$$ nats; this is Dawid's [prequential](https://en.wikipedia.org/wiki/Prequential_analysis) view of model assessment and the minimum-description-length view of the evidence. It's the same accounting that makes a language model a lossless compressor ([Delétang et al., 2023](https://arxiv.org/abs/2309.10668)), and prequential codes built from actual training runs are among the best practical description lengths we know for deep learning ([Blier & Ollivier, 2018](https://arxiv.org/abs/1802.07044)).

And the SGD analogy is closer than it might look. In (near-)single-epoch pretraining, the loss you record on each batch is computed *before* you update on that batch — every batch is held-out data for the current parameters. The summed first-pass training loss is therefore the prequential code length of the corpus under "architecture + initialization + optimizer" as the learning algorithm: the practical stand-in for a negative log marginal likelihood, with SGD steps in place of Bayesian conditioning. This correspondence is exactly what training-speed-based model selection exploits ([Lyle et al., 2020](https://arxiv.org/abs/2010.14499); [Ru et al., 2021](https://arxiv.org/abs/2006.04492)): "sum of training losses" is a marginal-likelihood surrogate, and "area under the training curve" is not a metaphor but the actual estimator.

[^order]: The decomposition holds for every ordering of the data, and for an exchangeable model every ordering gives the same total — the intermediate curves differ, the area doesn't. Corpora with curricula, and SGD itself, are not exchangeable; then "the curve" (and its area) depends on the data order, and you can average over orderings if you want an order-free quantity. The original post discusses this at length; here I'll just flag it and move on.

## Three Functionals, One Curve

With the curve in hand, the model-selection zoo reduces to geometry:

| Functional of the curve | Bayesian name | What you'd call it |
|---|---|---|
| **Final height:** $$\ell$$ after conditioning on (nearly) all $$N$$ points | negative log posterior predictive; expected version is leave-one-out cross-validation | validation loss, held-out perplexity |
| **Total area:** $$\sum_{n=1}^{N} \ell_n$$ | negative log marginal likelihood (LML), evidence | summed one-pass training loss; training-speed estimator |
| **Tail area:** $$\sum_{n=N-k+1}^{N} \ell_n$$ | negative **conditional** log marginal likelihood (CLML) | loss summed over the last $$k$$ steps of the pass |

The **conditional log marginal likelihood** is the newest entry, proposed by [Lotfi et al. (2022)](https://arxiv.org/abs/2202.11678) — an ICML 2022 Outstanding Paper — as a repair for the marginal likelihood's pathologies: condition on the first $$N-k$$ points (let the posterior warm up), then score only the joint likelihood of the remaining $$k$$:

$$
\log p(x_{N-k+1:N} \mid x_{1:N-k}, \mathcal{M}) = \sum_{n=N-k+1}^{N} \log p(x_n \mid x_{<n}, \mathcal{M}).
$$

In curve terms: cut off the head of the curve, where the prior dominates, and keep the area under the tail. The split fraction is a dial. At $$k = N$$ you recover the full marginal likelihood; at $$k = 1$$ you recover leave-one-out cross-validation, i.e., the final height. The CLML *interpolates between the marginal likelihood and cross-validation*. Averaged over data orderings, it coincides with the cumulative leave-$$p$$-out cross-validation score that [Fong & Holmes (2020)](https://arxiv.org/abs/1905.08737) had already studied — the two literatures arrived at the same functional from opposite ends.[^expectation]

It's worth being clear that per-step and per-token comparisons need matching units: divide each functional by the number of points it sums over and you get, respectively, the *final* loss, the *average loss over the whole run*, and the *average loss over the tail of the run*. Phrased that way, it is almost obvious that these can rank two models differently — and almost obvious when they can't.

[^expectation]: Throughout, there's a quiet distinction between the *realized* quantity on your particular dataset (the sum of $$\ell_n$$'s you actually observed) and its *expectation* over draws from the data distribution. The literature's terms — marginal likelihood, cross-validation score — mix these up freely, and for large $$N$$ the per-token averages concentrate anyway. The original post keeps the two rigorously apart, at some cost in terminology ("joint marginal cross-entropy" vs. "joint marginal information"); here I won't.

## Where They Must Agree

As $$n \to \infty$$, under the usual regularity conditions, the posterior concentrates (Bernstein–von Mises) and the one-step-ahead predictive converges to the best predictor in the model class. The loss curve flattens onto an asymptote $$L_\infty(\mathcal{M})$$: the irreducible entropy of the data plus the model's **misspecification gap** — how far even the best member of the class is from the data distribution.

All three functionals, in per-token units, converge to $$L_\infty(\mathcal{M})$$. So *given enough data, all three criteria produce the same ranking* — and none of them tells you anything beyond the asymptote. The differences between them live entirely in the transient: the excess area between the curve and its asymptote, which grows only logarithmically — roughly $$\tfrac{d}{2}\log n$$ for a $$d$$-parameter regular model, and $$\lambda \log n$$ with a smaller-than-$$d/2$$ learning coefficient $$\lambda$$ for singular models like neural networks ([Watanabe](https://www.cambridge.org/core/books/algebraic-geometry-and-statistical-learning-theory/9C8FD1BDC817E2FC79117C7F41544A3A)). The entire "Occam factor" — the complexity penalty people credit the marginal likelihood with — lives in that logarithmic sliver above the asymptote.

This sounds like a reason to stop caring: asymptotically, everything agrees. But "enough data" is doing heroic work. For expressive models the transient is the only regime we ever see — we pretrain for a single epoch precisely because the curve is still falling when the data runs out. Deep learning lives permanently in the pre-asymptotic regime, which is exactly the regime where the criteria can disagree.

## Where They Disagree: Crossing Curves

When can two curves give different rankings under different functionals? Decompose model quality into two independent axes:

**Misspecification sets the plateau.** No parameter setting closes the gap; the curve's asymptote is bounded away from the entropy of the data. Every real model is misspecified somewhere — a bounded context window, a fixed-size recurrent state that can't do arbitrary retrieval, a tokenizer that makes some structure expensive, an architecture that can't represent the interactions in the data. Model selection among misspecified models is asking: whose *floor* is lowest?

**Prior-data misfit sets the descent.** Two models with the same floor can approach it at very different speeds, depending on how much probability mass their priors put near the good parameter configurations. The Bayesian literature calls this **prior-data conflict**; for a deep learning pipeline, read "prior" as everything that shapes the trajectory before the data does — initialization scheme, architecture-encoded inductive biases, optimizer and schedule. A well-aligned prior is a head start; a misaligned one is a tax paid over the early curve.

<figure>
  <img src="/assets/img/2024-05-07-clml/prior_conflict_and_model_misspecification_0.67.svg" alt="Two panels of loss curves. Left: three curves fall at the same speed but plateau at different heights (misspecification). Right: three curves plateau at the same height but fall at different speeds (prior-data misfit)." style="max-width: 100%;">
  <figcaption><strong>The two axes, separated.</strong> Left: three models with different plateaus — different degrees of misspecification. Right: three models with the same plateau but different descent speeds — different degrees of prior-data misfit. Real models differ along both axes at once.</figcaption>
</figure>

If the two axes are *correlated* — the fastest starter also has the lowest floor — every functional picks the same winner and there is nothing to argue about. The interesting case is **anti-correlation**: the model that adapts fastest from small data has a worse floor, and the eventual winner starts slow. Then the curves cross, and the functionals come apart:

- **Final height** flips at the crossing point: past it, the strong-floor model is simply better *now*.
- **Total area** (the marginal likelihood) flips much later, if at all. The area deficit accumulated before the crossing has to be repaid before the LML changes its mind — the LML keeps books on early mistakes with no forgiveness. This hysteresis is also why the LML's correlation with generalization is *non-monotonic* in dataset size, a pattern Lotfi et al. document: at tiny $$N$$ the prior is all there is (and the LML is basically measuring it, fine); at intermediate $$N$$ the stale early terms actively mislead; at huge $$N$$ everything converges again.
- **Tail area** (the CLML) flips somewhere in between, at a point controlled by the split fraction — which is a hyperparameter of the *criterion*, not of the model, and there is no principled way to set it that doesn't smuggle in the answer.

<figure>
  <img src="/assets/img/2024-05-07-clml/anticorrelated_prior_conflict_and_model_misspecification_1.30.svg" alt="Three loss curves that intersect each other as dataset size grows; the model best at small n is worst asymptotically and vice versa." style="max-width: 100%;">
  <figcaption><strong>Anti-correlated misspecification and prior-data misfit.</strong> The model that wins at small \(n\) (fast descent, high floor) is the worst asymptotically; the eventual winner starts slowest. The ranking by final height changes multiple times with \(n\), and rankings by area change at different — later — points than rankings by height.</figcaption>
</figure>

If this picture looks familiar, it should: it is the scaling-law crossover plot. Architecture rankings that hold at small scale and invert at large scale are a documented, recurring phenomenon ([Tay et al., 2022](https://arxiv.org/abs/2207.10551)); [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) already showed LSTM curves plateauing where transformers kept improving. Every "architecture X beats transformers" claim staked at 350M parameters is implicitly a bet that the curves don't cross before the scale you care about. When you select architectures by small-scale proxy runs — as neural architecture search and scaling-law extrapolation both do — you are betting on a functional of the early curve to predict the late curve. That bet is sound exactly when no crossing lies between proxy scale and target scale; misspecification-vs-misfit anti-correlation is the thing that breaks it.

## In-Context Learning: The Same Picture, Frozen Weights

Everything so far marginalized over parameters. But the chain-rule identity applies to any joint predictive distribution — including that of a trained LLM with frozen weights, where conditioning happens in activations rather than in weights or posteriors.

Score a sequence of task examples under a frozen model: the per-example loss as a function of position is an **in-context learning curve**, and the log-probability of the whole demonstration sequence is — same algebra as before — the area under it. Under the reading of in-context learning as implicit Bayesian inference ([Ortega et al., 2019](https://arxiv.org/abs/1905.03030); [Xie et al., 2021](https://arxiv.org/abs/2111.02080); and literally by construction in prior-fitted networks, [Müller et al., 2022](https://arxiv.org/abs/2112.10510)), that area is a task-level marginal likelihood: the pretrained model plays the role of the prior over tasks, and each in-context example plays the role of a data point.

The whole menu of functionals reappears, and now the choice between them is a live eval-design question rather than Bayesian arcana:

- **Few-shot evaluation at $$k$$ shots** — score the query given $$k$$ demonstrations — is a *final height* criterion.
- **Scoring the full demonstration sequence** — as whole-prompt log-likelihood ranking does — is an *area* criterion, and it charges the model for its zero-shot and one-shot predictions even if you will always deploy with $$k$$ shots in context.
- **Skipping the first few positions** before averaging — a common ICL-eval hygiene move — is exactly the CLML.

And the failure mode transfers: a model with strong zero-shot priors but sluggish in-context adaptation, versus a model that starts poorly and adapts steeply, gives you crossing in-context curves — the same anti-correlation geometry, at inference time. Which one is "better at in-context learning" depends on the functional, i.e., on the deployment: fixed-$$k$$ prompting cares about the height at $$k$$; a long-context assistant that keeps learning over the session cares about the tail area.

## Which Functional Answers Which Question

There is no winner. There are different questions:

**"Which fixed hypothesis explains this data?" → total area (LML).** For comparing *priors as priors* — hypothesis testing, Bayes factors, MacKay-style arguments — charging the prior is not a bug; it is the entire point. This also covers the marginal likelihood's genuine practical successes: type-II maximum likelihood for GP kernel hyperparameters (differentiable, no validation split needed) and learning constraints or symmetries, where the LML consistently favors the most constrained model that still fits. If you want the classic exposition, read [MacKay's chapter 28](https://www.inference.org.uk/itprnn/book.pdf#page=355).

**"Which trained model generalizes best, frozen, at the current data size?" → final height (validation loss / cross-validation).** This directly measures the deployment quantity. It is also the cheapest of the three to estimate honestly, which is not a coincidence: the field converged on held-out evaluation because it answers the question the field usually asks.

**"Which learner will predict best while continuing to learn?" → area, usually tail area.** Online learning, continual learning, sequential decision-making, compression, and in-context learning over long sessions all integrate performance across the curve, so an area is the honest objective — with the head of the curve cut off if deployment always starts from a warm state.

**"I can't afford held-out evaluation" → use the curve you already have.** In the single-epoch regime, batch losses are computed on not-yet-seen data, so a running average of recent training losses *is* a held-out loss estimate, free of charge. Training-speed estimators operationalize this for model selection; scaling-law fits go one step further and extrapolate the curve rather than just evaluating a functional of it.

One caveat deserves its own paragraph, because in practice it dominates everything above: **for deep networks, none of the Bayesian quantities are computable, and the estimation error can exceed all the conceptual differences.** Estimating the LML from a prior sample is hopeless (astronomically few samples land anywhere useful), and Laplace or variational posteriors capture one mode of a many-moded landscape. In Lotfi et al.'s own DNN experiments, the Laplace-estimated CLML tracked the (Bayesian-model-averaged) validation loss almost exactly — the first arXiv version even computed the validation loss by accident, and fixing the bug barely changed the plots, which is itself the clearest evidence of how close the two quantities are in that regime.[^bug] When your estimator of the sophisticated criterion degenerates into the cheap criterion you already had, use the cheap criterion and keep the honesty.

[^bug]: I found the bug; the exchange with the authors, their response, and a detailed code review are in the [appendix of the original post](https://iclr-blogposts.github.io/2024/blog/clml/#appendix). To be fair to the CLML: most of that paper's experiments (density models, Fourier features, GPs, deep kernel learning) use exact or well-behaved likelihoods and stand independently of the DNN Laplace experiments.

## Occam's Razor Won't Settle It

The marginal likelihood's traditional selling point is that it "automatically implements Occam's razor": complex models spread their prior mass over more datasets, so they assign less evidence to any particular one. This is true — as a statement about *fixed priors*, i.e., about areas.

But Occam's razor, taken as "prefer the explanation with the shortest description," doesn't single out the marginal likelihood. The same razor, with a uniform prior over parameters, hands you maximum likelihood; with a non-uniform prior and a joint code for parameters-plus-data, MAP; applied to the tail of the curve, the CLML; applied to the final predictive, cross-validation. All of these are compression-flavored criteria, and the razor is happy to justify any of them.[^longcow] Meanwhile, standard model selection quietly assums a *uniform prior over the models themselves*, ignoring that an architecture and its prior also have description lengths. Occam's razor is a menu, not a verdict; you still have to order by asking what you'll do with the model.

[^longcow]: The razor's dependence on framing is old news in picture form: MacKay's classic how-many-boxes-behind-the-tree example ([p. 343](https://www.inference.org.uk/itprnn/book.pdf#page=355)) — and its internet-era rebuttal, the [long cow](https://www.reddit.com/r/confusing_perspective/comments/atvu6s/long_cow/) (h/t Freddie Bickford Smith): two ordinary cows, or one very long cow behind a tree? Simplicity is prior-dependent.

## A Toy Where Every Criterion Fails

None of this requires deep networks or approximation error to bite. Take Bayesian linear regression — exact posteriors, exact evidence, no estimation excuses — with 64 features and three hyperparameter settings mixing the two axes: a tight prior with an optimistic noise level, a diffuse prior with a middling noise level, and a moderate prior with a pessimistic noise level. All three are misspecified to different degrees (the assumed noise never matches the truth) and misfit to different degrees (the prior scales differ wildly).

<figure>
  <img src="/assets/img/2024-05-07-clml/binary_regression_information_metrics.svg" alt="Six panels of information metrics versus dataset size for three Bayesian linear regression models." style="max-width: 100%;">
  <figcaption><strong>Exact criteria, exactly disagreeing.</strong> Held-out loss, total area (negative LML), tail area (negative CLML, conditioning on half the data), per-token area rate, and training-speed proxies for three Bayesian linear regression models, as a function of dataset size (five trials each; log base 2). The area criterion never picks the model with the best held-out loss in this data range; the tail-area criterion only starts agreeing with held-out loss after roughly 80% of the data.</figcaption>
</figure>

The punchline is sharper in the model-selection view: sweep both the dataset size and the CLML's split point, and record which model wins.

<figure>
  <img src="/assets/img/2024-05-07-clml/binary_regression_conditional_joint_marginal_information_decision_boundary.svg" alt="A phase diagram of which of three models is selected, as a function of dataset size and conditioning-set size." style="max-width: 60%;">
  <figcaption><strong>The selected model depends on the criterion's own hyperparameters.</strong> Which of the three models has the best tail-area score (CLML), as a function of dataset size (x-axis) and how much data is conditioned on (y-axis). The white line marks the condition-on-half split. All three models are "the best" somewhere on this plane.</figcaption>
</figure>

Every criterion here is computed *exactly*, in a linear model, and the selected model still depends on the dataset size and on the criterion's own settings. Whatever difficulties large models add, they are on top of this — the ambiguity is intrinsic to the functionals, not an artifact of approximate inference. (Code: [toy experiment](https://colab.research.google.com/drive/1rUnOvkFIxVrIJACxyjcQiGHo3nA77T4T?usp=sharing), [visualizations](https://colab.research.google.com/drive/1q0esvQGSqd7d6zJfjbFcz-DGSKYi_WpC?usp=sharing).)

## Takeaways

- The negative log marginal likelihood is the area under the loss curve of an ideal Bayesian online learner; validation loss is the curve's final height; the CLML is the area under its tail, interpolating between the two. One curve, three functionals.
- In per-token units all three converge to the same asymptote, so they can only disagree pre-asymptotically — which, for expressive models trained for one epoch, is always.
- Disagreement requires crossing curves, and curves cross when prior-data misfit (descent speed) is anti-correlated with misspecification (plateau height). Scaling-law crossovers are this phenomenon at industrial scale.
- Match the functional to the question: comparing priors or testing hypotheses → area; deploying a frozen model → height; sequential prediction, compression, long-context in-context learning → tail area.
- In-context learning curves are the frozen-weights version of the same geometry, so eval-design choices (score at $$k$$ shots? average over the prompt? skip early shots?) are choices among these functionals whether you think about it or not.
- For deep networks, estimation error in LML/CLML approximations can exceed every conceptual difference above. When the fancy estimator collapses into the validation loss, use the validation loss.

## Appendix: Terminology Map

For readers coming from the [original post](https://iclr-blogposts.github.io/2024/blog/clml/) or the Bayesian literature:

| This post | Literature / original post |
|---|---|
| one-step-ahead loss $$\ell_n$$ | negative log posterior predictive; conditional marginal information $$\mathrm{H}[x_n \mid x_{<n}, \mathcal{M}]$$ |
| expected final height | conditional marginal cross-entropy; expected leave-one-out CV loss; held-out NLL |
| total area | negative log marginal likelihood (LML); joint marginal information; prequential code length |
| tail area | negative conditional log marginal likelihood (CLML, [Lotfi et al., 2022](https://arxiv.org/abs/2202.11678)); cumulative leave-$$p$$-out CV ([Fong & Holmes, 2020](https://arxiv.org/abs/1905.08737)); conditional joint marginal information |
| expected versions of the above | the corresponding cross-entropies (expectations of "information" quantities over the data distribution) |

*Acknowledgements: This post is a rewrite of my ICLR 2024 blog post, which grew out of an exchange with the authors of Lotfi et al. (2022) — my thanks to them for their engagement, and to Freddie Bickford Smith for comments on the original. LLM assistance was used in drafting and editing both versions.*
