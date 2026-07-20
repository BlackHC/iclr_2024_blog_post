# Review notes for `marginal-likelihood-llm-era.md`

Working notes from the drafting process (not for publication). The draft was reviewed by a
simulated panel of five readers — different personas run on different model types — each of
whom read the draft cold, gave structured feedback, and answered five comprehension questions
so we could check whether the intended messages actually land.

## Panel and reception

| Persona | Model | Verdict | Comprehension check (5 Qs) |
|---|---|---|---|
| Senior pretraining researcher (frontier lab, Bayes-skeptical) | Fable | Would finish *and* share ("the quotable version of a speech I give twice a year"); wants one real LM experiment | 5/5 correct |
| Bayesian ML researcher (MacKay/GP tradition) | Opus | Technically sound core, fair to Lotfi et al.; found the draft *unfair to the marginal likelihood itself* (missing consistency virtue) | 5/5 correct |
| Busy research lead (5-minute skim) | Haiku | Save + share; TL;DR far too long; headings/table/takeaways carry the story | 5/5 correct **from a skim alone** |
| 3rd-year PhD student (LLM evals background, shaky Bayes) | Sonnet | "Taught rather than talked past"; listed 12 concrete stumbling blocks (jargon, skipped steps) | 5/5 correct |
| Technical fact-checker (no persona) | Fable | Verified all 13 arXiv/book links (via search; proxy blocked direct fetches), toy-experiment numbers, and faithfulness to the original post + Lotfi et al. exchange; found 1 blocker + several major issues | — |

Headline: every persona reconstructed the central identity, the crossing-curves condition,
the CLML-as-tail-area definition, the proxy-scale risk, and the ICL mapping correctly —
including the skimmer. The framing survives contact with all four audience types.

## Main changes v1 → v2 (by source)

**Correctness (fact-checker + Bayesian reviewer):**
- Fixed inverted figure caption for the decision-boundary phase diagram (y-axis is
  *held-back size*, not conditioning-set size) — this was the one blocker.
- Bug paragraph re-attributed: the near-exact CLML match is with the *non-BMA* validation
  loss (CNNs, per the difference overlay); v1 of Lotfi et al. accidentally computed the
  *BMA* validation loss; "barely changed the plots" → "left the qualitative take-aways
  unchanged"; footnote now notes the authors read the results more favorably.
- "Given enough data all three criteria agree" now conditioned on *asymptotes differing*,
  plus a new paragraph on the tied-asymptote case: Bayes-factor consistency vs. CV's
  inconsistency for nested models — the marginal likelihood's irreplaceable use case
  (also the Bayesian reviewer's main fairness complaint about v1).
- Bernstein–von Mises usage corrected (misspecified variant; fails for singular models —
  why d/2 becomes Watanabe's λ ≤ d/2; realizable-case hedge footnote).
- k = 1 ↔ leave-one-out CV now stated with the exchangeability/averaging caveat, with the
  symmetry intuition spelled out.
- "Prior-data conflict" no longer used as a synonym for slow descent: descent speed is
  "prior-data misfit", presented as a spectrum from vagueness (Occam penalty working as
  designed) to Evans–Moshonov-sense conflict.
- Hook no longer claims choosing the final checkpoint means disagreeing with the LML's
  *answer* — it means choosing a different *criterion*.
- TL;DR sign slips fixed (negative log ML = sum of losses; *log* evidence = minus area).
- Broken Wikipedia "prequential analysis" link removed (MDL article linked instead);
  "assums" typo; nats-vs-bits note; "−LML/evidence" appositive fixed in the table;
  toy captions re-anchored to "eventual held-out winner"; PFNs = "prior-data fitted
  networks"; split-dial wording softened to cite the actual heuristics (80/20; 10–50%).

**LLM-practice accuracy (pretraining researcher):**
- New "two honest caveats" paragraph: optimizer + LR schedule = inference mechanism (not
  prior), so areas compare *learners* and only make sense across matched recipes; areas
  are order-sensitive where final height mostly isn't (curricula are part of what's scored).
- Single-epoch framing updated for data-constrained practice (Muennighoff et al. 2023,
  link verified); second-epoch losses aren't code lengths; training-loss-EMA caveat
  (training-mixture only, not comparable across mixtures; doesn't replace per-domain vals).
- Scaling-law passage now makes the substitution explicit (curve height at step n = loss
  after n points) and maps the two axes onto the L(D) = E + A·D^(−α) fit: floor E =
  misspecification, (A, α) = misfit — plus a "fit both curves and test for a crossing
  before target scale" diagnostic.
- Sliding-window perplexity evaluation replaces "common ICL hygiene" as the everyday CLML.
- Named live example: SSM/fixed-state copying-retrieval limits (Jelassi et al. 2024, link
  verified) as matched-descent/different-floor.
- Kaplan et al. LSTM claim moved to the ICL section (it is a per-token-position result).
- Training-speed estimators noted to inherit the LML's early-mistake books.
- New "when are you in the crossing regime" heuristic paragraph. NOTE: the specific
  heuristic (matched-recipe ablations → near-parallel curves; crossings cluster across
  families/regularization/mixture changes) came from the persona's claimed experience —
  it is plausible and hedged as "a field heuristic", but you may want to check it against
  your own experience before publishing.

**Accessibility (PhD student + skimmer):**
- TL;DR cut from ~215 to ~110 words and now includes the match-functional-to-question triage.
- Inline definitions added: Bayes factor, type-II ML/empirical Bayes, exchangeability,
  regular vs singular ("cash value" skip-note in footnote).
- Posterior predictive introduced as "the same integral with prior → posterior".
- Occam section compressed into a list; figure captions freed of math markup (MathJax-config
  independent); ICL section no longer overloads the symbol k; ℓ → ℓ_N in the table;
  terminology map notes H[·] is used loosely and moves the Fong & Holmes score to the
  averaged-over-orderings row.
- Figures switched to PNG (the area-under-curve SVG was 2.9 MB) except the 40 KB
  decision-boundary SVG.

## Panel feedback deliberately NOT implemented

- **"Add one real LM experiment"** (pretraining reviewer's top ask): two small runs on a
  public corpus showing area and height disagreeing would upgrade the post from frame to
  tool. Out of scope for this drafting pass, but I agree it's the single highest-value
  addition if you want to invest — the training-speed papers' setups would be a template.
- **Cutting the six-panel toy figure** (pretraining reviewer): kept, because the Bayesian
  reviewer and PhD student found the toy section load-bearing evidence ("exact criteria,
  no estimation excuses"). The phase diagram remains the punchline.
- **Cutting the Occam section entirely**: compressed instead — the model-description-length
  point and the long-cow footnote survive.

## Round 2: verification re-reviews of v2

The two most demanding reviewers re-read the revised draft against their own findings:

- **Bayesian researcher (Opus):** all 8 technical-audit items and all "missing" items
  RESOLVED; overall verdict "**now fair**" to both Lotfi et al. and the marginal-likelihood
  tradition; confirmed the new tied-asymptote paragraph is mathematically correct.
  Residual nits fixed in v3: dropped the "redundant block of parameters" example (truly
  redundant parameters integrate out penalty-free — it could nullify the Occam argument),
  added the asymptotic-ties qualifier to the TL;DR/hook crossing claim, cited Shao (1993)
  for CV's nested-model inconsistency, and disambiguated "plain (non-model-averaged)
  validation loss" in the bug paragraph.
- **Pretraining researcher (Fable):** 6/8 RESOLVED, 1 partial, 1 cosmetic; verdict flipped
  to "**would forward to the team channel**". New issues it caught in v2, fixed in v3:
  the order-insensitivity sentence overclaimed (deliberate curricula exist precisely
  because ordering moves the end state — now stated as asymmetric sensitivity); the
  tied-asymptote paragraph's advocacy overshoots (now: corner case for pretraining stacks /
  home turf for Bayesian questions; signal *grows* vs O(1) vs decays); added the
  NAS-scale-not-frontier-scale scope on training-speed estimators, the decaying-LR-bias
  caveat on reading L(D) off a single run, the reinstated "no fully principled split"
  clause, the Evans & Moshonov link, and removed the unverified Kaplan token count.

Both re-reviewers independently repeated the one big ask: **a real LM experiment** showing
area and height disagreeing ("the last thing between 'good conceptual post' and 'post
people at my lab argue about in the thread'").

## Remaining items for the author

- When moving to the personal blog: copy the five figures (or the responsive webp variants),
  fix `/assets/...` paths for the site's baseurl, and confirm `$$...$$` inline math and
  kramdown footnotes render there.
- Spot-check the `#appendix` anchor on the published original post, and the two Colab links.
- The Evans & Moshonov (2006) and Kleijn–van der Vaart references are cited by name without
  links (paywalled venues); add links if you have preferred versions.
- Consider whether you want an explicit note that the post's "field heuristic" and the
  SSM example postdate the original post's content (they are new claims, not restatements).
