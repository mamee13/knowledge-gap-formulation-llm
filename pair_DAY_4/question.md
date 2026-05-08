# Day 4 — Question

**Asker:** Mamaru Yirga
**Topic:** Evaluation and statistics

---

## My Question

What Cohen's Kappa of 0.000 at 88.9% Agreement Actually Means and Whether My Rubric Is Reliable

In `docs/inter_rater_agreement.md`, `tone_quality` achieves 88.9% raw agreement between my two
labeling rounds but a Cohen's kappa of 0.000. I note that "kappa=0 with non-zero agreement indicates
agreement is at chance level given marginal distributions" but I cannot explain the mechanism — why
does 88.9% agreement produce a kappa of zero, what marginal distribution would cause that, and whether
a rubric dimension with kappa=0.000 and 88.9% agreement is actually reliable or not. My
`docs/inter_rater_agreement.md` concludes the rubric "meets the ≥80% agreement threshold" and uses
raw agreement as the primary metric, but I cannot defend that choice against a reviewer who asks why I
did not use kappa as the primary metric. Knowing this would let me revise the "Key Finding" section of
`docs/inter_rater_agreement.md` to give a mechanically grounded defense of using raw agreement over
kappa for this specific rubric design.

**Artifact pointer:** `docs/inter_rater_agreement.md` → Round 1 vs Round 2 Results table + Key Finding  
**Grounding commit target:** Add one paragraph to the Key Finding section explaining the kappa paradox
for imbalanced binary labels, and why raw agreement is the appropriate primary metric when one label
dominates the marginal distribution.

