# EP62 — Bias, Fairness & Ethics in Deep Learning

**Fairness audit of a neural network on synthetic loan data with three planted biases.**

## Quick Start

```bash
pip install torch scikit-learn numpy pandas matplotlib fairlearn
python ep62_audit.py


What the experiment does
Builds a loan-style dataset where the true creditworthiness is known
Plants three biases independently and together:
Label bias (historical rejections of deserving group-1 applicants)
Representation bias (group 1 under-represented in training)
Measurement bias (noisier features for group 1)

Trains a small neural network on the historical labels
Audits against the true outcomes (selection rate, TPR, FPR, gaps)
Runs Fairlearn ThresholdOptimizer (demographic parity & equalized odds)
Shows a scenario where true base rates differ → the two fairness metrics conflict
A gap is marked “real” only if it exceeds the control condition by 2 standard deviations across 5 seeds.Key measured results (mean of 5 seeds)














































ConditionAccuracyParity GapOdds GapSelection RatioControl0.7760.0160.0220.98Label bias0.7620.122*0.162*0.74Representation bias0.7760.0200.0280.98Measurement bias0.7380.0370.176*0.97All three0.7300.160*0.315*0.68* = real gap (clears control by 2σ)Important limitations
Completely synthetic data
Only two groups
Not legal advice
Real-world audits require domain knowledge, more groups, and legal review
Challenge
Run the script and post which sources drove the parity gap vs the odds gap
Vary LABEL_FLIP and find where the four-fifths rule first breaks
Add a third demographic group — how do the metrics behave?
