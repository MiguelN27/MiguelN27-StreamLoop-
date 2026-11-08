# StreamLoop churn model: tuning report

**Bottom line:** systematic tuning raised the business metric (F2) on the held-out test set from **0.498 to 0.720**.
The tuned model catches **301 of 374 churners** in the test set, against 178 for the first version.
The cost is 179 more retention offers sent to customers who would have stayed.
The cross-validation estimate (0.726 ± 0.019) predicted the test result well, so the score is not a lucky split.

Full work: [`src/churn_tuning.ipynb`](src/churn_tuning.ipynb). Data: 7,043 customers, 26.5% churn, a stratified 80/20 split (5,634 train / 1,409 test).

## 1. Baseline vs. tuned

| Test set (1,409 customers, 374 churners) | Baseline (default hyperparameters) | Tuned | Change |
|---|---|---|---|
| **F2** (primary metric) | 0.498 | **0.720** | +0.222 |
| Recall (share of churners caught) | 0.476 | 0.805 | +0.329 |
| Precision (share of flagged customers who really churn) | 0.610 | 0.507 | −0.103 |
| F1 | 0.535 | 0.622 | +0.087 |
| ROC-AUC | 0.817 | 0.843 | +0.027 |
| Accuracy | 0.780 | 0.740 | −0.040 |
| Churners missed | 196 | **73** | −123 |
| Unneeded retention offers | 114 | 293 | +179 |
| **CV F2 on training folds** (mean ± std) | 0.506 ± 0.024 | 0.726 ± 0.019 | |

The default model scored **below** a trivial "flag every customer" rule (CV F2 0.644), because it mostly predicted "stays".
The tuned model clears that floor by a wide margin. Its ROC-AUC, which doesn't depend on the decision threshold, also improved.
So the tuned model ranks customers better; it doesn't just flag more of them.

## 2. Final hyperparameters

Model: `Pipeline(median imputation + one-hot encoding → RandomForestClassifier)`.

| Hyperparameter | Value | Note |
|---|---|---|
| `class_weight` | `balanced` | The decisive setting: it makes errors on the minority (churn) class cost more during training |
| `max_depth` | 5 | Shallow trees |
| `min_samples_leaf` | 24 | Large leaves |
| `max_features` | 0.3 | Each split considers 30% of the features |
| `criterion` | `entropy` | Indistinguishable from `gini` (CV F2 0.7260 vs 0.7258) |
| `n_estimators` | 507 | Taken from the random search; the number of trees didn't affect F2 |
| `min_samples_split` | 16 | No effect: with leaves of 24, a node needs 48 samples to split anyway |

All other settings are sklearn defaults, with `random_state=42`.

## 3. Why F2 is the metric

StreamLoop loses far more from a churner it fails to flag than from a retention offer sent to a customer who would have stayed.
- **Accuracy** (the sklearn default) rewards ignoring churners. With 73% of customers staying, "never flag anyone" scores 73%.
- **Recall alone** can be gamed. "Flag everyone" scores a perfect 1.0.
- **F2** (F-beta with β = 2) treats recall as twice as important as precision. It rewards catching churners and still penalises indiscriminate flagging.

Every step used the same F2 scorer: the baseline CV, both searches, and both test evaluations.

## 4. How the search was done

All searches used the same 5-fold stratified CV, `scoring=F2`, `n_jobs=-1` and `refit=True`, and they were **fitted on the training split only**.

1. **Broad search: `RandomizedSearchCV`.** 60 configurations × 5 folds = 300 fits over seven hyperparameters (number of trees, depth, split size, leaf size, `max_features`, `class_weight`, `criterion`). Best CV F2: 0.725. Findings:
   - `class_weight` explained almost all of the gain. Every configuration without it scored at or below baseline level (0.45–0.54). Every configuration with it scored 0.68–0.725.
   - Shallow trees with large leaves did best.
   - `max_features`, `criterion` and `n_estimators` barely mattered.
2. **Narrowing.** Applied as rules computed from `cv_results_`, not picked by hand:
   - Categorical settings: keep the values used by at least 3 of the top 10, plus the winner's.
   - Complexity integers: the winner's value ± one step.
   - Number of trees: fixed at the winner's value.
3. **Focused search: `GridSearchCV`.** 108 combinations × 5 folds = 540 fits. Because `min_samples_split` had no effect, these were 60 distinct models. Best CV F2: 0.726, which confirms the region the random search found.
   - The best leaf size (24) sat at the edge of the grid. But across leaf sizes 12–24, the best score moved by only 0.0006, about 30× less than the fold-to-fold std. Extending the grid would have chased noise.

## 5. Choosing the final model: the stability trade-off

**The top candidates are tied on their mean score.** The top 10 distinct grid candidates span 0.7242–0.7260 mean CV F2. Each has a fold-to-fold std of about 0.015–0.020, so their means differ by much less than one fold's noise. Picking "the highest mean" would be picking on noise.

**The rule was set in the plan before the searches ran:**
1. Shortlist every candidate within one std of the best mean. All 60 distinct candidates qualified (cut-off 0.707).
2. Switch to a steadier candidate only if its std is lower by more than the uncertainty of a std measured on 5 folds (±0.0067). That means a challenger needed a std below 0.0123. The steadiest candidate had 0.0147, so none qualified.

**Result: we kept the grid's top-ranked model.** The deciding factor was simplicity, not its 0.0002 lead.
- It has the **smallest train–CV gap in the top 10** (0.016): the shallowest trees and the largest leaves, so the least overfitting.
- The trade-off we accepted: its worst fold (0.693) is among the lowest of the top 10. Candidate #10 is steadier (std 0.0167, worst fold 0.707), but it uses deeper trees with smaller leaves and has a larger gap (0.022).
- Both choices are defensible. With differences this small we preferred the more regularised model.

`refit=True` retrained the chosen model on the full training split; nothing was refit by hand.

## 6. Safeguards against meaningless tuning results

- **Split first.** The train/test split happened before any model work. Before the split we only did cleaning that learns nothing from the data:
  - dropped the customer ID
  - converted 11 blank `TotalCharges` values to missing
  - encoded `Churn` as 1/0
- **Preprocessing inside the pipeline.** Imputation and one-hot encoding are fitted on each training fold only.
- **Training data only for the searches.** Neither search ever saw the test set.
- **Test set used exactly twice**, once for the baseline and once for the final model. A counter in `evaluate()` raises an error on a third use.

## 7. Limitations and next steps

- **Decision threshold.** The model uses the default 0.5 threshold. The next gain is to tune it on out-of-fold predictions from the training data, for example with sklearn's `TunedThresholdClassifierCV`, and never on the test set.
- **Cost-based metric.** F2 stands in for the real costs. If finance can provide the cost of a lost customer and the cost of a retention offer, an explicit cost-based metric should replace it.
- **Tighter estimates.** The test set holds only 374 churners, so the test F2 has its own uncertainty of a few points. Repeated CV (for example 5×5 folds) on the shortlisted models would tighten the stability estimates before a production decision.
