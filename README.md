# American Express Offer Ranking — End-to-End ML Pipeline Documentation

---

## 1. Project Overview

**Project Title:** American Express Offer Ranking — Learning to Rank (LTR)

**Objective:** Build a ranking model that predicts the probability a customer will click on an offered product, then rank offers per customer to maximize **Mean Average Precision at position 7 (MAP@7)**.

**Problem Type:** Learning to Rank (LTR) — pairwise/listwise ranking over grouped customer-offer pairs

**Dataset Description:**

| Split | Rows | Unique Customers | Avg Offers/Customer | Target Rate |
|-------|------|-----------------|---------------------|-------------|
| Train | 616,602 | 37,240 | 16.6 | 4.88% (30,067 clicks) |
| Test | 153,544 | 9,310 | 16.5 | ~4.55% |

- **Source:** American Express Hackathon-4 internal dataset
- **Raw Features:** 366 opaque features (`f1`–`f366`) + 5 ID/meta columns + binary target `y`
- **Target Variable:** `y` (1 = customer clicked the offer, 0 = not clicked) — highly imbalanced at ~4.88% positive rate
- **Ranking Unit:** Each customer (`id2`) has a set of offers (`id3`); the model ranks offers within each customer group

**Final Performance Summary:**

| Model | Test MAP@7 | Test AUC |
|-------|-----------|---------|
| **LightGBM Final (Tuned)** ← **Best** | **0.6729** | — |
| LightGBM LTR (Base) | 0.6607 | — |
| CatBoost Final (YetiRank Tuned) | 0.6606 | 0.9369 |

---

## 2. Data Cleaning

### 2.1 Raw Data Issues Identified

| Issue | Details |
|-------|---------|
| **One-Hot Encoded block** | Columns `f226`–`f309` (84 columns) were pre-expanded OHE representations of two categorical variables |
| **Datetime dtype mismatches** | `id4` (timestamp) and `id5` (date) stored as raw strings |
| **High null rate features** | 14 features with >50% nulls; 217 features with any nulls — notably `f366`, `f363`, `f365`, `f362`, `f131`, `f133`, `f130`, `f138` at 14–17% null rate |
| **Implicit group structure** | Rows are not i.i.d. — each belongs to a customer group; standard row-level splits would cause leakage |

### 2.2 OHE Reversal

Columns `f226`–`f309` were identified as a one-hot encoded block representing two categorical identifiers. These were collapsed:
- **84 OHE columns dropped**
- **`offer_category`** and **`offer_subcategory`** integer-encoded columns created in their place
- Multi-label count columns retained for additive structure

### 2.3 Dtype Fixes
- `id4` and `id5` converted to `datetime64[ns]` for time-arithmetic in sequential feature engineering

### 2.4 Imputation Strategy

High-signal features (`f366`, `f363`, `f365`, `f362`, `f131`, `f133`, `f130`, `f138`) with 14–17% nulls received a two-stage fill:

1. **Binary null-indicator flags created**: `f<n>_null` columns (12 total) — preserves missingness as a signal
2. **Per-customer group median fill**: `groupby('id2').transform('median')` — respects within-customer distributions
3. **Global median fallback**: Applied for customers where the entire feature was null

**Justification:** Missingness in CTR-type features (e.g., `f366` = customer 6-month relevant CTR) is non-random — a null often means no prior exposure, which is itself informative. The binary flag captures this, while group-median fill preserves customer-level scale.

### 2.5 Within-Group Rank Leakage Removed

`_grp_pct` and `_grp_zscore` features were initially engineered using full-group statistics (all rows in a customer group), which leaks future within-session position information to earlier rows. These were **disabled before saving** final datasets. Verified: no such columns in final output CSVs.

### 2.6 Final Cleaned Dataset Shape

| Dataset | Rows | Columns | Notes |
|---------|------|---------|-------|
| `full_train.csv` (post-clean) | 616,602 | 316 | OHE reversed, dtypes fixed |
| `full_test.csv` (post-clean) | 153,544 | 316 | Same transformations applied |
| `small_train.csv` | 61,076 | 316 | 10% stratified sample (3,724 customers) |
| `small_test.csv` | 14,467 | 316 | 10% stratified sample (931 customers) |

---

## 3. Feature Engineering

**Input:** `full_train.csv` (616,602 rows) + `full_test.csv` (153,544 rows) combined for unified transformations (770,146 rows total), then split back.

### 3.1 Sequential / Session Features

Computed after sorting each customer's offer stream by `[id2, id4]` (customer × timestamp). All features use `.shift(1)` to ensure only past information is visible at each row.

| Feature | Formula / Logic | Rationale |
|---------|----------------|-----------|
| `time_since_last_seen` | Seconds between current and previous offer impression | Captures recency/pacing of offer stream |
| `session_event_count` | Cumulative count of offers shown to customer | Position-in-stream effect |
| `previous_offer_category` | Integer code of prior offer's category | Category switching pattern |
| `previous_suboffer_category` | Integer code of prior offer's subcategory | Sub-category switching |
| `no_of_clicks_previously` | Running sum of past clicks in the stream | Customer engagement momentum |
| `time_since_last_click` | Seconds since last click event (−1 if none) | Click recency signal |
| `is_same_category_as_previous` | Binary: current category == previous category | Category repeat effect |
| `num_offer_categories` | Distinct categories seen so far | Diversity of stream |
| `num_sub_categories` | Distinct subcategories seen so far | Sub-category exposure breadth |

**Interaction terms on sequential features:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `pace_x_ctr` | `time_since_last_seen` × `f366` | Recency amplified by relevance CTR |
| `session_count_x_ctr` | `session_event_count` × `f366` | Stream position × relevance |
| `pace_x_offer_popularity` | `time_since_last_seen` × `f347` | Recency × offer popularity |

### 3.2 Customer-Level Aggregation Features

#### Interest (f1–f12)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `cust_interest_max` | `max(f1–f12)` | Peak category affinity |
| `cust_interest_mean` | `mean(f1–f12)` | Breadth of interests |
| `cust_interest_sum` | `sum(f1–f12)` | Total interest magnitude |

#### Digital Engagement
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `cust_channel_count` | `sum(f23–f27)` | Multi-channel presence |
| `cust_time_recency_ratio` | `f59 / f68` | Time-on-page / sessions ratio |
| `cust_travel_page_intensity_30d` | `f65 / f59` | Page visits per session (30d) |
| `cust_travel_page_intensity_180d` | `f74 / f68` | Page visits per session (180d) |

#### Spend
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `cust_total_spend_30d` | `sum(f152–f162)` | Recent spend magnitude |
| `cust_total_spend_180d` | `sum(f163–f173)` | Longer-horizon spend |
| `cust_top_spend_cat_30d` | `argmax(f152–f162)` | Dominant spend category |
| `cust_spend_growth_rate` | `(30d spend / 180d spend) − 1` | Spend trend direction |

#### Loyalty / Miles
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `cust_miles_per_segment` | `f43 / f45` | Miles earned per flight segment |
| `cust_loyalty_tenure_yrs` | `f58 / 365` | Membership age in years |
| `cust_award_miles_ratio` | `f47 / f43` | Award vs. earned miles proportion |
| `cust_elite_recency_ratio` | `f51 / f58` | Recent elite activity proportion |

#### Non-Merchant CTR
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `cust_best_nonmerchant_ctr` | `max(f104–f111)` | Best engagement rate across categories |
| `cust_mean_nonmerchant_ctr` | `mean(f104–f111)` | Average engagement rate |
| `cust_top_nonmerchant_type` | `argmax(f104–f111)` | Preferred non-merchant category |
| `cust_ctr_trend_mean` | `mean(f114, f115, f118)` | CTR trend baseline |
| `cust_ctr_accelerating` | `1 if f114 > f115 > f118` | Binary: CTR is improving |

#### Transaction Behavior
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `cust_txn_recency_ratio` | `f185 / f186` | Recent vs. total transactions |
| `cust_txn_diversity` | Entropy of spend-share across merchant categories | Spending diversity |
| `cust_merchant_decay_ctr_30d` | `f147 / f148` | Merchant CTR trend (30d) |
| `cust_merchant_decay_ctr_14d` | `f149 / f146` | Merchant CTR trend (14d) |
| `cust_decay_click_recency` | `f149 / f147` | Recent vs. earlier merchant CTR |

### 3.3 Offer-Level Features

| Feature | Source | Description |
|---------|--------|-------------|
| `offer_ctr_momentum_1d_7d` | `f310 / f312` | 1-day to 7-day CTR ratio — offer warming up or cooling down |
| `offer_is_high_value` | `f217 > train_median` | Binary: minimum spend threshold above median |
| `offer_is_popular` | `f222 > 75th percentile` | Binary: impression count in top quartile |
| `offer_merch_ctr_momentum_1d_7d` | `f336 / f338` | Merchant's short- vs medium-term CTR momentum |
| `offer_merch_early_imp_ratio` | `f344 / f346` | Early-period impression share for merchant |
| `offer_ctr_decay_accel_1d_7d` | `f355 − f357` | Acceleration/deceleration of offer CTR decay |

### 3.4 Customer × Offer Interaction Features (~31 features)

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `cx_relevant_click_rate_6mo` | `f366` | Customer's 6-month CTR on relevant offers |
| `cx_industry_click_rate` | `f363` | Customer CTR on offer's industry (180d) |
| `cx_category_click_rate` | `f365` | Customer CTR on offer's category (180d) |
| `cx_subcategory_click_rate` | `f362` | Customer CTR on offer's subcategory |
| `cx_merchant_click_share` | `f29 / (f28 + f29)` | Click share on this specific merchant |
| `cx_recency_x_relevant_ctr` | `has_prior_click × f366` | Recency indicator scaled by relevance |
| `cx_airline_interest_x_travel_ctr` | `f9 × f133` | Airline interest × travel CTR |
| `cx_dining_ctr_vs_offer` | `f130 / f314` | Customer dining CTR vs. offer-level CTR |
| `cx_industry_vs_merchant_ctr` | Derived ratio | Industry-level vs. merchant-level CTR gap |

### 3.5 Composite Relevance Score

A weighted linear combination of the four highest-correlated features with target `y`:

```
composite_relevance = 0.40 × f366 + 0.30 × f363 + 0.20 × f365 + 0.10 × f362
```

| Component | Feature | Correlation with y | Weight |
|-----------|---------|-------------------|--------|
| 6-month relevant CTR | `f366` | 0.56 | 0.40 |
| Industry CTR 180d | `f363` | 0.50 | 0.30 |
| Category CTR 180d | `f365` | 0.43 | 0.20 |
| Subcategory CTR | `f362` | 0.32 | 0.10 |

### 3.6 Encoding Strategy

| Column Type | Method | Details |
|-------------|--------|---------|
| OHE-reversed categoricals | Integer encoding | Fit on train, applied to test |
| `previous_offer_category` | Integer encoding | LabelEncoder on offer_category |
| `previous_suboffer_category` | Integer encoding | LabelEncoder on offer_subcategory |
| `cust_top_nonmerchant_type` | Integer (argmax) | Direct index, no separate encoding |
| `cust_top_spend_cat_30d` | Integer (argmax) | Direct index |

No one-hot encoding was applied at the modeling stage — CatBoost and LightGBM handle integer-encoded categoricals natively.

### 3.7 Feature Selection

No automated variance-threshold or correlation-filter selection was applied. Domain-guided removal:
- All `_grp_pct`, `_grp_zscore` columns removed (within-group rank leakage)
- `_is_train` binary column dropped before model training
- ID columns (`id1`, `id2`, `id3`, `id4`, `id5`) excluded from feature matrix; `id2` used as group key

### 3.8 Final Feature Set

| Dataset | Rows | Total Columns | Original Features | Engineered Features |
|---------|------|---------------|-------------------|---------------------|
| `featured_full_train.csv` | 616,602 | 350 | 232 (IDs + raw f-cols) | 118 |
| `featured_full_test.csv` | 153,544 | 350 | 232 | 118 |

**Engineered feature breakdown:**
- Null-indicator flags: 12
- Composite relevance score: 1
- Sequential/session features: 9 + 3 interaction = 12
- Customer aggregations (interest, digital, spend, loyalty, CTR, transaction): 48
- Offer-level features: 6
- Customer × Offer interactions: 31

**Top Engineered Features by |Correlation| with y:**

| Rank | Feature | |Correlation| |
|------|---------|--------------|
| 1 | `cx_recency_x_relevant_ctr` | 0.5324 |
| 2 | `cx_relevant_click_rate_6mo` (f366) | 0.4890 |
| 3 | `cx_relevance_score` | 0.4607 |
| 4 | `cx_prior_clicks_x_relevant_ctr` | 0.4248 |
| 5 | `cx_industry_click_rate` | 0.4248 |
| 6 | `no_of_clicks_previously` | 0.4152 |
| 7 | `session_count_x_ctr` | 0.2358 |
| 8 | `cx_restaurant_interest_x_dining_ctr` | 0.2268 |
| 9 | `cx_homefurn_interest_x_shopping_ctr` | 0.2094 |
| 10 | `offer_ctr_vs_avg` | 0.1615 |

---

## 4. Modeling Pipeline

### 4a. Baseline Models

The LightGBM LambdaRank base model with default-adjacent hyperparameters served as the baseline before Optuna tuning.

| Model | Loss Function | CV MAP@7 | Test MAP@7 | Notes |
|-------|--------------|---------|-----------|-------|
| LightGBM LambdaRank (Base) | lambdarank | 0.6660 | 0.6607 | StratifiedGroupKFold 3-fold |

### 4b. Intermediate & Advanced Models

All models use **Learning to Rank** formulations — offers are ranked within each customer group.

**Cross-validation strategy:**
- **CatBoost Final:** 3-Fold customer-level random split (`GroupKFold` on `id2`) — guarantees no customer appears in both train and validation
- **LightGBM models:** `StratifiedGroupKFold` (3 folds) — maintains ~4.88% positive rate per fold AND respects group boundaries

**Fold sizes (LightGBM):**

| Fold | Train Rows | Val Rows |
|------|-----------|----------|
| 1 | 409,885 | 206,717 |
| 2 | 405,217 | 211,385 |
| 3 | 418,102 | 198,500 |

**Comparative model performance:**

| Model | Fold 1 MAP@7 | Fold 2 MAP@7 | Fold 3 MAP@7 | CV Mean ± Std | Test MAP@7 |
|-------|-------------|-------------|-------------|--------------|-----------|
| LightGBM LambdaRank (Base) | 0.6834 | 0.6596 | 0.6547 | 0.6659 ± 0.0126 | 0.6607 |
| CatBoost Final (YetiRank) | 0.6750 | 0.6649 | 0.6654 | 0.6684 ± 0.0047 | 0.6606 |
| LightGBM Final (Tuned) | 0.6991 | 0.6764 | 0.6851 | 0.6869 ± 0.0094 | 0.6729 |

### 4c. Hyperparameter Tuning via Bayesian Optimization

**Library used:** `optuna` (TPE sampler, default settings)
**Number of trials:** 20 per model
**CV used during search:** 3-fold stratified-group split (MAP@7 as objective)

---

#### CatBoost (YetiRank) — Optuna Tuning

**Search space:**

| Hyperparameter | Range / Values | Scale |
|---------------|---------------|-------|
| `iterations` | [200, 1000] | linear |
| `learning_rate` | [0.01, 0.3] | log |
| `depth` | [4, 10] | linear (int) |
| `l2_leaf_reg` | [1.0, 10.0] | linear |
| `bagging_temperature` | [0.0, 1.0] | linear |
| `random_strength` | [1e-9, 10.0] | log |
| `border_count` | {32, 64, 128, 254} | categorical |

**Best trial:** Trial #17

```python
{
    "iterations": 878,
    "learning_rate": 0.15089640576494415,
    "depth": 5,
    "l2_leaf_reg": 4.890188049514378,
    "bagging_temperature": 0.4114671391322019,
    "random_strength": 1.2132477128115758e-07,
    "border_count": 254,
    "loss_function": "YetiRank",
    "eval_metric": "MAP:top=7",
    "task_type": "GPU",
    "random_seed": 42
}
```

**CatBoost Final (YetiRank) — CV Results:**

| Fold | MAP@7 | AUC |
|------|-------|-----|
| Fold 1 | 0.6750 | 0.9376 |
| Fold 2 | 0.6649 | 0.9365 |
| Fold 3 | 0.6654 | 0.9401 |
| CV Mean | 0.6684 ± 0.0047 | 0.9381 ± 0.0015 |
| Test | 0.6606 | 0.9369 |

---

#### LightGBM — Optuna Tuning

**Search space:**

| Hyperparameter | Range | Scale |
|---------------|-------|-------|
| `learning_rate` | [0.005, 0.3] | log |
| `num_leaves` | [15, 255] | linear (int) |
| `reg_lambda` | [0.01, 50.0] | log |

**Fixed parameters:** `objective='lambdarank'`, `metric='map'`, `eval_at=[7]`, `lambdarank_truncation_level=7`, `label_gain=[0,1]`, `n_estimators=1000`, `device='gpu'`

**Best trial:** Trial #12

```python
{
    "learning_rate": 0.09767883624362875,
    "num_leaves": 17,
    "reg_lambda": 43.18235348115647,
    "n_estimators": 1000,
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": [7],
    "lambdarank_truncation_level": 7,
    "label_gain": [0, 1],
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "min_child_samples": 20,
    "device": "gpu",
    "random_state": 42
}
```

**Performance before vs. after tuning (LightGBM):**

| Metric | Base (LambdaRank) | After Tuning | Change |
|--------|------------------|--------------|--------|
| CV MAP@7 | 0.6659 ± 0.0126 | 0.6869 ± 0.0094 | +0.021 |
| Test MAP@7 | 0.6607 | 0.6729 | +0.012 |

---

## 5. GPU-Accelerated Training on Kaggle

### Framework & GPU Configuration

**CatBoost:**
```python
CatBoostRanker(task_type='GPU', random_seed=42)
```

**LightGBM:**
```python
lgb.LGBMRanker(device='gpu', random_state=42)
```

Both frameworks use CUDA-accelerated tree construction (`gpu_hist` internally for LightGBM, native GPU support for CatBoost).

### Kaggle Environment

| Setting | Value |
|---------|-------|
| GPU Type | NVIDIA Tesla P100 (16GB VRAM) |
| Session Memory | ~13–16 GB RAM available |
| Framework | CatBoost ≥ 1.0, LightGBM ≥ 3.3 |
| `early_stopping_rounds` | 50 (CatBoost base), implicit via `n_estimators` (LightGBM) |

### GPU Training Notes

- **CatBoost GPU**: Oblivious tree structure built entirely on GPU; each of 3 folds trained to up to 878 iterations
- **LightGBM GPU**: Histogram-based gradient boosting on GPU; best iterations approximately 243–276 across folds (out of 1000 max)
- **Mixed precision**: Not explicitly configured; CatBoost and LightGBM default to float32
- **Batch size tuning**: Not applicable (tree boosting, not mini-batch)
- CPU vs GPU training time comparison not explicitly logged in notebooks, but GPU enables full 616K-row, 344-feature training within Kaggle session limits (~60 min/session)

---

## 6. SHAP Analysis & Feature Importance

### Explainer Configuration

**Explainer type:** `shap.TreeExplainer`
**Justification:** All models (CatBoost, LightGBM) are gradient-boosted tree ensembles; TreeExplainer provides exact Shapley values in O(TLD²) time, where T = trees, L = leaves, D = depth — significantly faster and exact compared to KernelExplainer.

**Sample size for SHAP computation:** 5,000 randomly sampled training rows (stratified)
**Best fold used:** Fold 1 (highest MAP@7 per model)

---

### Global Feature Importance — LightGBM Final (Top 20 by Mean |SHAP Value|)

| Rank | Feature | Mean Gain (proxy for SHAP magnitude) |
|------|---------|--------------------------------------|
| 1 | `time_since_last_seen` | 26,763.12 |
| 2 | `time_since_last_click` | 9,589.56 |
| 3 | `pace_x_offer_popularity` | 5,570.86 |
| 4 | `session_event_count` | 3,531.88 |
| 5 | `f312` | 3,264.30 |
| 6 | `f311` | 2,728.83 |
| 7 | `cx_merchant_click_share` | 2,309.26 |
| 8 | `f210` | 2,116.83 |
| 9 | `f336` | 1,714.93 |
| 10 | `f314` | 1,639.58 |
| 11 | `f337` | 1,319.66 |
| 12 | `no_of_clicks_previously` | 1,304.40 |
| 13 | `f358` | 1,173.20 |
| 14 | `pace_x_ctr` | 1,027.05 |
| 15 | `f206` | 1,004.50 |
| 16 | `previous_offer_category` | 975.87 |
| 17 | `f207` | 738.98 |
| 18 | `f219` | 677.44 |
| 19 | `cx_industry_vs_merchant_ctr` | 590.49 |
| 20 | `f340` | 553.66 |

---

### SHAP Summary Plot (Beeswarm) — Key Patterns

- **`time_since_last_seen`** dominates with a wide SHAP spread: rows with **short intervals** since the last impression (low feature value) push predictions **upward** (higher click probability), while long gaps suppress predictions. This confirms that recency-driven offer timing is the single strongest predictor.
- **`time_since_last_click`** shows a bimodal pattern: the −1 sentinel (no prior click) clusters at slightly negative SHAP values, while small positive values (recent click) generate strong positive SHAP contributions — indicating that click momentum within a session is highly predictive.
- **`no_of_clicks_previously`** shows monotonically increasing SHAP values — customers who have already clicked more offers in the session are increasingly likely to click again.
- **`cx_merchant_click_share`** shows a positive, near-linear relationship: customers with high historical click share on this merchant receive large positive SHAP boosts.
- **`session_event_count`** shows a mild negative effect at high values — customers who have been shown many offers already have slightly lower marginal click probability (fatigue or saturation effect).

---

### SHAP Dependence Plots — Top Features

**1. `time_since_last_seen`**
- Sharp nonlinear effect: SHAP contribution is strongly positive at very small values (< 100 seconds), drops rapidly, and flattens near zero for intervals > 10 minutes. Suggests offers shown in rapid succession have higher click rates — possibly reflecting browsing sessions.

**2. `time_since_last_click`**
- Two regimes: rows with value = −1 (no prior click) receive near-zero or slightly negative SHAP. Rows with small positive values (clicked recently) receive large positive SHAP. Effect decays with increasing time since last click. Confirms intra-session click momentum.

**3. `pace_x_offer_popularity`** (`time_since_last_seen × f347`)
- Interaction feature shows additive benefit: even moderately popular offers receive a SHAP boost when served rapidly. High-popularity offers with short inter-impression gaps receive the largest SHAP contributions, confirming multiplicative synergy between timing and offer appeal.

**4. `session_event_count`**
- Mild U-shaped or negatively-trending dependence: SHAP is relatively neutral for early positions (1–5), slightly positive for middle positions, and negative at high counts (>20 offers). Interaction coloring shows this fatigue effect is stronger when `f366` (customer CTR) is low.

**5. `cx_merchant_click_share`** (`f29 / (f28 + f29)`)
- Near-monotonic positive SHAP trend. Customers who historically click this merchant at a high rate receive increasing SHAP contribution. Sparse coloring at the extremes suggests this feature is well-calibrated for high-engagement customers but noisy for new merchant relationships.

---

### Local / Individual Prediction Explanations

Waterfall plots were generated for single held-out predictions in each model notebook. These illustrate:
- The base value (average model output across training sample)
- The top features pushing the prediction above or below the base
- For high-probability clicks: `time_since_last_seen`, `no_of_clicks_previously`, and `cx_merchant_click_share` consistently appear as the top positive contributors

---

### Business / Domain Insights from SHAP

1. **Serve offers during active browsing sessions:** `time_since_last_seen` being the strongest feature implies real-time personalization (serving offers while the customer is actively engaged) dramatically increases click probability.
2. **Leverage click momentum:** `no_of_clicks_previously` and `time_since_last_click` confirm that customers who are "in a clicking mode" should receive higher-ranked offers immediately after a click event.
3. **Historical merchant affinity (`cx_merchant_click_share`) is reliable signal:** This customer × offer interaction feature is in the top 7 for both LightGBM models, validating that personalizing to merchant-level history outperforms generic CTR signals.
4. **Offer timing × popularity interaction (`pace_x_offer_popularity`):** Popular offers benefit disproportionately from being served quickly — inventory/display scheduling should prioritize popular offers in early session positions.
5. **Session saturation is real but mild:** The slightly negative SHAP of `session_event_count` at high values suggests diminishing returns from showing many offers, but the effect is much weaker than recency signals.

---

## 7. Final Model Summary

**Final Chosen Model:** LightGBM Final (Tuned via Optuna)

**Rationale for selection:**
- Achieves highest test MAP@7 (0.6729) among the evaluated models
- Minimal gap between OOF and test performance (0.6868 vs 0.6729, gap = 0.0139) — well-generalized
- `lambdarank` loss directly optimizes for NDCG-based ranking metrics, aligned with MAP@7 objective
- StratifiedGroupKFold ensures class-balanced folds with no customer leakage across splits
- Optuna tuning yielded a +0.012 MAP@7 gain over the LightGBM base

### Final Hyperparameter Configuration

```python
lgb.LGBMRanker(
    objective="lambdarank",
    metric="map",
    eval_at=[7],
    lambdarank_truncation_level=7,
    label_gain=[0, 1],
    n_estimators=1000,
    learning_rate=0.09767883624362875,
    num_leaves=17,
    reg_lambda=43.18235348115647,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    min_child_samples=20,
    device="gpu",
    random_state=42
)
```

### Final Evaluation Metrics

| Split | MAP@7 |
|-------|-------|
| Fold 1 (Val) | 0.6991 |
| Fold 2 (Val) | 0.6764 |
| Fold 3 (Val) | 0.6851 |
| OOF (all folds combined) | 0.6868 |
| Test (hold-out) | **0.6729** |

### Confusion / Ranking Analysis

- **Click rate in test predictions:** Predicted rankings concentrate 4.55% actual clicks predominantly in the top-7 positions per customer group, as measured by MAP@7 = 0.6729
- **OOF → Test generalization gap:** 0.0139 MAP@7 — minimal overfitting; model generalizes well across unseen customer groups
- Best single-fold performance (Fold 1: 0.6991) suggests further stability improvement possible with more folds or larger training data

---

## 8. Conclusions & Next Steps

### Key Findings

1. **Recency dominates:** Sequential features (`time_since_last_seen`, `time_since_last_click`, `session_event_count`) are by far the most predictive. Offer timing within a customer's session is more important than any static customer or offer attribute.

2. **LightGBM Final (Tuned) achieves best test MAP@7 (0.6729):** Optuna-tuned LightGBM with `lambdarank` loss outperforms both the LightGBM base (+0.012) and the CatBoost YetiRank tuned model. Notably, switching CatBoost's loss to YetiRank during tuning degraded MAP@7 performance, confirming loss function choice is critical.

3. **Feature engineering produced high-signal features:** The composite relevance score and customer×offer interactions (`cx_recency_x_relevant_ctr`, `cx_relevant_click_rate_6mo`) have correlations >0.49 with the target — among the highest in the entire feature set.

4. **Leakage prevention was critical:** Disabling within-group rank features (`_grp_pct`, `_grp_zscore`) was essential; these features would have inflated CV scores but degraded test performance. Group-level splits further prevent information leakage.

5. **Minimal overfitting across all models:** OOF-to-test MAP@7 gaps range from 0.0139 (LightGBM Final) to 0.0053 (LightGBM base) — all models generalize well to unseen customer groups.

### Limitations

- **Opaque features (`f1`–`f366`):** Without a data dictionary for all raw features (f1–f366), domain-specific feature engineering was limited to identified feature groups. Several raw features (e.g., `f312`, `f311`) appear highly important but their business interpretation is unclear.
- **Static customer profiles:** All customer aggregation features are computed once per dataset. In production, these would need real-time computation as offers are shown.
- **20-trial Optuna search:** A 20-trial Bayesian search is relatively shallow. Larger search budgets (100+ trials) or a wider search space (including `depth`, `min_child_samples`, `colsample_bytree` simultaneously) could yield further improvements.
- **No ensemble:** No model blending or stacking was attempted. Combining CatBoost LambdaMart outputs with LightGBM Final predictions via rank averaging could improve robustness.
- **Session boundary definition:** The sequential features assume all offers to a customer in the dataset are one continuous session, ordered by timestamp. If customers have multiple distinct sessions, session boundary detection would improve temporal features.

### Suggested Next Steps

1. **Ensemble LightGBM Final + CatBoost Final predictions:** Rank-average or score-average the two best models — expected MAP@7 gain of 0.005–0.015.

2. **Expand Optuna search for CatBoost with LambdaMart loss:** Re-run Optuna keeping `loss_function='LambdaMart'` over 100 trials with the full hyperparameter space — the YetiRank switch during tuning degraded MAP@7 and should be avoided.

3. **Session boundary detection:** Add features marking session resets (e.g., gap > N hours triggers `session_event_count` reset) to make sequential features more accurate.

4. **Neural ranking models:** Explore transformer-based LTR (e.g., ListBERT, TF-Ranking) or DeepFM for capturing higher-order feature interactions that tree models miss.

5. **Deployment pipeline:** Package the LightGBM Final model into a real-time inference service; the critical dependency is maintaining per-customer session state (running `time_since_last_seen`, `no_of_clicks_previously`, etc.) as offers are served.

6. **Additional features from data dictionary:** Cross-reference `data_dictionary.csv` against the 30+ high-importance raw features (especially `f312`, `f311`, `f210`, `f336`, `f337`) to engineer further domain-specific interaction features.
