# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve 1.73M test Source 1 businesses against 10M Source 2/3 records with a scalable
**blocking + gradient-boosted matcher** pipeline that runs on a single laptop. Three hashed
TF-IDF retrievers over *combination tokens* (word pairs, region-tagged words, consonant
skeletons, joined names) feed a learned pruner that keeps 40 candidates per business; a
**3-seed LightGBM ensemble** with a stacked second stage scores them, and per-business
match sets are chosen with **per-country thresholds tuned directly on the macro F0.5 metric
on out-of-fold predictions**. Key ideas: rule-based Indic-script transliteration that
removes zero-width joiners and recognises Indic legal abbreviations, house-number arithmetic
that exposes generated decoys, **character n-gram features** for transliterated names and
typos, and competition features between businesses claiming the same record.

**Final model (e011) macro F0.5 on out-of-fold predictions: 0.9723** (US=0.978, IN=0.963,
FR unseen). Score progression from earlier runs: e007=0.9695 → e008=0.9712 → e009=0.9713 →
e010=0.9716 → **e011=0.9723** (3-seed ensemble).

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploring the 2.2M-entity training set (all from provided data only):

| Finding | Value | Consequence |
| --- | --- | --- |
| Singletons | 5.6% of S1 | Recall across *all* of an entity's matches dominates the metric |
| Matches per S1 | mode 3, up to 10+ (1-2 in S2 + 1-3 in S3) | Extra-match decisions matter |
| S2/S3 record in >1 ground-truth list | 0 | Each pool record belongs to at most one S1 (one-to-one constraint) |
| Unmatched pool records | 26% of S2+S3 | Many distractors, including generated decoys |
| S1 names shared with another S1 (same country) | 50% | Chains: address must decide; name-only copies are ambiguous |
| Records with Indic-script text | ~22% of India | Names written in Devanagari, Telugu, Tamil, Gujarati, Bengali |
| US postcodes | almost none | Street, number, city and state carry the address signal |
| France | test only (15% of test S1) | No country one-hot; every feature country-agnostic |

Noise patterns observed in true copies: legal-suffix changes and reordering (`LLC Clark
Industries`), typos (`Pineapp1e`), random or website names (`Calozeph`, `@emmylove`,
`maguiresprairiecafe.com`), DBA names, empty addresses, dropped or zero-padded house-number
digits (`2777 -> 277`, `5445 -> 05445`), city/neighbourhood swaps, transliterated state names.

**Decoys.** 83% of our false merges were pool records that belong to no business. They copy a
real business but shift its house number by a small *positive* offset from
{1, 2, 3, 4, 5, 7, 9, 11, 13, 21} (e.g. `43550 -> 43571`) and sometimes add `PMB nnnn` or
`No nn`. True copies keep the exact number in 91% of pairs.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pruner + stacked gradient-boosted classifier + metric-tuned set decision  
**Core Innovation:** Combination-token retrieval (word pairs, region-tagged tokens, skeletons)
that keeps blocking recall high without a vocabulary explosion, together with decoy-aware
number features and cross-business competition features.

Pipeline: normalise (parallel, cached) → per-country retrieval → learned pruning to 25
candidates → ~110 pair/context features → LightGBM stage 1 → stacked stage 2 → one-to-one
resolution → thresholds tuned on out-of-fold macro F0.5 → output files.

---

## 3. Candidate Generation (Blocking)

- **Normalisation:** accent stripping; rule-based Indic-to-Latin transliteration (one table
  for all ISCII-layout scripts, zero-width joiners removed); legal-suffix stripping including
  Indic abbreviations (`pra. li.` = Pvt. Ltd.); DBA split; abbreviation and state
  dictionaries for US, India and France; consonant skeleton with devoicing and English
  soft/hard `c` so transliterations match (`Southern Infra LLP` and its Telugu spelling share
  the skeleton `strn infr`).
- **Blocking keys used:** three hashed TF-IDF views searched within each country with
  multi-threaded sparse top-k (`sparse_dot_topn`, Apache-2.0):
  - *name*: words, DBA words, consonant skeletons, order-free word pairs, joined forms
    (`=emmylove`), and words/skeletons tagged with the region (`clark@az`, `~svstk@bihar`);
  - *address*: words, house numbers plus a 3-digit prefix (so dropped digits still match),
    adjacent word pairs, postcode;
  - *name+address*: both combined, so a record must agree on name *and* address to rank high.
  Region = US state (read from the raw address), Indian state, or the French département
  (postcode prefix). Tokens appearing in more than 20,000 records are dropped from retrieval;
  because pairs and region tags are rare even when their words are common, this keeps the
  sparse product cheap without losing discriminating information.
- **Pruning:** the union of the three searches (60 per view) is ranked by a small LightGBM
  pruner trained on a *separate* S1 sample (similarities, per-view ranks, gaps to the best
  candidate, script flags) and cut to 25 candidates per S1. `candidate_pairs.tsv` is exactly
  this set.
- **Candidate pairs generated:** 25 per S1 entity (≈43M pairs for the test set).
- **How we ensured true matches were not lost:** blocking is measured separately from the
  model (pair recall, oracle macro-F0.5 ceiling, recall@k) on every experiment, and every
  missed pair is inspected. This found the original failure (dropping common single tokens
  such as city names) and drove each retrieval change:

| Version | Pair recall | Oracle ceiling (macro F0.5) |
| --- | --- | --- |
| Word TF-IDF (name, address) | 0.826 | 0.920 |
| + combined name+address view | 0.871 | 0.947 |
| + word pairs, region tags, DBA, joined names | 0.943 | 0.980 |
| + learned pruner | 0.960 | 0.986 |
| + deeper search, skeleton region tags, 25 candidates | 0.969 | 0.990 |

---

## 4. Matching Model

**Features used (~92 stage-1 + 17 stage-2 = ~109, e011):**
- Name features: rapidfuzz ratio / partial / token-sort / token-set, Jaro-Winkler,
  normalised Levenshtein, skeleton similarity, no-space similarity (for handles and
  websites), IDF-weighted token overlap, rarest shared and unshared token, acronym match,
  legal-suffix agreement, DBA cross-match, numbers in names, **character n-gram Jaccard
  (trigram and 4-gram), initial-letter match, first-token containment** (e010).
- Address features: token-set / sort / partial ratios on the cleaned address, Jaccard,
  postcode agreement, landmark agreement, house-number agreement, **signed house-number
  difference, minimum absolute difference and unmatched numbers** (decoy detection),
  unit-number difference, PMB / PO-box markers, missing-field flags, **address character
  n-gram Jaccard** (e010).
- Other: retrieval cosines and ranks per view, pruner score; context within each S1
  (rank, gap to best, candidate counts); **reverse competition** (rank of this S1 among all
  S1 retrieving the same candidate, margin to the best rival); copy-to-copy similarity to the
  S1's top candidates; chain frequency of the name; script flags.

**Model type:** LightGBM (MIT licence) binary classifier, 127 leaves, early stopping,
**3-seed ensemble (seeds 42, 13, 7) averaged at predict time** (e011), 3-fold out-of-fold
predictions grouped by S1. A **stage-2 LightGBM** re-scores each pair from the stage-1
probabilities of its neighbourhood (rank and gap within the S1, the strongest rival S1
claiming the same candidate); its dominant feature is the margin between this business and the
best rival. The stage-2 model sees the stage-1 p1 plus 50+ CARRY features (top stage-1 gain
features + per-S1 context) and a reverse-context block (e009).

**Training sample:** whole regions (US and Indian states) sampled per country (~173k S1),
retrieved against the *full* pool, so every sampled business's local competitors are present,
exactly as at test time when all S1 are scored.

**Threshold selection method:** each pool record is assigned to at most one S1 (the highest
probability). Per S1, the top candidate is kept if p ≥ t_first, the best candidate of the other
source if p ≥ t_other, further candidates if p ≥ t_extra. **Per-country thresholds**
(US, IN, FR unseen) are grid-searched on out-of-fold predictions against the exact macro F0.5
metric (singletons included), taking the centre of a flat region. A plug-in expected-F0.5 set
selector was also tested and did not beat the tuned thresholds.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold on region-complete training sample):** 0.9723 (e011,
  final model). Per-country breakdown: US=0.978, IN=0.963, FR (unseen) derived from group
  statistics. Local validation tracked the leaderboard closely (e001: 0.925 local, 0.915
  leaderboard).

| Experiment | Main change | Local macro F0.5 |
| --- | --- | --- |
| e001 | baseline retrieval + LightGBM | 0.925 |
| e004 | combination-token retrieval, region sample, reverse/sibling features | 0.958 |
| e005 | learned pruner, decoy number features | 0.9675 |
| e006 | deeper search, skeleton region tags, 25 candidates | 0.9704 |
| e007 | stage-2 stacking | 0.9695 |
| e008 | India script fixes (ZWJ strip, Indic legal abbrev, c-sound skeleton), k=100, max_k=40 | 0.9712 |
| e009 | expanded stage-2 CARRY (50+ features) + per-region thresholds | 0.9713 |
| e010 | char n-gram Jaccard (name+addr), initial-letter, first-token containment | 0.9716 |
| **e011** | **3-seed LightGBM ensemble (seeds 42,13,7) with all e010 features** | **0.9723** |

- **Common false positives (wrong merges):** generated decoys — same name and street with a
  shifted house or unit number (`D.No.1-11-29/4` vs `/11`), especially when the S1 record has no
  number and the decoy adds one; businesses sharing a building.
- **Common false negatives (missed matches):** name-only copies (empty address) of chain names
  shared by several S1 businesses, which are inherently ambiguous; random-name copies at an
  address shared by several businesses; heavily truncated Indian addresses combined with
  transliterated names.

---

## 6. Conclusion

A carefully measured blocking stage mattered more than model choice: moving from single-word to
combination tokens lifted the oracle ceiling from 0.92 to 0.99 at a fixed small candidate
budget. The remaining gains came from understanding how the data was generated (decoy
house-number offsets, transliterated names, chains) and modelling competition between
businesses. The whole pipeline uses only the provided data, standard CPU libraries and
MIT/Apache-licensed models.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — see its `README.md` for exact commands.

| Module | Purpose |
| --- | --- |
| `src/ber/io.py` | Safe TSV reading/writing |
| `src/ber/translit.py`, `normalize.py`, `dictionaries.py` | Transliteration and multi-view normalisation |
| `src/ber/prep.py` | Parallel normalisation to parquet (cached by code hash) |
| `src/ber/retrieve.py` | Hashed TF-IDF retrieval, union, learned pruning |
| `src/ber/train_pruner.py` | Trains the pruner on a separate S1 sample |
| `src/ber/featurize.py` | Pair, context, reverse and sibling features (parallel) |
| `src/ber/model.py`, `stage2.py` | LightGBM stage 1 and stacked stage 2 |
| `src/ber/decide.py`, `metrics.py` | One-to-one resolution, thresholds, exact metric |
| `src/ber/run_experiment.py`, `make_submission.py` | Training experiment; test outputs + validation |

Reproduce final submission: `python -m ber.run_experiment configs/exp/e011_ensemble.yaml`,
then `python -m ber.make_submission configs/exp/e011_ensemble.yaml` (writes both files to
`output/`).

### B. Additional Results

Error buckets for the best model are written to `results/<exp>/error_buckets.csv`; the blocking
recall@k curve and feature importances to `metrics.json` and `feature_importance.csv`.
