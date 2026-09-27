# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve 1.73M test Source 1 businesses against 10M Source 2/3 records with a scalable
**blocking + gradient-boosted matcher** pipeline that runs on a single laptop. Three hashed
TF-IDF retrievers over *combination tokens* (word pairs, region-tagged words, consonant
skeletons, joined names) feed a learned pruner that keeps 25 candidates per business; a
LightGBM pair model plus a stacked second stage scores them, and per-business match sets are
chosen with thresholds tuned directly on the macro F0.5 metric. Key ideas: rule-based
Indic-script transliteration, house-number arithmetic that exposes generated decoys, and
competition features between businesses claiming the same record.

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
| + second-hop retrieval from likely copies (≤5 extra per S1) | 0.975 | 0.991 |

**Second-hop retrieval.** Copies of one business resemble *each other* even when one drifted far
from the S1 record (random name at the copy's exact address, name-only record with the copy's
spelling). For each S1 we take its likely copies (top-3 candidates with a high name+address
similarity) and retrieve *their* nearest pool neighbours in all three views, adding up to five new
candidates flagged `is_hop`.

**France (test-only).** French copies alternate between the région ("Nouvelle-Aquitaine"), the
département ("Gironde") or neither, which the training countries never show (US states normalise
to one code). These phrases are removed from French addresses before retrieval and features. On
the leaderboard this lifted the implied French score from ≈0.86 to ≈0.915.

---

## 4. Matching Model

**Features used (~110):**
- Name features: rapidfuzz ratio / partial / token-sort / token-set, Jaro-Winkler,
  normalised Levenshtein, skeleton similarity, no-space similarity (for handles and
  websites), IDF-weighted token overlap, rarest shared and unshared token, acronym match,
  legal-suffix agreement, DBA cross-match, numbers in names.
- Address features: token-set / sort / partial ratios on the cleaned address, Jaccard,
  postcode agreement, landmark agreement, house-number agreement, **signed house-number
  difference, minimum absolute difference and unmatched numbers** (decoy detection),
  unit-number difference, PMB / PO-box markers, missing-field flags.
- Other: retrieval cosines and ranks per view, pruner score; context within each S1
  (rank, gap to best, candidate counts); **reverse competition** (rank of this S1 among all
  S1 retrieving the same candidate, margin to the best rival); copy-to-copy similarity to the
  S1's top candidates; chain frequency of the name; script flags.

**Model type:** LightGBM (MIT licence) binary classifier, 127 leaves, early stopping, 3-fold
out-of-fold predictions grouped by S1. A **stage-2 LightGBM** re-scores each pair from the
stage-1 probabilities of its neighbourhood (rank and gap within the S1, the strongest rival S1
claiming the same candidate); its dominant feature is the margin between this business and the
best rival. Stage 2 also carries **cluster-consistency** features: the consensus house number and
canonical legal form among the S1's strong candidates, and whether each candidate agrees (decoys
copy a real record but shift the number or change the legal form, so they disagree with the
consensus even when the S1 record itself has no number); and **source-cardinality** features
(strong S2 / S3 copies found so far, whether a candidate is the best of a still-empty source).
These halved the false merges (pair precision 0.993 → 0.996).

**Test-density weighting.** The test pool has ~1.1 more unmatched records per business than the
training pool, so test businesses more often have several borderline candidates (India: 7.9% with
2+ borderline candidates vs 3.5% in training). Training rows and the threshold search are weighted
per S1 by P_test(bucket) / P_train(bucket) of that ambiguity bucket, so the model and thresholds
target the test distribution.

**Training sample:** whole regions (US and Indian states) sampled per country (~173k S1),
retrieved against the *full* pool, so every sampled business's local competitors are present,
exactly as at test time when all S1 are scored.

**Threshold selection method:** each pool record is assigned to at most one S1 (the highest
probability). Per S1, the top candidate is kept if p ≥ t_first, the best candidate of the other
source if p ≥ t_other, further candidates if p ≥ t_extra. The three thresholds are grid-searched
on out-of-fold predictions against the exact macro F0.5 metric (singletons included), taking the
centre of a flat region. A plug-in expected-F0.5 set selector was also tested and did not beat
the tuned thresholds.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold on region-complete training sample):** 0.9784 (e011);
  best leaderboard score 0.962 (e011).

| Experiment | Main change | Local macro F0.5 | Leaderboard |
| --- | --- | --- | --- |
| e001 | baseline retrieval + LightGBM | 0.925 | 0.915 |
| e004 | combination-token retrieval, region sample, reverse/sibling features | 0.958 | – |
| e005 | learned pruner, decoy number features | 0.9675 | 0.947 |
| e006 | deeper search, skeleton region tags, 25 candidates | 0.9704 | – |
| e007 | stage-2 stacking | 0.9716 | – |
| e008 | India script fixes (joiners, Indic legal abbreviations, c-sound skeleton) | 0.9723 | 0.949 |
| e010 | canonical legal forms + cluster-consistency stage 2 | 0.9774 | – |
| e011 | second-hop retrieval, source-cardinality features, French région/département cleanup | 0.9784 | **0.962** |
| e012 | test-density weighting | [fill] | [fill] |

**Where the remaining error is (oracle analysis on e010).** Making every decision on retrieved
candidates perfect would give 0.990; retrieving every true copy (with current decisions) 0.988.
About a third of the decision error is name-only copies of chain names shared by several S1
businesses; with the same legal form such a copy is a true match only ~11% of the time, so it is
largely irreducible from the provided fields.

**Local vs leaderboard gap.** Re-weighting validation to the test set's ambiguity profile
explains part of the gap (India 0.963 → 0.955); the remainder is France (no labels). A
leave-one-country-out check (train US, score India) measured the transfer loss and ruled out
per-country rank normalisation (0.885 → 0.874) and self-training as remedies.

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

Reproduce: `bash reproduce.sh` (pruner → e011 experiment and test scores → test-density weights →
e012 experiment and test scores; writes both files to `output/`). The steps are listed in
`README.md`.

### B. Additional Results

Error buckets for the best model are written to `results/<exp>/error_buckets.csv`; the blocking
recall@k curve and feature importances to `metrics.json` and `feature_importance.csv`.
