# Research validity and reproducibility

The technical audit identifies invalid independent-test interpretation, rather than
evidence of fabricated calculations. The reported 93.99% HAM and 85.91% ISIC
accuracies are historical fitting results: both cohorts contributed to additional
training and checkpoint selection. Source inspection does not verify those numbers.
Historical models and results must remain intact and labelled with their exposure.

## Safeguards in `dev`

- `adapt_panderm_once.py`, `adapt_six_models.py` and `refine_joint_cohorts.py`
  reject execution before loading training artifacts or writing results. Explicit
  `--allow-test-fitting` permits historical reproduction with a warning; it cannot
  restore an independent-test interpretation.
- Catalogs and comparison subsets use stable alphabetical ordering. Changing test
  scores or preferred-checkpoint flags cannot select a model. The original view
  excludes known contaminated checkpoints; it still contains inspected historical
  evaluations and is not labelled an independent benchmark.
- Adaptations remain selectable for inference and in a separate fitting archive.
  Comparisons against the unchanged reference disclose unequal training exposure.
- Federated UI distinguishes synthetic software checks, separate PanDerm training,
  and the evidence needed for real-data utility or achieved differential privacy.

## Recalculate actual prediction artifacts

Use an independently supplied ground-truth manifest, not labels copied from the
prediction file. Both truth and exposure CSVs require unique `image_id` and `label`
columns. Labels are `akiec,bcc,bkl,df,mel,nv,vasc`. For overlap checks also supply
globally reconciled `lesion_id`, image-content `image_sha256` and, when available,
`patient_id`. Prefixing IDs with release names does not resolve shared lesions.

The exposure manifest must contain the union of all training, calibration,
validation, hyperparameter and checkpoint-selection examples in the model lineage,
including earlier experimental branches. Deduplicate it by image ID after resolving
conflicting identities/labels. Previously inspected test cohorts are exposure too.
Unknown identities must remain unknown rather than being replaced with new IDs.

Predictions require `image_id` and seven probabilities named by class (or
`score_akiec`, etc.). Optional `true_label` is checked against independent truth.
Row order can differ; missing, duplicate or extra IDs are rejected.

```powershell
python scripts/audit_predictions.py --predictions experiment/predictions.csv --truth experiment/blind_truth.csv --exposure experiment/all_exposure.csv --reported-metrics experiment/metrics.json --split official_test --require-disjoint --output experiment/audit-receipt.json
```

The command recomputes metrics, checks supplied reported scores, lists identity
overlaps and hashes input artifacts. Missing image-hash or lesion evidence fails
`--require-disjoint`. Patient overlap is rejected when supplied; missing patient IDs
remain an explicit limitation. Without `--require-disjoint`, fitting results can be
audited without being certified as disjoint. Receipts never overwrite an existing file.
Passing this check proves consistency with supplied evidence; it cannot establish
that exposure is complete or predictions came from a particular checkpoint.

## Fresh blind evaluation still required

1. Choose one model using training-only cross-validation or a new lesion-disjoint
   validation split. The repeatedly inspected 575-image HAM validation cohort,
   including the reported 395 images used by the Base branch for training, cannot
   serve as fresh independent evidence across those branches.
2. Preserve all source exposure manifests, feature-extractor versions, preprocessing,
   seeds, dependency versions and training logs. Hash the selected checkpoint,
   scaler, feature projector, metadata and evaluation configuration before obtaining
   blind labels. Record that frozen choice with an independent evaluator.
3. Obtain a genuinely new labelled dataset; reconcile image duplicates, lesions and
   patients against the complete exposure history across HAM/ISIC releases. Have an
   independent evaluator verify the split. Do not infer independence from filename
   differences or from an empty intersection when identities are missing.
4. Run the frozen model once without training, calibration, threshold tuning or
   checkpoint reselection. Save image-level probabilities and independent labels,
   then use the audit command above. A code change cannot retroactively clean old
   scores or supply a new accuracy measurement.
5. For federated claims, run real-image federated and centralized experiments with
   matched backbones, splits, preprocessing and training budgets. Preserve client
   partitions and round histories. Report these separately from non-federated PanDerm.
6. For DP claims, retain achieved per-client epsilon, delta, accountant state,
   clipping, noise, sampling and cumulative steps across rounds. Identify the
   protected unit. A configured target epsilon or synthetic no-DP demonstration
   does not establish an achieved real-data privacy guarantee.

This branch does not rerun expensive training, generate replacement accuracy numbers,
establish patient independence, or alter the original worktree's report artifacts.
The existing Markdown/DOCX report and exported presentation remain historical
artifacts; this document records the current software safeguards and evidence gaps.
