# Training all three classifiers

Run from `C:\Users\Manan Kulshrestha\Downloads\AI_project`.

```powershell
.\.venv\Scripts\python.exe -u scripts/train_feature_models.py --download
```

Alternatively double-click `train-all.cmd`. Run only one copy of the pipeline at
one time. A download and training job may already be running in this chat.

The command downloads the **entire official ISIC 2018 Task 3 release**: 10,015
HAM10000 training-collection images, 193 official validation images, and 1,512
official test images. It never falls back to synthetic data or a subset. Downloads
resume from verified HTTP ranges, validate archive CRCs and retain source hashes.

The pipeline then:

1. Splits the 7,470 HAM10000 lesions into disjoint train/validation/test groups,
   stratifying by diagnosis. With seed 42, the image counts are 7,014 / 1,509 /
   1,492. All 10,015 images are retained. Lesion grouping IDs are not supplied for
   the external official validation/test sets, so no cross-set lesion-disjointness
   claim is made for those external evaluations.
2. Extracts one 1,280-dimensional float32 feature vector per image using the
   original ImageNet EfficientNet-B0 weights and their official preprocessing.
   BatchNorm is retained and the backbone stays in evaluation mode with no
   gradients. Feature extraction resumes from its last completed batch.
3. Unloads EfficientNet. Fits a shared StandardScaler on **training features only**.
4. Trains a balanced multinomial logistic regression baseline (C=0.01, 0.1, 1), a
   1280 -> 256 -> 64 -> 7 MLP (LayerNorm, ReLU, dropout 0.4/0.25, AdamW, early
   stopping), and balanced XGBoost classifiers (depth 3/5, histogram trees, GPU
   where available, early stopping). These are initial comparison settings,
   not an exhaustive hyperparameter search.
5. Selects classifier settings using internal validation macro-F1, with balanced
   accuracy breaking ties. Evaluates saved/reloaded models on the internal test
   and external official sets without tuning on their labels.

One canonical image view is cached per image. No random augmentation or backbone
fine-tuning is used in this baseline. These are centralized classifiers, not
federated or differentially private models. The old image-model training scripts
remain separate; their checkpoints cannot be mixed with this feature pipeline.

## Results and logs

- `outputs/feature_models/efficientnet_b0/REPORT.md`: readable comparison.
- `outputs/feature_models/efficientnet_b0/comparison.csv` and `comparison.png`.
- `outputs/feature_models/efficientnet_b0/{linear,mlp,xgboost}/`: checkpoints,
  per-class metrics, predictions, confusion matrices and selected settings.
- `outputs/feature_models/efficientnet_b0/split_manifest.csv`: exact split IDs.
- `data/features/`: reusable feature arrays and provenance metadata.
- `artifacts/download-full-isic.log` and `artifacts/train-feature-models.log`:
  logs for the jobs started in this chat.

The Streamlit demo at http://127.0.0.1:8501 lists completed feature classifiers.
Refresh the page after training. Its default model is chosen using validation
results. The original frozen backbone is used again for each new image; only
classifier training avoids loading EfficientNet.

## Rerun and extend

To rerun training with the existing dataset and feature cache:

```powershell
.\.venv\Scripts\python.exe -u scripts/train_feature_models.py
```

To train the same three classifiers using another supported pretrained extractor:

```powershell
.\.venv\Scripts\python.exe -u scripts/train_feature_models.py --backbone resnet18
.\.venv\Scripts\python.exe -u scripts/train_feature_models.py --backbone mobilenet_v2
```

Each extractor has its own cache and output directory. Keep the same split seed
for a fair comparison. Use `--output-dir` for separate experimental runs.

## SMOTE experiments with F1 and F0.5

```powershell
.\.venv\Scripts\python.exe -u scripts/train_feature_models.py --resampling smote --smote-target 500
.\.venv\Scripts\python.exe scripts/compare_feature_runs.py
```

This preserves the original class-weighted checkpoints and saves new models under
`outputs/feature_models/efficientnet_b0_smote/`. The original train-only scaler is
fitted on real training examples, then SMOTE is applied only to scaled training
features. The same synthetic feature vectors are used for all three classifiers.
Classes with fewer than 500 training examples are raised to 500; no original rows
are discarded. For this dataset that adds 1,220 vectors to 7,014 real training
examples. Synthetic feature vectors are not additional real lesion images.

Class weighting is disabled in this SMOTE run, so this compares two balancing
strategies rather than stacking oversampling with class weighting. Image counts,
split IDs, validation data and test data are unchanged. The SMOTE seed, neighbor
count, class counts, package version and resampled-array hashes are recorded in
`resampling.json`.

Every new run reports macro and weighted F1/F0.5 and per-class F0.5 scores. Macro
scores average the seven classes equally; F0.5 emphasizes precision more than
recall. Model selection remains based on validation macro-F1 for consistency with
the original experiment. F0.5 is reported as an additional metric.

`BASELINE_COMPARISON.md`, `comparison_with_baseline.csv`, `per_class_comparison.csv`
and `baseline_comparison.png` contain the matched comparison. Original-run F0.5 is
calculated from its saved predictions without retraining or altering its artifacts.
The demo lists both the original and SMOTE models with separate names.

Test the new pipeline mechanics without downloading training data:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_feature_pipeline.py -q
```

Dataset source: https://challenge.isic-archive.com/data/#2018
Cite Tschandl, Rosendahl & Kittler (2018), https://doi.org/10.1038/sdata.2018.161,
and the ISIC 2018 challenge paper https://arxiv.org/abs/1902.03368. Observe the
source dataset license. This remains a research prototype, not a diagnostic tool.
# Improved frozen-backbone pipeline

Run `.venv\Scripts\python.exe scripts/train_improved.py` to extract/cache nine views
of every image and train all three classifier families. Results and checkpoints are
written separately to `outputs/feature_models/efficientnet_b0_improved/`.
The original runs and their split assignments are preserved.

The backbone remains frozen. Full-image rotation/flip averaging complements the
original center crop. Scaling and optional PCA are fitted only on real training
images. Each family gets a bounded representation, class-weight and regularization
search. XGBoost stops on macro F1; MLP adds label smoothing, stronger weight decay
and seed comparisons. Only MLP/XGBoost may use a validation-selected ensemble;
the linear baseline remains one logistic regression. No SMOTE is used in this run.

All three heads are finalized before test evaluation. Macro F1 selects models;
macro/weighted F1 and F0.5, per-class metrics and predictions are saved for all
four held-out sets. A validation-selected prior adjustment means the returned
class scores should not be interpreted as calibrated probabilities.

For the controlled SMOTE/weighting ablation, run
`.venv\Scripts\python.exe scripts/ablate_smote.py`. After training and the ablation,
run `.venv\Scripts\python.exe scripts/report_improved.py` to audit identical splits,
recompute historical scores, produce comparison plots and calculate paired
lesion-bootstrap intervals. Repeated tuning on one validation split can overfit;
the historical test sets are not a new prospective external validation.

# Seven calibrated lesion specialists

Run `train-specialists.cmd` or
`.venv\Scripts\python.exe scripts/train_specialists.py` to train seven independent
one-vs-rest classifiers using the existing complete frozen multi-view cache.
No image subset or metadata is added. Final base models use the same 7,014 training
images, and all four historical held-out sets remain separate.

Each lesion compares 11 candidates across three lesion-grouped training folds:
linear classifiers, 256/64-unit neural networks and XGBoost. Outer fold scoring
lesions never enter the corresponding fitting or calibration subsets. Each neural
specialist independently tests label smoothing 0, 0.03 and 0.10, towards that fold's
training prevalence. Linear/XGBoost regularization and calibration are selected
separately; neural label smoothing is not applied to those algorithms.

Final winners are refitted on all training images and sigmoid-calibrated using
base-model predictions made on independent training folds. Per-class binary F1
thresholds come from training OOF scores. A five-value global fusion-strength grid
is selected on internal validation only. Normalized class-selection scores are
different from independently calibrated binary probabilities.

Artifacts are in `outputs/feature_models/efficientnet_b0_specialists/`.
`scripts/report_specialists.py` recomputes all nine historical baselines and reports
accuracy, macro/weighted F1 and F0.5, precision, recall, specificity, ROC AUC,
average precision, confusion counts and calibration losses. Binary specialist
metrics and per-class results after the combined argmax are reported separately.
`scripts/verify_specialists.py` audits artifact preservation, fold separation,
saved metrics, actual-image inference and the Streamlit option.

# Regularized specialist SMOTE comparison

Run `train-regularized-specialists.cmd` to reproduce the paired experiment. It preserves all older experiments and writes two new directories:
`outputs/feature_models/efficientnet_b0_specialists_regularized` and
`outputs/feature_models/efficientnet_b0_specialists_smote`.

Both use the same five candidates per lesion (two regularized linear models, depth-2/depth-3 XGBoost, and a 256/64-unit neural network). Three lesion-separated outer training folds contain an additional lesion-separated stopping partition. XGBoost observes at least 200 rounds, with a maximum of 1,200 and 100-round patience; the neural network observes at least 40 epochs, with a maximum of 180 and 30-epoch patience. The best checkpoint can be earlier than those minimum observation periods. Outer heads are refitted at each selected budget; the final head uses the median budget and all 7,014 fitting images.

SMOTE runs only after fitting the fold scaler, only on fitting images; it grows the minority class to at most three times its size, capped at half the majority count. Both variants recompute square-root class weights after resampling. Stopping, validation and test images retain their observed class distributions.

Selection uses training out-of-fold positive-class F1. Candidates within 0.01 of the best F1 prefer the smaller fitting/out-of-fold gap. Calibration and threshold fitting also use training out-of-fold predictions, so those are selection diagnostics, not an unbiased final estimate. Internal validation chooses one of five predefined fusion strengths. No test metric chooses models or budgets.

The report at `outputs/diagnostics/regularized_specialists/REPORT.md` compares all previous models, both new systems, each specialist's F1/F0.5/precision/recall, and training versus held-out gaps. Training scores use original fitting images, never synthetic vectors. A hash receipt verifies that previous outputs remain unchanged. Previously reported tests are reused comparisons; obtaining a small gap alone does not establish a better model.

# Frozen PanDerm Base embeddings

Run `extract-panderm.cmd` to extract embeddings for every one of the existing 11,720 images, retaining the exact historical split and row order. This downloads only the official pretrained checkpoint and pinned encoder source when absent; no new image dataset is needed. Existing EfficientNet features, classifiers and results remain available.

The adapter in `src/models/panderm.py` loads the official ViT-B/16 encoder source at commit `fd7a80748ba7fc3e203fed88f909f4689d0d6f24`, verifies its SHA256, and strictly checks every encoder tensor before loading. Only the unused classification head is discarded. It avoids importing unrelated upstream training dependencies. The checkpoint is read with `weights_only=True`.

The transform exactly follows the authors' `classification/models/builder.py`: RGB, resize shorter side to 256 with bilinear interpolation, center crop to 224, tensor conversion, mean `(0.485,0.456,0.406)` and std `(0.228,0.224,0.225)`. The `.228` value is intentional to reproduce upstream code. The output is the 768-dimensional final-normalized CLS token returned by `forward_features(..., is_train=False)`, with a frozen eval-mode backbone, no gradient computation and float32 inference/storage. TF32 is disabled for CUDA matrix multiplication and cuDNN convolutions to improve consistency across batch shapes. There is one deterministic view per image.

Extraction supports atomic progress metadata and resumption. Cache identity includes image file statistics, the full split manifest, source and checkpoint SHA256, preprocessing, and library versions. Completion requires finite, nonzero embeddings for all rows and independent single-image checks for seven held-out examples. The receipt at `outputs/diagnostics/panderm_embeddings/extraction_receipt.json` records parameter count, checkpoint/cache sizes, observed GPU allocation, all split counts and hashes. It does not report classifier performance because this stage performs extraction only. A classifier trained later must fit scaling and any resampling on the training partition only.

# Four PanDerm comparisons

## Bounded search around the linear head

`train-panderm-capacity.cmd` compares eight linear-plus-correction settings
(2/2, 8/4, 32/8 and 64/16 hidden units; correction scale .25 or 1) and two
single-hidden GELU heads (32 or 128 units). Results are written separately to
`outputs/feature_models/panderm_large_capacity/`. A residual head first fits the
same train-only standardized logistic control (C=.01, class power .5), freezes
its linear logits, then learns a zero-initialized additive nonlinear correction.
Zero correction is an explicit epoch0 checkpoint; shrinking a replacement
bottleneck is therefore distinguished from adding capacity to a preserved base.

The same three training lesion folds select settings and stopping budgets,
without evaluating new held-out test scores until the selected model is frozen.
Neural fits use GELU, dropout .2, AdamW lr .0007/decay .02, class power .5,
smoothing .03, maximum160 epochs, minimum50 observations and patience40. Final
fitting uses all 8,207 training images and the median best fold epoch. The report
compares the best new candidate with previous matched-fold neural scores and
reports both test sets, rather than claiming that a new candidate necessarily
improves on the best existing model.

`capacity_search.png` plots tuning F1 against parameter count by architecture
family. Paired lesion-bootstrap intervals compare the fixed selected model with
the linear head; they exclude training/search uncertainty. Existing tests are
reused exploratory comparisons. Neither a single smooth peak nor a global
optimum is established by the bounded search. Old models/results remain intact.

## Half, quarter and tiny two-hidden-layer heads

`train-panderm-widths.cmd` adds three heads in
`outputs/feature_models/panderm_large_widths/`: 1024 -> 96 -> 32 -> 7,
1024 -> 48 -> 16 -> 7, and 1024 -> 8 -> 4 -> 7. These are half/quarter widths
relative to the 192/64 model, plus a tiny nonlinear bottleneck head with 8,295
parameters (versus 7,175 for logistic regression). The tiny head is not a strict
superset of the linear function family despite its slightly greater parameter
count. Each uses LayerNorm/ReLU after hidden layers and softmax for inference.

The exact same three training lesion folds and frozen Large feature cache from
the 192/64 experiment are reused. Three settings per architecture compare class
weighting and modest dropout/weight decay. Half/quarter use the 192/64 search
settings; tiny uses lower dropout (0 or .1) and decay (.005/.01) to avoid excessive
regularization of very few units. All use fixed AdamW lr .0007, batch128, smoothing
.03, maximum180 epochs, minimum50 observations and patience40. Pooled training
OOF macro F1 selects settings; the median best fold epoch sets the final budget
for fitting all 8,207 training rows. Validation and tests do not choose settings.
Fold scoring used for stopping is tuning evidence, not unbiased evaluation.

The report compares all nine PanDerm Large variants on the unchanged released
HAM and official splits, with macro/weighted F1/F0.5, accuracy and train/held-out
gaps. It checks old artifact hashes, saved OOF/final predictions, actual image
inference, selected architecture dimensions and training-only scalers. All old
models remain available in the app alongside the three new labelled options.

## Additional training-fold tuned models

`train-panderm-refined.cmd` adds one XGBoost and one smaller neural network in
`outputs/feature_models/panderm_large_refined/`, preserving all previous model
artifacts with a SHA256 receipt. Both reuse the exact completed PanDerm Large
cache and released HAM split. Three stratified, lesion-separated folds inside
the 8,207 training images choose settings and stopping budgets; each fold fits
its own scaler and optional PCA. The released validation set and both tests do
not choose the new settings. Fold scores used for early stopping are tuning
diagnostics, not unbiased cross-validation estimates.

The bounded search compares 26 tree configurations (depths 1/2/3/4/6, raw,
PCA128/PCA256 or supervised LDA6 features, unweighted/square-root weighting,
lower L2=5 and no L1) and three smaller-network configurations. LDA uses a
shrinkage estimate and only fitting-fold labels, mapping 1,024 dimensions to six
class-discriminative directions before depth 1/2/3 trees. These six configurations
were added before test evaluation; the initial protocol is preserved. This is
supervised preprocessing, explicitly distinguished from raw-embedding XGBoost.
XGBoost allows 2,000 rounds, at least
400 observations and patience 200 on macro F1. The NN uses 1024 -> 192 -> 64 -> 7
instead of 1024 -> 256 -> 64 -> 7, with modest dropout/weight decay, at least 50
of up to 180 epochs and patience 40. Final models fit all training rows at the
median best fold budget. The scripts save complete fold histories, OOF scores,
per-image predictions and train/validation/test statistics for comparison.
`scripts/check_panderm_linear_cv.py` optionally measures the existing linear
configuration on these same training folds as a diagnostic.

The Base comparison below is a historical custom experiment. For the requested
published backbone, use `train-panderm-large.cmd`. It uses the official PanDerm
Large checkpoint and the completed 11,720 x 1,024 feature cache. Four classifiers
share the authors' released HAM split (8,207 train / 575 validation / 1,232 test):
the unchanged authors' linear-probe implementation, a custom L2 linear head,
a 1,024 -> 256 -> 64 -> 7 neural network, and XGBoost. The previously completed
author probe is verified and reused; custom heads are fitted on all released
training rows. One unrecognized `trainisic` row is excluded exactly as upstream.
This released split differs from the manuscript's stated 60/20/20 proportions.

The common official test set (1,512 images) is reported separately, allowing
comparison with historical runs that used a different internal split. The Large
experiment saves everything in `outputs/feature_models/panderm_large_comparison/`;
it preserves every older classifier/result. Validation macro F1 selects custom
settings; XGBoost explicitly disables its default metric so early stopping uses
negative macro F1, with a minimum 200 observed rounds and patience 100. The paper
probe retains the authors' raw features, fixed C=71.68 and LBFGS recipe. All four
use a frozen encoder and the authors' exact image transform, without metadata or
SMOTE. F1 and F0.5 are reported with both macro and weighted averaging. Run
`scripts/report_panderm_large.py` to recompute predictions, verify train-only
scaling and artifact preservation, and check actual-image inference through all
four saved classifiers.

Run `train-panderm.cmd` to train and verify the frozen-backbone comparison in `outputs/feature_models/panderm_base_comparison/`. Every method uses the completed 768-dimensional PanDerm Base cache and the original 7,014 training images, 1,509 validation images, 1,492 internal test images, 193 official validation images and 1,512 official test images.

The bare PanDerm Base checkpoint contains only the encoder and cannot output seven lesion classes. `plain_panderm.json` records that limitation without assigning invented classification metrics. A fourth measurable baseline uses fixed cosine 1-nearest-neighbor retrieval from training embeddings and labels, with no learned parameters. This is a separately labelled reference-based readout, not bare or zero-shot PanDerm. Its training diagnostic excludes every image sharing the query's lesion ID; its hard votes are not calibrated probabilities.

The supervised alternatives are an L2 multinomial logistic-regression head, a 768→256→64→7 neural network (214,407 parameters), and XGBoost. A train-only StandardScaler travels inside each supervised checkpoint. Validation macro F1 chooses the declared settings and best checkpoints; all heads are finalized before test reporting. Class weighting is trained on original training labels; this comparison uses no SMOTE, additional image views, metadata fusion or multi-head ensemble. The backbone remains frozen throughout.

The neural network uses moderate dropout/weight decay, up to 180 epochs, at least 40 observed epochs, and 30-epoch patience. XGBoost observes at least 200 of a possible 1,200 rounds with 100-round patience; both restore the best checkpoint, which can be earlier than the minimum observation period. The report includes all historical model comparisons, per-class F1/F0.5/precision/recall, training gaps, paired lesion bootstrap, real-image inference verification and hashes proving earlier artifacts remain unchanged. Previously inspected test results are comparisons rather than a fresh blind benchmark.
