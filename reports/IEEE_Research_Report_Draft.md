# Privacy-Preserving Skin Lesion Classification Using Federated Learning and Dermatology Foundation Models

**Author(s):** [Insert names]  
**Affiliation:** [Insert institution and department]  
**Email:** [Insert email]  
**Status:** Draft research report

**Abstract—** This report presents a research prototype for seven-class dermoscopic image classification, comprising a simulated federated-learning framework and separate experiments with PanDerm foundation-model representations. The federated framework supports a centralized convolutional baseline, non-identically distributed client partitions, federated averaging, optional FedProx regularization, and optional differentially private training. In a PanDerm joint refinement experiment, a Large neural head achieved 93.99% accuracy and 87.38% macro F1 on 1,232 HAM images and 85.91% accuracy and 79.02% macro F1 on 1,512 ISIC images. Both cohorts were used for adaptation and checkpoint selection; these are fitting results and do not establish generalization. Independent evaluation and a documented privacy analysis remain necessary.

**Index Terms—** skin lesion classification, federated learning, differential privacy, FedProx, PanDerm, dermoscopy.

## I. INTRODUCTION

Skin-lesion classification is challenging because lesion categories vary in appearance and occur at unequal frequencies. Machine learning can benefit from diverse image datasets, but sharing patient images across institutions may be constrained by privacy and governance requirements. Federated learning (FL) trains a shared model through local computation and aggregation without routinely pooling raw images [1].

This project investigates a simulated privacy-aware federated training pipeline alongside classifiers built on pretrained PanDerm features. Its contributions are a centralized and federated classification workflow, optional differential privacy (DP) and FedProx mechanisms, experiments with PanDerm classifier heads, and a dashboard for examining saved models. The PanDerm experiments are separate from the federated pipeline and do not demonstrate federated or differentially private PanDerm training.

The study is organized around three questions: how well can available image representations support seven-class classification; how can a simulated federated workflow accommodate heterogeneous client data and privacy-aware training; and what evidence is still needed to assess generalization? The present report describes the implementation and completed exploratory results. It does not assert that federated learning improves accuracy or that the system is ready for clinical use.

## II. BACKGROUND AND RELATED WORK

HAM10000 contains 10,015 dermoscopic images across seven diagnostic categories [2]. Federated averaging (FedAvg) aggregates locally trained models to learn from decentralized data [1]. FedProx adds a proximal term to address heterogeneous client optimization [3]. PanDerm is a dermatology foundation model pretrained on multimodal dermatological images [4]. Differences in datasets, splits, preprocessing, and metrics prevent direct comparison between this project's results and published scores.

EfficientNet provides a family of convolutional networks whose scaling approach balances depth, width, and input resolution [5]. Differentially private stochastic gradient methods use bounded per-example contributions and random noise to support quantifiable privacy accounting under a specified mechanism and assumptions [6]. The ISIC 2018 challenge provides a related lesion-diagnosis task and image cohorts [7]. Neither a privacy parameter nor a reported classification score can be interpreted independently of the associated training and evaluation protocol.

## III. MATERIALS AND METHODS

### A. Dataset and Preprocessing

The primary dataset is HAM10000 [2], with seven labels: actinic keratoses and intraepithelial carcinoma (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv), and vascular lesions (vasc). The default input is 224 × 224 pixels with ImageNet normalization. Training augmentation is configurable; validation and test transformations are deterministic. Current real-image CNN and federated pipelines require lesion identifiers and split by lesion to keep related images together. Synthetic data are available for smoke tests.

The PanDerm refinement uses 1,232 HAM and 1,512 ISIC images from two former test cohorts. They were subsequently used for fitting and selection and are not independent test sets for the adapted models. A 575-image HAM validation cohort has further overlap and repeated-inspection limitations discussed below.

For the separate EfficientNet feature-model workflow, the 10,015-image HAM collection was grouped into 7,470 lesions and split with seed 42 into 7,014 training, 1,509 validation, and 1,492 test images. The official ISIC validation and test cohorts contain 193 and 1,512 images respectively. Official-set lesion identifiers are unavailable in the project records, so independence across releases cannot be assumed. These split sizes refer to that workflow and should not be substituted for the PanDerm cohort sizes.

### B. Federated Classification Framework

The default CNN uses an ImageNet-pretrained EfficientNet-B0 backbone with a frozen feature extractor and a trainable classification head; other backbones are supported. The default configuration specifies five simulated clients, 20 rounds, five local epochs, and a Dirichlet concentration parameter of 0.5 for non-IID partitioning. These are settings, not evidence of a completed run.

FedAvg aggregates client models after local training [1]. When FedProx is selected, a proximal penalty is added to the local loss:

**Lᵢ(w) = Fᵢ(w) + (μ/2)‖w − wᵗ‖²,**

where Fᵢ is the local objective, wᵗ is the global model at the start of the round, and μ controls the penalty [3]. The default configuration selects FedAvg; μ = 0.01 applies only when FedProx is enabled.

Conceptually, the server distributes the current parameters, each participating client trains on its local partition, and the server aggregates the returned updates. Dirichlet partitioning makes class proportions differ across simulated clients; lower concentration values produce greater heterogeneity. The simulation is useful for controlled experiments, but partitioning one dataset does not reproduce the full acquisition, population, and operational differences between hospitals.

### C. Differential Privacy

Optional DP training uses per-example gradient clipping and noise through Opacus. The configuration specifies target ε = 8, δ = 10⁻⁵, and maximum gradient norm 1 with automatic noise calibration. Per-client privacy accounting accumulates across local training rounds. These are configured targets, not measured guarantees for a particular run. Accounting covers training-image updates; validation and test statistics are separate, and patient-level privacy is not established. Federated training alone does not guarantee privacy.

For a defensible privacy claim, a future run must record the actual noise multiplier, sampling and participation assumptions, training steps, accountant configuration, resulting per-client ε and δ, and the adjacency unit. In particular, multiple images may correspond to one lesion or patient; an image-level accounting result should not be described as a patient-level guarantee. Communication protection, access control, and the treatment of evaluation outputs also need separate assessment.

### D. Foundation-Model Experiments and Metrics

PanDerm experiments use pretrained Base and Large representations with linear, neural, and XGBoost-based classifier heads. Joint refinement fit the two former test cohorts with equal total loss weight per cohort. Four Large heads were attempted for up to eight updates. Checkpoint selection maximized the smaller cohort gain over an unchanged local authors' recipe using the mean of accuracy, macro F1, macro F0.5, weighted F1, and balanced accuracy. The reference is a local reproduction of the recipe, not the manuscript classifier checkpoint.

In this refinement, neural and linear heads used cross-entropy with label smoothing of 0.01, AdamW learning rate 0.00014 and weight decay 0.01; class weights used original training frequencies to power 0.25. Each neural or linear update accumulated gradients across the full combined data. XGBoost instead appended boosting rounds at learning rate approximately 0.00444, so an update is not equivalent across these model families. The backbone and feature projections were preserved. The selection score gives equal numerical weight to five correlated metrics and reflects a chosen objective, not a universal measure of clinical utility.

The separate EfficientNet feature pipeline extracts 1,280-dimensional vectors with the pretrained backbone in evaluation mode, fits a standard scaler on training features only, and trains linear, multilayer-perceptron, and XGBoost classifiers. Its validation macro F1 guides model selection. These centralized feature models and the fine-tuned CNN baseline are distinct experiments; their trained weights should not be conflated with the federated configuration.

Accuracy measures correct predictions; macro F1 averages classwise F1 equally, while weighted F1 weights classes by support. Balanced accuracy averages classwise recall, and macro F0.5 emphasizes precision relative to recall. Multiple metrics help expose class imbalance but do not establish clinical suitability.

For class k, precision is TPₖ/(TPₖ + FPₖ), recall is TPₖ/(TPₖ + FNₖ), and F1 is 2 precision × recall/(precision + recall), with appropriate handling of zero denominators. Macro averaging gives each class equal weight, whereas weighted averaging emphasizes common classes. Because melanoma and other categories have different clinical consequences, a single aggregate metric cannot replace class-specific sensitivity, predictive values, and evaluation of misclassification patterns.

## IV. EXPERIMENTAL RESULTS

Table I reproduces selected joint-refinement results. Both cohorts were used for adaptation and checkpoint selection. All values are percentages and are **fitting metrics**, not independent test estimates.

**TABLE I. SELECTED PANDERM RESULTS ON FITTING COHORTS**

| Model | HAM accuracy | HAM macro F1 | ISIC accuracy | ISIC macro F1 |
|---|---:|---:|---:|---:|
| Large, neural head 192→64 | 93.99 | 87.38 | 85.91 | 79.02 |
| Large, neural head 128 | 93.26 | 85.75 | 85.71 | 78.60 |
| Large, linear head | 92.53 | 81.74 | 84.33 | 76.19 |
| Large, XGBoost + LDA | 91.48 | 79.18 | 82.21 | 73.08 |
| Authors' recipe, unchanged local reference | 92.37 | 80.98 | 82.94 | 71.58 |

The 192→64 neural head achieved the highest accuracy and macro F1 among the listed Large heads on both fitting cohorts. The comparison does not establish held-out superiority: the adapted heads used both cohorts during fitting, whereas the reference was unchanged. The gap between accuracy and macro F1 motivates examination of minority classes.

### A. Centralized CNN Baseline

A separate centrally fine-tuned EfficientNet-B0 run completed 15 epochs and selected epoch 11 using internal validation macro F1. On 1,509 internal validation images it recorded 85.62% accuracy, 77.51% balanced accuracy, and 76.53% macro F1. On 1,492 internal test images the corresponding values were 84.32%, 76.17%, and 74.52%. On the 1,512-image official test cohort they were 81.61%, 72.30%, and 71.78%. These test sets had been used in earlier experiments and are not fresh blind benchmarks; cross-release lesion independence is unverified. The fine-tuned baseline is a separate model and protocol, so these figures should not be treated as a controlled comparison against the adapted PanDerm heads.

### B. Interpretation of the Refinement

The 192→64 head's HAM and ISIC fitting macro F1 exceed the local reference by 6.40 and 7.44 percentage points, respectively. Those differences describe fit to the same cohorts used for adaptation and selection; they are not estimates of prospective improvement. The joint report also finds that the three Large linear and neural heads exceed the reference on the chosen five-metric mean in both fitting cohorts, while the Large XGBoost head does not meet that condition on HAM. The project records report no confidence intervals or statistical tests for these differences.

The 575-image HAM validation results for the Large 192→64 head were 92.87% accuracy and 79.30% macro F1 after refinement. Although excluded from that additional fitting and checkpoint selection, this cohort was examined repeatedly in earlier experiments. Its results are exploratory and should not be represented as a pristine final test.

This draft does not claim a numerical federated-versus-centralized improvement or measured DP budget without a clearly identified comparable completed run. Future results should identify run settings, cohorts, privacy-accountant outputs, and uncertainty estimates.

## V. DISCUSSION AND LIMITATIONS

Both former test cohorts contributed to adaptation and checkpoint selection. The 575-image HAM validation set was excluded from additional joint fitting and selection, but its scores were inspected in earlier experiments; Base training overlaps 395 of those images. Cross-release lesion and patient independence between ISIC and HAM has not been established. Current measurements therefore cannot establish independent generalization or clinical effectiveness.

Clients are simulated partitions, not verified independent hospitals. The project has not established patient-level privacy or security against all attacks on model updates. Class imbalance, representativeness, calibration, skin-tone variation, acquisition conditions, and deployment robustness require further study. The image interface produces seven-category scores but does not validate lesion presence. This is an academic prototype, not a clinical diagnostic tool.

Future work should reserve an untouched patient- or lesion-disjoint test set, audit cross-dataset overlap, preregister model selection, and compare centralized, FedAvg, FedProx, and DP variants under matched conditions. Report per-class sensitivity and precision, confidence intervals, calibration, measured privacy parameters, and subgroup performance where reliable metadata are available. External multi-institutional and prospective evaluation would be needed before clinical claims.

An improved evaluation protocol should freeze preprocessing, partition manifests, model-selection criteria, and privacy settings before running a final test. Patient or lesion grouping should be verified across all releases using available identifiers and duplicate checks, and any unresolved overlap disclosed. Compare models using the same images and report confidence intervals using an appropriate clustered resampling unit. Assess calibration and clinically relevant error patterns separately from the chosen composite metric. Any external validation should describe population, device, setting, prevalence, and missing metadata.

## VI. REPRODUCIBILITY AND IMPLEMENTATION

The repository records the default parameters in config.yaml and provides scripts for centralized training, federated simulation, evaluation, and the Streamlit dashboard. The EfficientNet baseline report, joint-refinement protocol, selected checkpoints, metrics, and prediction receipts are retained under outputs. Reproduction requires the corresponding datasets, pretrained model weights, environment, and exact artifact versions; configuration alone does not reproduce an archived result. The project's training guide documents split manifests and feature extraction. Record software versions, random seeds, hardware, checkpoint hashes, and dataset access terms when preparing a final submission.

The dashboard provides research exploration and inference over saved seven-class predictors. Scores are not established as calibrated probabilities, and the interface does not verify that a submitted image depicts a lesion. Deployment safeguards, clinical workflow integration, and prospective validation are outside the present evidence.

## VII. CONCLUSION

This project implements a simulated federated framework and a separate PanDerm feature-based workflow for dermoscopic classification. Adaptation results show that several heads fit the studied cohorts well, but reuse of those cohorts prevents reliable claims about unseen-image performance. Independent evaluation and documented privacy analysis are the next steps.

## REFERENCES

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in *Proc. 20th Int. Conf. Artificial Intelligence and Statistics*, 2017, pp. 1273–1282. [Online]. Available: https://proceedings.mlr.press/v54/mcmahan17a.html

[2] P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” *Scientific Data*, vol. 5, art. no. 180161, 2018, doi: 10.1038/sdata.2018.161.

[3] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in *Proc. Machine Learning and Systems*, vol. 2, 2020. [Online]. Available: https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html

[4] S. Yan *et al.*, “A multimodal vision foundation model for clinical dermatology,” *Nature Medicine*, 2025, doi: 10.1038/s41591-025-03747-y.

[5] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in *Proc. 36th Int. Conf. Machine Learning*, 2019, pp. 6105–6114. [Online]. Available: https://proceedings.mlr.press/v97/tan19a.html

[6] M. Abadi *et al.*, “Deep learning with differential privacy,” in *Proc. ACM SIGSAC Conf. Computer and Communications Security*, 2016, pp. 308–318, doi: 10.1145/2976749.2978318.

[7] International Skin Imaging Collaboration, “ISIC 2018 challenge: Task 3, lesion diagnosis.” [Online]. Available: https://challenge.isic-archive.com/landing/2018/47/

---
*Drafting note: Add author details, experiment dates, and acknowledgments as applicable. Apply the appropriate IEEE conference or journal template and required two-column layout before submission. Verify results against archived experiment receipts.*
