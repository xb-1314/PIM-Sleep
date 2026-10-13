# PIM-Sleep: A Physiology-Informed Sleep Foundation Model for Tasks Across Multiple Timescales

This repository contains the research implementation of **PIM-Sleep**, a multimodal sleep representation model guided by morphological events. The framework learns shared EEG and EOG representations and transfers them to sleep staging, whole-night sleep disorder classification, and sample-level sleep spindle detection.

**Authors:** Bo Xu, Song Zhang, Leyang Pan, Haokai Zhang, Dongrui Gao, Jiaxin Xie, and Shaofei Ying.

## Introduction

PIM-Sleep uses six physiological event types as pretraining targets: alpha activity, slow waves, sleep spindles, rapid eye movements, vertex sharp waves, and K-complexes. Modality-specific convolutional encoders map one EEG channel and bilateral EOG channels into a fused token for each 30-second epoch. A shared Transformer models consecutive epoch tokens.

Pretraining combines full-event prediction with BERT-style masked-event prediction. The full-event objective updates the entire backbone. In the masked branch, epoch embeddings are detached before corruption, so this objective trains contextual inference without updating the signal encoders or fusion module.

The main workflow is:

```text
PSG recordings
    -> resampling, channel selection, and morphological event labeling
    -> EEG/EOG epoch encoding and contextual pretraining
    -> task-specific transfer
         |-- sleep staging: contextual tokens + temporal convolutional head
         |-- sleep disorder classification: stage-wise features + sleep architecture
         `-- spindle detection: local EEG features + same-frame multimodal features
```

## Highlights

- **Physiology-informed supervision:** six morphological event types provide interpretable pretraining targets.
- **Multimodal epoch encoding:** an independent EEG encoder and a shared bilateral EOG encoder support missing-channel gating.
- **Contextual representation learning:** 20 consecutive epochs provide a 10-minute context window during pretraining and sleep staging.
- **Joint event learning:** full-event prediction and detached masked-event prediction share the contextual backbone and use separate event heads.
- **Transfer across timescales:** downstream outputs cover 30-second stages, whole-night disorder classes, and sample-level spindle probabilities.

## Repository Structure

```text
PIM-Sleep/
├── README.md
├── requirements.txt
├── morphology_detector_yasa.py
├── 01sleeppreprocessing_morphological_yasa_two_multimodal_eog.py
├── 01sleeppreprocessing_morphological_yasa_isruc_two_multimodal_eog.py
├── 01sleeppreprocessing_morphological_yasa_dreams_healthy_two_multimodal_eog.py
├── 01sleeppreprocessing_morphological_yasa_clas_two_multimodal.py
├── 01sleeppreprocessing_morphological_yasa_cap_two_multimodal.py
├── 1sleeppreprocessing_morphological_yasa_bp_slp_C3_multimodal.py
├── 03end.py
└── 04predictor_morphological_min_loso_eeg_eoglonezuobi.py
```

| File | Purpose |
|---|---|
| `morphology_detector_yasa.py` | YASA-based spindle, slow-wave, and REM detection; rule-based alpha, vertex, and K-complex detection; supporting waveform features |
| `01sleeppreprocessing_morphological_yasa_two_multimodal_eog.py` | MASS-SS3 preprocessing |
| `01sleeppreprocessing_morphological_yasa_isruc_two_multimodal_eog.py` | ISRUC preprocessing |
| `01sleeppreprocessing_morphological_yasa_dreams_healthy_two_multimodal_eog.py` | DREAMS healthy-subject preprocessing |
| `01sleeppreprocessing_morphological_yasa_clas_two_multimodal.py` | CUITS preprocessing; the code retains the internal dataset name `CLAS` |
| `01sleeppreprocessing_morphological_yasa_cap_two_multimodal.py` | CAP preprocessing and subject-level disorder labels |
| `1sleeppreprocessing_morphological_yasa_bp_slp_C3_multimodal.py` | BP-SleepX preprocessing; the code retains the internal dataset name `BP_SLP` |
| `03end.py` | Joint full-event and detached masked-event pretraining |
| `04predictor_morphological_min_loso_eeg_eoglonezuobi.py` | Downstream model definitions, training, evaluation, and visualization |

## Getting Started

### Environment

Use **Python 3.11** and a dedicated virtual environment. A CUDA-capable GPU is recommended for training. The scripts select CUDA when available and otherwise use CPU.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, select the PyTorch build appropriate for your driver and platform using the [official installation guide](https://pytorch.org/get-started/locally/). The project uses the `torch.amp` API and requires PyTorch 2.4 or later.

YASA is pinned to **0.7.0** because the detector uses `Hypnogram` inputs and string stage filters, while the preprocessing scripts also call `SleepStaging.predict_proba()`. Both interfaces are available in 0.7.0; the probability method is removed in 0.8. See the [YASA staging implementation](https://github.com/raphaelvallat/yasa/blob/v0.7.0/src/yasa/staging.py). Keep `morphology_detector_yasa.py` in the repository root so the preprocessing scripts can import it.

The dependency file specifies installation constraints. It does not represent an archived environment from the manuscript experiments. Dependency resolution has been checked for CPython 3.11 on Windows x86-64. This verifies package metadata compatibility; complete experiment reproduction additionally requires the data, checkpoints, and matching run configurations.

### Datasets

| Dataset | Role in the manuscript | Access |
|---|---|---|
| MASS-SS3 | Pretraining | [MASS-SS3 dataset record](https://doi.org/10.5683/SP3/9MYUCS); request access through the data provider |
| ISRUC-S1 | Pretraining | [ISRUC-SLEEP](https://sleeptight.isr.uc.pt/) |
| DREAMS | Sleep staging, spindle detection, and healthy controls for disorder classification | [DREAMS databases](https://zenodo.org/records/2650142) |
| CAP | HC/NFLE/RBD disorder classification | [CAP Sleep Database](https://archive.physionet.org/pn6/capslpdb/) |
| CUITS | Sleep staging and spindle detection | Contact the study authors for availability and access conditions |
| BP-SleepX | Sleep staging and spindle detection | Contact the study authors for availability and access conditions |

Acquire recordings and reference annotations from their providers and follow the applicable dataset terms. Raw recordings, private datasets, and pretrained checkpoints are not included in the supplied source files.

## Data Preparation

### Signal and Label Conventions

Signals are resampled to **100 Hz** and segmented into **30-second epochs**, giving **3,000 samples per channel**. The model input uses one EEG channel, left EOG, and right EOG. Missing EOG channels are zero-filled and accompanied by a channel-availability vector.

Sleep-stage labels use:

```text
0 = W, 1 = N1, 2 = N2, 3 = N3, 4 = REM
```

For morphological supervision, YASA-generated stages constrain event detection. The continuous-signal detector uses N1/N2/N3 for spindles, N2/N3 for slow waves, and REM for rapid eye movements. Alpha detection is restricted to W/N1, K-complex detection to N2, and vertex detection to N1. Auxiliary theta and slow-eye-movement features may also be saved by preprocessing; the pretraining target vector contains the six events listed above.

The preprocessing scripts save per-subject files named `<subject_id>_morphology.npz`. Core fields are:

| Field | Shape / contents |
|---|---|
| `subject_id` | Subject identifier |
| `eeg_epochs` | `[N, 3000]` EEG signal |
| `eog_loc_epochs` | `[N, 3000]` left EOG signal |
| `eog_roc_epochs` | `[N, 3000]` right EOG signal |
| `eog_available` | `[2]` availability indicators for left/right EOG |
| `sampling_rate`, `epoch_duration` | `100` Hz and `30` seconds |
| `waveform_features` | Per-epoch feature dictionaries with `has_alpha`, `has_slowwave`, `has_spindle`, `has_rem`, `has_vertex`, and `has_kcomplex` |
| `true_sleep_stages` | Reference stage labels, `[N]` |
| `yasa_sleep_stages` | Automatically estimated stages, `[N]` |
| `sleep_disorder_label` | Subject-level class when applicable: HC=0, NFLE=1, RBD=2 |

The pretraining loader excludes negative event targets, including its `-100` ignore value, from the loss. Preprocessing also generates text descriptions for compatibility with earlier workflows; the main backbone takes signal tensors.

### Run Preprocessing

Edit `data_path` and `output_path` in each script's `__main__` block. For BP-SleepX, edit `PreprocessConfig.dataset_root` and select `label_technician` from `T1`, `T2`, or `T3`; the supplied default is `T3`.

Run the preprocessing scripts for the datasets used in your experiment. For pretraining:

```bash
python 01sleeppreprocessing_morphological_yasa_two_multimodal_eog.py
python 01sleeppreprocessing_morphological_yasa_isruc_two_multimodal_eog.py
```

For DREAMS staging and healthy-control features:

```bash
python 01sleeppreprocessing_morphological_yasa_dreams_healthy_two_multimodal_eog.py
```

Keep the reference annotation files in the layout expected by the dataset-specific readers. For example, the DREAMS reader looks for `HypnogramAASM_<subject>.txt` beside the EDF recording.

## Pretraining

Configure `EEGPretrainConfig` in `03end.py`. Replace the machine-specific paths with your processed data locations, for example:

```python
self.pretrain_dirs = [
    "./data/MASS_SS3/processed/",
    "./data/ISRUC_S1/processed/",
]
self.output_dir = "./sleep_eeg_pretrained_backbone_event_detached_bert_context"
```

Then run from the repository root:

```bash
python 03end.py
```

| Setting | Default |
|---|---:|
| Epochs per window / window stride | 20 / 10 |
| Hidden dimension | 256 |
| Transformer layers / attention heads | 2 / 8 |
| Training epochs / batch size | 30 / 8 |
| Optimizer | AdamW |
| Learning rate / weight decay | 8e-5 / 1e-4 |
| Mask target probability | 0.10 |
| Mask / keep / random replacement | 0.80 / 0.10 / 0.10 |
| Full-event / masked-event loss weights | 1.0 / 0.05 |
| Masked token embeddings detached | True |

Training and validation are split by subject, with approximately 10% of subjects reserved for validation. The default checkpoint criterion is **full-event validation BCE**. Validation masking uses a fixed seed.

The main checkpoint is:

```text
sleep_eeg_pretrained_backbone_event_detached_bert_context/
    sleep_eeg_bert_backbone_pretrained.pt
```

The script also writes full-model checkpoints, periodic backbone checkpoints, training history, and dataset metadata. Set `FinetuneConfig.eeg_pretrained_model_path` to the backbone checkpoint before downstream evaluation.

## Downstream Tasks

All main task settings are in `FinetuneConfig` in `04predictor_morphological_min_loso_eeg_eoglonezuobi.py`. The current scripts use configuration classes rather than command-line task arguments. Enable the desired task and disable the other `run_*` flags, then run:

```bash
python 04predictor_morphological_min_loso_eeg_eoglonezuobi.py
```

The supplied defaults enable **CUITS/CLAS spindle detection**. Update data paths and output directories before launching a task.

### Sleep Staging

Example configuration:

```python
self.run_sleep_stage_classification = True
self.run_sleep_disorder_classification = False
self.run_spindle_detection = False
self.sleep_stage_run_mode = "train"
self.stage_train_dirs = ["./data/DREAMS/processed/"]
self.output_root = "./results"
self.stage_output_dir = "./results/staging/DREAMS"
```

Sleep staging uses a three-block TCN head with dilation rates 1, 2, and 4. Subject-level LOSO is used: one subject is held out, and the remaining subjects are trained for 100 fixed epochs. The first five epochs train the head with the backbone frozen; the final Transformer layer is then unfrozen. A training-loss plateau can trigger unfreezing of the penultimate Transformer layer. The fixed final-epoch model is evaluated.

Overlapping window logits are averaged by original epoch index. The current default decoder is **confidence-gated Viterbi**, with training-derived stage transitions and inference-time TCN Dropout. The fixed settings are:

| Setting | Default |
|---|---:|
| TCN inference Dropout | 0.02 |
| Inference seed | 2056622951 |
| Viterbi transition weight | 1.32 |
| Confidence gate / probability-margin gate | 0.85 / 0.40 |
| Minimum sequence-score gain | -0.40 |

These settings are part of the evaluation configuration. For deterministic raw predictions after window aggregation, set `saved_stage_primary_method="raw"` and `saved_stage_mc_dropout_scope="none"`. The saved-checkpoint evaluation path uses `sleep_stage_run_mode="saved_repro"` and `saved_stage_checkpoint_dir`.

### Sleep Disorder Classification

Example configuration:

```python
self.run_sleep_stage_classification = False
self.run_sleep_disorder_classification = True
self.run_spindle_detection = False
self.disorder_train_dirs = ["./data/CAP/processed/"]
self.disorder_include_dreams_hc = True
self.disorder_dreams_hc_dirs = ["./data/DREAMS/processed/"]
self.output_root = "./results"
self.disorder_output_dir = "./results/disorder"
```

The cohort combines CAP HC/NFLE/RBD recordings with the 20 DREAMS healthy subjects. Whole-night features are extracted with a frozen pretrained backbone. Overlapping contextual outputs are stitched into one representation per original epoch.

This task uses **reference sleep-stage annotations** to construct stage-wise features and sleep architecture. It therefore evaluates classification from PSG representations together with reference staging.

The implemented classifier has two branches:

1. **Stage-wise branch:** epoch features are normalized within each epoch, then grouped into W/N1/N2/N3/REM. Each stage vector contains its mean, population standard deviation, and all five stage proportions. A `StandardScaler -> PCA -> RandomForestClassifier` pipeline is fitted for each available stage, and stage probabilities are averaged.
2. **Sleep-architecture branch:** stage proportions, transitions, bout statistics, and sleep-continuity features are classified by class-balanced logistic regression.

Outer four-fold subject-level CV evaluates the full procedure. Inner three-fold CV first selects the architecture classifier's regularization, then selects RF settings and the architecture fusion weight. Inner out-of-fold predictions also select HC/RBD decision multipliers. ACC, F1, and confusion matrices use the adjusted class decisions; AUROC uses the original fused probabilities.

The current ten repeat seeds are:

```text
35, 36, 37, 38, 39, 40, 41, 42, 43, 44
```

Each repeat produces complete outer out-of-fold predictions. Summary metrics are the mean and population standard deviation across the ten complete repeats. The mixed CAP/DREAMS cohort should be considered when interpreting generalization across recording sources.

### Sleep Spindle Detection

Example path configuration for the supplied CUITS/CLAS entry point:

```python
self.run_sleep_stage_classification = False
self.run_sleep_disorder_classification = False
self.run_spindle_detection = True
self.spindle_train_dirs = ["./data/CUITS/processed/"]
self.spindle_annotation_dir = "./data/CUITS/spindle_annotations/"
self.output_root = "./results"
self.spindle_output_root = "./results/spindles/CUITS"
```

The current loader requires **35 matched subjects**, C3-A1 signals, and the following filenames:

```text
Base_SLP001_morphology.npz
Base_SLP001_Spindle.txt
...
```

Each annotation file contains onset and duration in seconds; its optional channel header is `[Spindle/C3-A1]`. Training and scoring use valid N2 samples. The model processes 3-second frames with 80% overlap, combines local EEG features with the same frame's multimodal representation, and upsamples logits to sample resolution. The retained 20-position embedding table permits checkpoint loading; each actual detection input contains one frame.

The loss is weighted BCE plus 0.5 times Dice loss. The positive weight is the square root of the negative/positive ratio in unique valid training samples, clipped to [1, 10]. Frame probabilities are averaged at their original sample positions. Event extraction uses 0.1-second smoothing, high/low hysteresis, durations of 0.5–3.0 seconds, and merging of gaps strictly below 0.1 seconds.

The supplied default configuration uses:

- subject-level LOSO;
- two training epochs and a fixed final-epoch checkpoint;
- probability threshold 0.6;
- primary event-matching IoU 0.6, reporting only IoU 0.6.

To report the manuscript's primary metric and multi-IoU curve, configure:

```python
self.spindle_primary_iou_threshold = 0.3
self.spindle_report_iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
```

The probability threshold and matching IoU threshold control different stages of evaluation. Training beyond five epochs activates the implementation's Transformer-unfreezing schedule. To keep the backbone frozen for an entire configured run, set `spindle_freeze_backbone_epochs` equal to `spindle_epochs` and disable `spindle_enable_plateau_second_transformer_stage`.

The current `main()` routes spindle detection to the CUITS/CLAS loader and subject LOSO. Other loaders and random-split helpers are present, but reproducing the DREAMS/BP-SleepX spindle experiments and the manuscript's random-split comparison requires the corresponding entry-point and dataset configuration.

## Results Reported in the Manuscript

The following values are reported in the accompanying manuscript. They are separate from results generated by the current default script settings.

| Dataset | Sleep staging accuracy (%) | Spindle event F1 at IoU 0.3 (%) |
|---|---:|---:|
| DREAMS | 86.95 | 78.87 |
| CUITS | 88.31 | 85.06 |
| BP-SleepX | 89.18 | 78.77 |

For CAP plus DREAMS HC disorder classification, the manuscript reports accuracy **84.29 ± 1.78%**, macro-AUROC **93.90 ± 0.47%**, and macro-F1 **84.75 ± 1.83%** across ten repeated four-fold experiments.

## Reproducibility

For each experiment, retain the subject split, input data version, annotation source, checkpoint configuration, dependency versions, and decoder/postprocessing settings. Use the generated manifests and JSON summaries to associate reported metrics with the actual run.

The checked-in scripts contain machine-specific paths and development defaults. In particular, the current spindle epoch count and IoU settings differ from the manuscript evaluation, and the current disorder seed list is the list shown above. The manuscript results require their corresponding experiment configurations and checkpoints.

The downstream file also contains optional generation augmentation, external generalization, slow-wave detection, spectral branches, fold-specific adaptation, and test-time adaptation. These are controlled separately from the three main task paths described here.

## Citation

If you use this implementation, please cite the PIM-Sleep manuscript. The entry below records the supplied manuscript; published bibliographic details can be added when available.

```bibtex
@unpublished{xu_pim_sleep,
  title  = {{PIM-Sleep}: A Physiology-Informed Sleep Foundation Model for Tasks Across Multiple Timescales},
  author = {Xu, Bo and Zhang, Song and Pan, Leyang and Zhang, Haokai and Gao, Dongrui and Xie, Jiaxin and Ying, Shaofei},
  note   = {Manuscript}
}
```

## Acknowledgments

The implementation uses [PyTorch](https://pytorch.org/), [MNE-Python](https://mne.tools/), [YASA](https://github.com/raphaelvallat/yasa), and [scikit-learn](https://scikit-learn.org/). We thank the providers of MASS, ISRUC-SLEEP, DREAMS, CAP, and the study datasets.

## Contact

Corresponding author: Shaofei Ying, [yingshaofei@uestc.edu.cn](mailto:yingshaofei@uestc.edu.cn).
