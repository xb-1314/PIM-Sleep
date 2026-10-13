import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import copy
import csv
import gc
import hashlib
import json
import pickle
import random
import re
from collections import defaultdict
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, roc_auc_score
)

# =========================
# 配置
# =========================
class FinetuneConfig:
    def __init__(self):
        # =========================
        # 任务执行开关
        # 想跑哪个任务就 True，不想跑就 False
        # =========================
        # Test-only mode reloads the saved LOSO checkpoint that corresponds to
        # each held-out subject. It never trains or updates model parameters.
        # "saved_repro" loads fixed-final-epoch LOSO checkpoints and reports
        # the single configured decoder. Use
        # "train" only when a genuinely new set of checkpoints is wanted.
        self.run_sleep_stage_classification = False
        self.sleep_stage_run_mode = "train"  # "saved_repro" or "train"
        self.run_saved_sleep_stage_test = False
        self.run_stage_generation_augmentation = False
        self.run_stage_external_generalization = False
        self.run_sleep_disorder_classification = False
        self.run_slow_wave_detection = False
        self.run_spindle_detection = True
        # DREAMS专家标注的纺锤波事件检测开关。
        # 任务执行开关保持原配置；运行纺锤波任务时启用 run_spindle_detection。
  

        # 下游 1：DREAMS healthy 睡眠分期（接 03generation_decoder 权重）
        self.stage_train_dirs = [
            r"D:/dataset/DatabaseSubjects/processed_morphology_CLAS_EOG/",
            # r"D:/dataset/DREAMERS/DREAMS Patients database/data/processed_morphology_CLAS/",
            # r"D:/dataset/CLAS/sleep_base/processed_morphology_CLAS_EOG/",
            # r"D:/dataset/ISRUC_S3/RawData/processed_morphology_CLAS_EOG_five/",
        ]
        self.stage_external_test_dirs = [
            r"D:/dataset/ISRUC_S1/processed_morphology_CLAS_EOG/",
        ]
        # ===== 第四下游任务：睡眠分期生成增强 =====
        # Task-local generator: 04 constructs and trains the masked-patch
        # reconstruction head directly from this event-pretrained backbone.
        # It never imports 03generation_decoder.py or reads generation_model.pt.
        self.generation_direct_backbone_path = (
            r"./sleep_eeg_pretrained_backbone_event_detached_bert_context/"
            r"sleep_eeg_bert_backbone_pretrained.pt"
        )
        self.generation_small_sample_output_dir = (
            r"D:/dataset/sleep_mass_dreams_clas_experiments_eeg_eog_none/"
            r"gener_decoder/shujushengchengmax"
        )
        # SleepGPT-inspired, model-compatible direct generation head.
        self.generation_direct_epochs = 60
        self.generation_direct_head_only_epochs = 10
        self.generation_direct_patience = 8
        self.generation_direct_min_delta = 1e-5
        self.generation_direct_checkpoint_interval = 10
        self.generation_direct_batch_size = 4
        self.generation_direct_head_lr = 1e-4
        self.generation_direct_encoder_lr = 1e-5
        self.generation_direct_cnn_lr = 3e-6
        self.generation_direct_weight_decay = 1e-4
        self.generation_direct_patch_samples = 100
        self.generation_direct_mask_ratio = 0.75
        self.generation_direct_decoder_dim = 128
        self.generation_direct_decoder_layers = 2
        self.generation_direct_decoder_heads = 4
        self.generation_direct_decoder_dropout = 0.1
        self.generation_direct_smooth_l1_beta = 0.5
        self.generation_direct_frequency_loss_weight = 0.2
        self.generation_direct_frequency_head_fraction = 0.5
        self.generation_direct_frequency_waveform_fraction = 0.5
        self.generation_direct_frequency_min_hz = 0.5
        self.generation_direct_frequency_max_hz = 30.0
        self.generation_direct_stft_window_samples = 200
        self.generation_direct_stft_hop_samples = 100
        self.generation_direct_stft_fft_size = 256
        self.generation_direct_frequency_fusion_gate_init = 0.1
        self.generation_direct_validation_fraction = 0.1
        # Legacy LOSO/generation-backbone settings are retained for unrelated
        # modes, but the small-sample task below never reads either value.
        self.generation_full_model_path = (
            r"./sleep_eeg_generation_decoder/generation_model_epoch_0010.pt"
        )
        self.generation_source_script = r"./03generation_decoder.py"
        # Dataset selector for this task only: "clas" or "dreams".
        self.generation_small_sample_dataset_mode = "dreams"
        generation_dataset_profiles = {
            "clas": {
                "name": "CLAS",
                "dirs": [
                    r"D:/dataset/CLAS/sleep_base/"
                    r"processed_morphology_CLAS_EOG/",
                ],
                "train_pool": 27,
                "validation": 4,
                "test": 4,
                "output_name": "clas",
            },
            "dreams": {
                "name": "DREAMS",
                "dirs": [
                    r"D:/dataset/DatabaseSubjects/"
                    r"processed_morphology_CLAS_EOG/",
                ],
                "train_pool": 12,
                "validation": 4,
                "test": 4,
                "output_name": (
                    "shujushengchengmax"
                ),
            },
        }
        generation_dataset_mode = str(
            self.generation_small_sample_dataset_mode
        ).strip().lower()
        if generation_dataset_mode not in generation_dataset_profiles:
            raise ValueError(
                "generation_small_sample_dataset_mode must be "
                "'clas' or 'dreams'."
            )
        generation_dataset_profile = generation_dataset_profiles[
            generation_dataset_mode
        ]
        self.generation_stage_train_dirs = list(
            generation_dataset_profile["dirs"]
        )
        self.generation_small_sample_dataset_name = str(
            generation_dataset_profile["name"]
        )
        self.generation_target_stages = [0, 1, 2, 3, 4]
        self.generation_samples_per_stage = None
        self.generation_save_synthetic_npz = True
        self.generation_mask_seed = 420000
        # Dataset-specific train-pool/validation/test sizes are selected by
        # generation_small_sample_dataset_mode above. The DREAMS experiment
        # uses the approved nested 1/2/4/7/12 conditions and five repeats.
        generation_train_size_profiles = {
            "dreams": [1, 2, 4, 7, 12],
            "clas": [1, 2, 4, 7, 12, 27],
        }
        self.generation_small_sample_train_sizes = list(
            generation_train_size_profiles[generation_dataset_mode]
        )
        self.generation_small_sample_repeats = 5
        self.generation_small_sample_train_pool_subjects = int(
            generation_dataset_profile["train_pool"]
        )
        self.generation_small_sample_validation_subjects = int(
            generation_dataset_profile["validation"]
        )
        self.generation_small_sample_test_subjects = int(
            generation_dataset_profile["test"]
        )
        self.generation_small_sample_train_sizes = [
            value for value in self.generation_small_sample_train_sizes
            if value <= self.generation_small_sample_train_pool_subjects
        ]
        if self.generation_small_sample_train_pool_subjects not in (
            self.generation_small_sample_train_sizes
        ):
            self.generation_small_sample_train_sizes.append(
                self.generation_small_sample_train_pool_subjects
            )
        self.generation_small_sample_split_seed = 20260731
        self.generation_small_sample_split_search_trials = 20000
        self.generation_small_sample_subset_seed = 7312026
        self.generation_small_sample_classifier_seed = 27312026
        self.generation_small_sample_selection_metric = "acc"
        self.generation_small_sample_selection_tiebreakers = [
            "macro_f1",
            "kappa",
        ]
        # Diagnostic only: print current-epoch test metrics after every
        # validation pass. These values never participate in checkpoint
        # selection or early stopping.
        self.generation_small_sample_test_each_epoch = True
        self.generation_small_sample_output_name = str(
            generation_dataset_profile["output_name"]
        )
        # Run the two phases separately:
        # 1) "prepare_generation_cache": fit one generator on the fixed
        #    12-subject DREAMS training pool, then freeze it and build five
        #    full-pool caches using five independent Mask seeds.
        # 2) "downstream_training": never runs the generator; it consumes the
        #    locked caches and performs the paired classifier trainings.
        self.generation_small_sample_run_mode = "downstream_training"
        # Long experiment controls. Completed arms are skipped automatically.
        # max_repeats=None means all five repeats.
        self.generation_small_sample_max_repeats = None
        # Training-subject selector ("button"):
        # - set to one configured train size (DREAMS: 1/2/4/7/12);
        # - set to "all" to run every default condition valid for that dataset.
        # This controls downstream training only. Generation cache preparation
        # always covers the selected dataset's fixed training pool, so changing
        # this value never requires regeneration.
        self.generation_small_sample_selected_train_size = 3
        self.run_slow_wave_generation_augmentation = False
        self.slow_wave_generation_target_labels = [0, 1]
        self.slow_wave_generation_samples_per_label = 200

        # 下游 2：CAP + DREAMS HC 睡眠障碍三分类主实验
        self.disorder_train_dirs = [
            r"D:/dataset/cap/",
        ]
        # External healthy subjects are included in pooled subject-level CV.
        self.disorder_include_dreams_hc = True
        self.disorder_dreams_hc_dirs = [
            r"D:/dataset/DatabaseSubjects/processed_morphology_CLAS_EOG/",
        ]
        self.disorder_dreams_hc_expected_count = 20
        # Model B: keep the event-detached backbone frozen, stitch overlapping
        # 20-epoch outputs into one unique full-night sequence, and perform
        # W/N1/N2/N3/REM-aware subject-level aggregation. This task does not
        # use the sleep-stage fixed-STFT branch.
        self.disorder_output_dir = (
            r"D:/dataset/sleep_mass_dreams_clas_experiments_eeg_eog_none/"
            r"gener_decoder/disorder_cap_dreams_hc"
        )
        self.disorder_feature_batch_size = 8
        self.disorder_subject_batch_size = 8
        self.disorder_n_splits = 4
        self.disorder_epochs = 100
        self.disorder_patience = 15
        self.disorder_min_delta = 1e-4
        self.disorder_head_lr = 1e-4
        self.disorder_weight_decay = 1e-4
        self.disorder_attention_hidden_dim = 64
        self.disorder_head_dropout = 0.30
        self.disorder_use_class_weights = True
        # SleepGPT-inspired CAP probe: one multiclass RF per sleep stage.
        # The five W/N1/N2/N3/REM probability vectors are averaged, while one
        # shared PCA/RF setting is selected from fused inner-CV predictions.
        # The final five models are refit on the complete outer trainval set.
        self.disorder_rf_inner_splits = 3
        # Preserve both the typical embedding of each sleep stage and its
        # within-stage variability. The latter is intended to retain transient
        # NFLE-related changes that stage-wise mean pooling can average away.
        self.disorder_rf_use_stage_std = True
        self.disorder_rf_pca_components = [16, 24, 32]
        self.disorder_rf_max_depths = [5, None]
        self.disorder_rf_min_samples_leaf = [2, 3]
        self.disorder_rf_n_estimators = 500
        self.disorder_rf_max_features = "sqrt"
        self.disorder_rf_class_weight = "balanced_subsample"
        self.disorder_rf_n_jobs = -1
        # A compact whole-night sleep-architecture classifier complements the
        # stage-embedding RFs. Its regularization and its contribution to the
        # final probability are selected strictly inside each outer trainval
        # fold; weight 0.0 keeps the original stage-only model as a candidate.
        self.disorder_architecture_logistic_cs = [0.01, 0.10, 1.0]
        self.disorder_architecture_fusion_weights = [
            0.0, 0.05, 0.10, 0.15, 0.20
        ]
        # Prefer the smallest architecture contribution whose inner OOF AUC
        # remains effectively tied with the maximum. This prevents a noisy
        # 3-fold estimate from assigning too much weight to the weaker
        # sleep-architecture classifier.
        self.disorder_architecture_fusion_auc_tolerance = 0.005
        self.disorder_architecture_epoch_seconds = 30.0
        # Select class multipliers from inner OOF predictions only. They affect
        # the final argmax decision, while ROC-AUC keeps the original scores.
        self.disorder_decision_hc_multipliers = [
            1.0, 1.25, 1.50, 1.75, 2.0
        ]
        self.disorder_decision_rbd_multipliers = [
            1.0, 1.10, 1.20, 1.35, 1.50
        ]

        self.disorder_repeat_seeds = [
           35,36,37,38,39,40,41,42,43,44
        ]
        self.disorder_selection_metric = "macro_ovr_auroc"
        self.disorder_selection_tiebreakers = [
            "macro_f1",
            "balanced_acc",
            "acc",
        ]
        self.slow_wave_train_dirs = [
            r"D:/dataset/CLAS/sleep_base/processed_morphology_CLAS_EOG/",
        ]
        # 慢波任务使用独立短名称，避免 Windows 完整输出路径超过 260 字符。
        # 睡眠分期仍使用原有 experiment_name 和检查点目录。
        self.slow_wave_experiment_name = "slowwave_clas_eog_loso_v1"

        # =========================
        # 下游任务6：纺锤波逐点/事件级检测
        # =========================
        # CLAS C3-A1 morphology NPZ signals and spindle onset/duration TXT annotations.
        self.spindle_train_dirs = [
            r"D:/dataset/CLAS/sleep_base/processed_morphology_CLAS_EOG/",
        ]
        self.spindle_annotation_dir = r"D:/dataset/CLAS/Base_SLP035_Spindle"
        self.spindle_dataset_name = "CLAS C3-A1 (N2)"
        self.spindle_output_root = (
            r"D:/dataset/sleep_mass_dreams_clas_experiments_eeg_eog_none/"
            r"gener_decoder/clas_spindles"
        )
        self.spindle_experiment_name = (
            "sub-0.5-3-0.6_fixed_threshold_0p6"
        )
        # CLAS spindle labels; restrict training/evaluation to N2.
        self.spindle_expert = "CLAS"
        # Detect completion within the current experiment directory.
        self.spindle_auto_resume = True

        # 留一被试评估：每折留出1名被试，其余被试全部用于训练。
        # 不设置独立验证集，不使用early stopping，训练至 spindle_epochs 的最后一轮。
        self.spindle_split_protocol = (
            "leave_one_subject_out_fixed_threshold_train_loss_plateau"
        )
        # 仅使用完整被试分离的 LOSO；旧的短帧参数保留以兼容数据构建。
        self.spindle_split_mode = "subject"
        self.spindle_random_frame_seconds = 3.0
        self.spindle_random_frame_overlap = 0.80
        self.spindle_random_block_seconds = 60.0
        # Subject-LOSO uses the same frame construction, while keeping whole
        # subjects strictly separated between training and testing.
        self.spindle_subject_frame_seconds = self.spindle_random_frame_seconds
        self.spindle_subject_frame_overlap = self.spindle_random_frame_overlap
        self.spindle_random_train_fraction = 0.70
        self.spindle_random_validation_fraction = 0.15
        self.spindle_random_test_fraction = 0.15
        self.spindle_random_segment_experiment_name = (
            "suiji-0.5-3-0.6"
        )
        self.spindle_num_folds = 8

        # 保留长度20的位置编码表，以完整加载预训练骨干的位置编码权重；
        # 当前纺锤波检测的实际输入序列长度为1个短frame。
        self.spindle_seq_len = 20
        self.spindle_sampling_rate = 100
        self.spindle_batch_size = 16

        # 仿照睡眠分期：第1～5轮只训练检测头，第6轮解冻最后一层
        # Transformer；该阶段连续20轮无提升后，再解冻倒数第二层。
        # EEG CNN、EOG编码器和融合层始终冻结。
        self.spindle_epochs = 2
        self.spindle_freeze_backbone_epochs = 5
        self.spindle_plateau_transition_patience = 20
        self.spindle_head_lr = 1e-4
        self.spindle_backbone_lr = 1e-5
        self.spindle_enable_plateau_second_transformer_stage = True
        self.spindle_lr_last_transformer_stage2 = 3e-6
        self.spindle_lr_second_last_transformer = 1e-6
        self.spindle_weight_decay = 1e-4
        self.spindle_pos_weight_max = 10.0
        # 损失=BCE + 0.5*Dice。BCE的正类权重使用训练被试唯一采样点统计，
        # 采用sqrt(负样本数/正样本数)，并限制最大为10，防止过度预测正类。
        self.spindle_dice_loss_weight = 0.5

        # 概率高阈值在训练前固定；不在测试被试上搜索阈值。
        self.spindle_probability_threshold = 0.6

        # 主指标阈值：预测事件与真实事件的IoU达到该值才算匹配成功。
        self.spindle_primary_iou_threshold = 0.60
        self.spindle_report_iou_thresholds = [0.6]

        # SUMOv2事件后处理：
        # 1）在有效区间内平滑概率并使用高/低滞回阈值；
        # 2）删除持续时间<0.3秒的事件；
        # 3）超过2.5秒的事件在概率低谷处分割，而不是整段删除；
        # 4）只有间隔严格<0.1秒时才合并相邻预测；
        # 5）只在连续N2区间中逐点分割。
        self.spindle_min_duration_seconds = 0.5
        self.spindle_max_duration_seconds = 3
        self.spindle_merge_gap_seconds = 0.10
        # Event post-processing: smooth probabilities, use hysteresis
        # thresholding, then split overlong events at probability valleys.
        self.spindle_probability_smoothing_seconds = 0.10
        self.spindle_hysteresis_low_ratio = 0.60
        self.spindle_long_event_valley_search_seconds = 0.25
        self.spindle_n2_only_segmentation = True
        self.spindle_head_channels = 128
        self.spindle_head_layers = 3
        self.spindle_head_dropout = 0.20

        # 旧的文本辅助分支目录。纯 EEG 主线中不再使用，保留字段仅为兼容。
        self.pretrained_model_dir = r"./sleep_domain_pretrained_mass_CLAS"
        # EEG 主干预训练权重（主线真正使用）
        # Use the same upstream checkpoint that produced the original
        # loso_plateau_two_transformer_* checkpoints evaluated by kuaishutest.
        # The selftrain/selftrainduo checkpoints are different models and
        # therefore cannot reproduce that downstream raw/Viterbi/gated result.
        # r"./sleep_eeg_pretrained_backbone_event_detached_bert_context\sleep_eeg_bert_backbone_pretrained.pt"
        self.eeg_pretrained_model_path = (
            r"./sleep_eeg_pretrained_backbone_event_detached_bert_context\sleep_eeg_bert_backbone_pretrained.pt"
        )
        self.load_full_backbone_pretrain = True
        self.interpolate_pretrained_pos_embed = False
        # Sleep-stage backbone init:
        # - "event": load 03noneteacher event-only backbone
        #   (eeg_pretrained_model_path);
        # - "generation_decoder": load the MAE-adapted encoder from
        #   03generation_decoder.py (generation_full_model_path). CNN encoders
        #   come from backbone.*, and seq_encoder is taken from the trained
        #   visible_encoder copy used during patch MAE.
        self.stage_backbone_checkpoint_source = "event"
        # "linear_probe": freeze backbone for the whole run and train only task heads.
        # "finetune": keep the previous freeze-then-unfreeze downstream behavior.
        self.probe_mode = "finetune"
        self.use_pretrained_backbone = True

        self.output_root = (
            r"D:/dataset/sleep_mass_dreams_clas_experiments_eeg_eog_none/"
            r"gener_decoder"
        )
        # LOSO：其余被试训练固定轮数，最后一轮模型仅在训练结束后测试。
        self.experiment_name = "dreams"
        # Optional absolute override for sleep-stage LOSO outputs. When set,
        # folds are written directly under this directory.
        self.stage_output_dir = (
            r"D:/dataset/sleep_mass_dreams_clas_experiments_eeg_eog_none/"
            r"gener_decoder/Label_ablation/full/full_and_masked_masked_weight_0p1/DREAMS/loso_fixed_final"
        )
        # 复测只加载新协议的末轮模型；None 表示当前 LOSO 输出目录。
        self.saved_stage_checkpoint_dir = None
        self.saved_stage_test_output_name = "test_only_mc_gated_repro_seed2056622951"
        self.saved_stage_test_output_dir = None

        # 单一固定预测方式，参数在训练前确定，不读取测试表现做选择。
        self.use_saved_stage_mc_gated_postprocess = True
        self.saved_stage_mc_seed = 2056622951
        self.saved_stage_mc_dropout_scope = "tcn"
        self.saved_stage_mc_dropout_p = 0.02
        self.saved_stage_viterbi_weight = 1.32
        self.saved_stage_gate_confidence_threshold = 0.85
        self.saved_stage_gate_margin_threshold = 0.40
        self.saved_stage_gate_min_sequence_gain = -0.40
        self.saved_stage_primary_method = "gated_viterbi"
        self.stage_mc_gated_inference_seed = 2056622951
        # 最后一轮模型的可视化使用同一种固定预测方式和 TCN 特征。
        self.save_best_stage_visualizations = True
        self.stage_visualization_max_points = 3000
        self.stage_umap_n_neighbors = 15
        self.stage_umap_min_dist = 0.10
        self.stage_attention_layer = -1
        # None runs all 20 folds; set to 1 only for a quick diagnostic.
        self.saved_stage_max_folds = None

        # 旧文本参数，纯 EEG 主线中不再使用，保留字段仅为兼容。
        self.max_text_len = 64
        self.freeze_text_encoder_epochs = 0
        self.unfreeze_last_n_layers = 0
        self.bert_chunk_size = 0
        self.text_proj_dim = 0
        self.fusion_dim = 0

        self.seq_len = 20
        self.stride = 10
        self.batch_size = 2
        self.generation_context_seq_len = self.seq_len

        # Transductive test-time adaptation. After the best epoch has been
        # selected, use only the held-out subject's unlabeled EEG/EOG to
        # recompute CNN BatchNorm running statistics. Model weights, the
        # Transformer, fusion module and TCN head remain unchanged.
        self.use_test_time_adabn = False
        self.adabn_scope = "eeg_eog_encoder_bn_only"

        # Lightweight relative-frequency branch used only by sleep staging.
        # Every band is divided by the channel's total 0.5-30 Hz power, so the
        # branch does not use absolute signal amplitude.
        self.use_stage_relative_spectral_branch = False
        self.stage_spectral_sampling_rate = 100.0
        self.stage_spectral_bands = [
            (0.5, 4.0),   # delta
            (4.0, 8.0),   # theta
            (8.0, 12.0),  # alpha
            (12.0, 16.0), # sigma / spindle
            (16.0, 30.0), # beta
        ]
        self.stage_spectral_hidden_dim = 128
        self.stage_spectral_dropout = 0.10
        self.stage_spectral_gate_init = 0.10
        # The generation-augmentation experiment uses a waveform-derived,
        # fixed-STFT branch. It is enabled only inside that task so the ordinary
        # sleep-stage baseline configuration remains unchanged.
        self.generation_stage_use_fixed_stft_branch = True
        self.stage_spectral_mode = "relative_band_power"
        self.stage_stft_window_samples = 200
        self.stage_stft_hop_samples = 100
        self.stage_stft_fft_size = 256
        self.stage_stft_frequency_min_hz = 0.5
        self.stage_stft_frequency_max_hz = 30.0

        # Per-fold domain-adaptive self-supervised pretraining. Only the 19
        # training subjects are used; sleep-stage labels and the held-out test
        # subject are never read by this phase.
        self.use_fold_self_supervised_adaptation = False
        self.fold_ssl_epochs = 10
        self.fold_ssl_patience = 3
        self.fold_ssl_min_delta = 1e-4
        self.fold_ssl_learning_rate = 1e-5
        self.fold_ssl_weight_decay = 1e-4
        self.fold_ssl_mask_ratio = 0.15
        self.fold_ssl_valid_subject_ratio = 0.10
        self.fold_ssl_batch_size = 4
        self.fold_ssl_predictor_hidden_dim = 256
        self.fold_ssl_grad_clip_norm = 1.0

        self.num_epochs = 100
        # 不设置内层验证集；固定训练 stage_loso_epochs 轮，测试只在末轮执行。
        self.stage_loso_use_inner_validation = False
        self.stage_checkpoint_selection_policy = "fixed_final_epoch"
        self.stage_loso_epochs = 100
        self.stage_loso_max_folds = None
        self.freeze_backbone_epochs = 5
        # After the frozen warm-up, keep CNN/fusion/earlier Transformer layers
        # frozen and fine-tune only the final TransformerEncoderLayer.
        self.stage_backbone_unfreeze_strategy = "last_transformer_layer"
        self.reset_early_stopping_on_unfreeze = True
        # When the last-layer-only phase first reaches the early-stopping
        # patience, do not stop or restore an older checkpoint. Continue from
        # the current model, unfreeze the second-last Transformer layer, and
        # give the new phase one additional patience window.
        self.enable_plateau_second_transformer_stage = True
        self.lr_last_transformer_stage2 = 3e-6
        self.lr_second_last_transformer = 1e-6
        self.reset_early_stopping_on_second_unfreeze = True
        self.use_stage_class_weights = False

        # early stopping
        self.early_stopping_patience = 20
        self.early_stopping_min_delta = 1e-4
        self.early_stopping_metric = "acc"  # "acc", "kappa", "macro_f1", or "weighted_f1"
        # When two checkpoints have exactly the same accuracy, prefer the one
        # with the higher macro-F1.
        self.selection_tiebreak_metric = "macro_f1"

        self.lr_backbone = 1e-5
        self.lr_head = 1e-4
        self.weight_decay = 1e-4
        self.grad_clip_norm = 1.0
        # Viterbi uses a transition matrix estimated only from the 19 training
        # subjects. Keep the weight fixed; do not tune it on the test subject.
        self.use_viterbi_postprocess = True
        self.viterbi_transition_weight = 0.3
        # Select temporal smoothing strictly on inner validation. A zero weight
        # is the raw unique-epoch prediction and lets each fold disable Viterbi.
        self.viterbi_transition_weight_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
        self.viterbi_transition_smoothing = 1.0
        # The simple LOSO path trains on all 19 subjects directly, so no refit is needed.
        self.refit_on_all_trainval = False

        self.dropout = 0.25
        self.seq_num_layers = 2
        self.seq_num_heads = 8
        self.stage_head_type = "tcn"  # "linear", "mlp", or "tcn"
        self.stage_mlp_hidden_dim = 256
        self.stage_mlp_dropout = 0.3
        self.stage_tcn_channels = 256
        self.stage_tcn_layers = 3
        self.stage_tcn_kernel_size = 3
        self.stage_tcn_dropout = 0.35

        self.n_splits = 5
        self.seed = 42

        self.use_amp = True

        # EEG 分支设置
        self.eeg_hidden_dim = 256
        self.eeg_kernel_size = 7
        self.tasks = {
            "sleep_stage_classification": {
                "num_classes": 5,
                "class_names": ["W", "N1", "N2", "N3", "REM"],
            },
            "sleep_disorder_classification": {
                "num_classes": 3,
                "class_names": ["HC", "NFLE", "RBD"],
            },
            "slow_wave_detection": {
                "num_classes": 2,
                "class_names": ["no_slowwave", "slowwave"],
            }
        }


# =========================
# 工具函数
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_kappa(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    present_labels = sorted(set(y_true) | set(y_pred))
    if len(present_labels) < 2:
        return 0.0
    score = cohen_kappa_score(y_true, y_pred)
    if np.isnan(score):
        return 0.0
    return float(score)


def _safe_balanced_accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    present_labels = sorted(set(y_true))
    if len(present_labels) < 2:
        return float(accuracy_score(y_true, y_pred))
    score = balanced_accuracy_score(y_true, y_pred)
    if np.isnan(score):
        return 0.0
    return float(score)


def _safe_binary_auroc(y_true, y_score):
    if len(y_true) == 0:
        return 0.0
    present_labels = sorted(set(y_true))
    if len(present_labels) < 2:
        return 0.0
    try:
        score = roc_auc_score(y_true, y_score)
    except Exception:
        return 0.0
    if np.isnan(score):
        return 0.0
    return float(score)


def _softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    denom = np.sum(exp_x)
    if denom <= 0:
        return np.zeros_like(exp_x)
    return exp_x / denom


def save_confusion_matrix_plot(y_true, y_pred, class_names, save_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_disorder_confusion_matrix_plot(
    y_true, y_pred, class_names, save_path, title
):
    """Save raw counts and row-normalized recalls for disease classification."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = cm.astype(np.float64) / np.clip(
        cm.sum(axis=1, keepdims=True), 1, None
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    panels = [
        (cm, "Counts", "d", None),
        (normalized * 100.0, "Row-normalized recall (%)", ".1f", 100.0),
    ]
    for ax, (matrix, subtitle, value_format, vmax) in zip(axes, panels):
        image = ax.imshow(
            matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax
        )
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted class",
            ylabel="Ground truth",
            title=subtitle,
        )
        threshold = float(np.nanmax(matrix)) * 0.5 if matrix.size else 0.0
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(
                    col,
                    row,
                    format(matrix[row, col], value_format),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > threshold else "black",
                    fontsize=10,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mean_normalized_confusion_matrix_plot(
    mean_matrix, std_matrix, class_names, save_path, title
):
    """Plot the cell-wise mean +/- std of repeat-level normalized matrices."""
    mean_matrix = np.asarray(mean_matrix, dtype=np.float64)
    std_matrix = np.asarray(std_matrix, dtype=np.float64)
    expected_shape = (len(class_names), len(class_names))
    if mean_matrix.shape != expected_shape or std_matrix.shape != expected_shape:
        raise ValueError(
            "Mean/std confusion matrices must both have shape "
            f"{expected_shape}; got {mean_matrix.shape} and {std_matrix.shape}."
        )

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        mean_matrix * 100.0,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
    )
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Ground truth",
        title=title,
    )
    for row in range(mean_matrix.shape[0]):
        for col in range(mean_matrix.shape[1]):
            mean_value = float(mean_matrix[row, col] * 100.0)
            std_value = float(std_matrix[row, col] * 100.0)
            ax.text(
                col,
                row,
                f"{mean_value:.1f}\n+/-{std_value:.1f}",
                ha="center",
                va="center",
                color="white" if mean_value > 50.0 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean recall (%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_tsne_plot(features, labels, class_names, save_path, title, seed=42, max_points=3000):
    features = np.asarray(features)
    labels = np.asarray(labels)

    if len(features) <= 1:
        return

    if len(features) > max_points:
        idx = np.random.RandomState(seed).choice(len(features), max_points, replace=False)
        features = features[idx]
        labels = labels[idx]

    perplexity = min(30, max(5, len(features) // 10))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        perplexity=perplexity
    )
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):
            ax.scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.7, label=name)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


STAGE_PLOT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]


def _stage_visualization_caption(best_epoch, primary_method, metrics):
    return (
        f"final epoch={int(best_epoch)} | method={primary_method}\n"
        f"Acc={float(metrics.get('acc', 0.0)):.4f} | "
        f"Kappa={float(metrics.get('kappa', 0.0)):.4f} | "
        f"Macro-F1={float(metrics.get('macro_f1', 0.0)):.4f} | "
        f"Weighted-F1={float(metrics.get('weighted_f1', 0.0)):.4f}"
    )


def save_best_stage_confusion_matrix(
    labels,
    preds,
    class_names,
    save_path,
    caption,
):
    """Save count and row-normalized matrices for the selected best decoder."""
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_normalized = cm.astype(np.float64) / np.clip(
        cm.sum(axis=1, keepdims=True), 1, None
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    panels = [
        (cm, "Counts", "d", None),
        (cm_normalized * 100.0, "Row-normalized (%)", ".1f", 100.0),
    ]
    for ax, (matrix, subtitle, value_format, vmax) in zip(axes, panels):
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted stage",
            ylabel="True stage",
            title=subtitle,
        )
        threshold = float(np.nanmax(matrix)) * 0.5 if matrix.size else 0.0
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(
                    col,
                    row,
                    format(matrix[row, col], value_format),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > threshold else "black",
                    fontsize=9,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Best-model sleep-stage confusion matrix\n{caption}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prepare_stage_embedding_features(features, seed=42):
    features = np.asarray(features, dtype=np.float32)
    scaled = StandardScaler().fit_transform(features).astype(np.float32)
    if scaled.shape[1] > 50 and scaled.shape[0] > 50:
        n_components = min(50, scaled.shape[1], scaled.shape[0] - 1)
        scaled = PCA(
            n_components=n_components,
            random_state=int(seed),
        ).fit_transform(scaled).astype(np.float32)
    return scaled


def compute_stage_tsne(features, seed=42):
    prepared = _prepare_stage_embedding_features(features, seed=seed)
    n_samples = int(prepared.shape[0])
    if n_samples < 3:
        return None
    perplexity = min(30.0, max(2.0, float(n_samples - 1) / 3.0))
    return TSNE(
        n_components=2,
        random_state=int(seed),
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(prepared)


def compute_stage_umap(features, seed=42, n_neighbors=15, min_dist=0.10):
    # Lazy import avoids adding UMAP/Numba startup time to every normal launch.
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "UMAP visualization requires umap-learn. Install it with "
            "`pip install umap-learn`."
        ) from exc
    prepared = _prepare_stage_embedding_features(features, seed=seed)
    n_samples = int(prepared.shape[0])
    if n_samples < 3:
        return None
    effective_neighbors = min(max(2, int(n_neighbors)), n_samples - 1)
    reducer = UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=float(min_dist),
        metric="cosine",
        random_state=int(seed),
        n_jobs=1,
    )
    return reducer.fit_transform(prepared)


def save_stage_embedding_plot(
    embedding,
    labels,
    class_names,
    save_path,
    method_name,
    caption,
):
    if embedding is None or len(embedding) == 0:
        return
    embedding = np.asarray(embedding)
    labels = np.asarray(labels, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if np.any(mask):
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=15,
                alpha=0.72,
                color=STAGE_PLOT_COLORS[class_idx % len(STAGE_PLOT_COLORS)],
                label=f"{class_name} (n={int(mask.sum())})",
                linewidths=0,
            )
    ax.set_title(f"Best-model TCN features: {method_name}\n{caption}")
    ax.set_xlabel(f"{method_name}-1")
    ax.set_ylabel(f"{method_name}-2")
    ax.legend(frameon=True, fontsize=9, markerscale=1.4)
    ax.grid(alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_stage_attention_plots(
    mean_attention_by_head,
    stage_attention_by_head,
    class_names,
    save_path,
    heads_save_path,
    caption,
    layer_index,
):
    mean_attention_by_head = np.asarray(mean_attention_by_head, dtype=np.float64)
    stage_attention_by_head = np.asarray(stage_attention_by_head, dtype=np.float64)
    position_attention = np.nanmean(mean_attention_by_head, axis=0)
    stage_attention = np.nanmean(stage_attention_by_head, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    pos_image = axes[0].imshow(position_attention, cmap="magma", aspect="auto")
    axes[0].set_title("Mean epoch-to-epoch attention")
    axes[0].set_xlabel("Key epoch position")
    axes[0].set_ylabel("Query epoch position")
    fig.colorbar(pos_image, ax=axes[0], fraction=0.046, pad=0.04)

    stage_image = axes[1].imshow(stage_attention, cmap="magma", aspect="equal")
    axes[1].set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Key sleep stage",
        ylabel="Query sleep stage",
        title="Mean attention by true-stage pair",
    )
    threshold = float(np.nanmax(stage_attention)) * 0.55
    for row in range(stage_attention.shape[0]):
        for col in range(stage_attention.shape[1]):
            value = stage_attention[row, col]
            axes[1].text(
                col,
                row,
                f"{value:.3f}" if np.isfinite(value) else "-",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if np.isfinite(value) and value > threshold else "black",
            )
    fig.colorbar(stage_image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Last Transformer attention (layer index {layer_index})\n{caption}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    n_heads = int(mean_attention_by_head.shape[0])
    n_cols = min(4, n_heads)
    n_rows = int(np.ceil(n_heads / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    finite_values = mean_attention_by_head[np.isfinite(mean_attention_by_head)]
    common_max = float(finite_values.max()) if finite_values.size else 1.0
    for head_idx, ax in enumerate(axes.flat):
        if head_idx >= n_heads:
            ax.axis("off")
            continue
        image = ax.imshow(
            mean_attention_by_head[head_idx],
            cmap="magma",
            aspect="auto",
            vmin=0.0,
            vmax=common_max,
        )
        ax.set_title(f"Head {head_idx + 1}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
    fig.colorbar(image, ax=list(axes.flat), fraction=0.02, pad=0.02)
    fig.suptitle(
        f"Per-head attention, Transformer layer {layer_index}\n{caption}",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.86, wspace=0.28, hspace=0.34)
    fig.savefig(heads_save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def task_output_dir(cfg, task_name):
    if task_name == "sleep_stage_classification":
        stage_output_dir = str(
            getattr(cfg, "stage_output_dir", "") or ""
        ).strip()
        if stage_output_dir:
            return stage_output_dir
    if task_name == "slow_wave_detection":
        exp_name = str(
            getattr(cfg, "slow_wave_experiment_name", "") or ""
        ).strip()
    elif task_name == "sleep_disorder_classification":
        disorder_output_dir = str(
            getattr(cfg, "disorder_output_dir", "") or ""
        ).strip()
        if disorder_output_dir:
            return disorder_output_dir
        exp_name = str(getattr(cfg, "experiment_name", "") or "").strip()
    else:
        exp_name = str(getattr(cfg, "experiment_name", "") or "").strip()
    if exp_name:
        return os.path.join(cfg.output_root, task_name, exp_name)
    return os.path.join(cfg.output_root, task_name)


def stage_protocol_signature(cfg):
    """Record the sleep-stage training protocol used by this generated run."""
    names = [
        "seed",
        "seq_len",
        "stride",
        "batch_size",
        "eeg_hidden_dim",
        "eeg_kernel_size",
        "seq_num_layers",
        "seq_num_heads",
        "dropout",
        "stage_head_type",
        "stage_tcn_channels",
        "stage_tcn_layers",
        "stage_tcn_kernel_size",
        "stage_tcn_dropout",
        "stage_loso_epochs",
        "stage_loso_max_folds",
        "freeze_backbone_epochs",
        "stage_backbone_unfreeze_strategy",
        "enable_plateau_second_transformer_stage",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_metric",
        "selection_tiebreak_metric",
        "lr_backbone",
        "lr_head",
        "lr_last_transformer_stage2",
        "lr_second_last_transformer",
        "weight_decay",
        "grad_clip_norm",
        "use_stage_class_weights",
        "use_viterbi_postprocess",
        "viterbi_transition_weight",
        "viterbi_transition_smoothing",
        "saved_stage_primary_method",
        "stage_checkpoint_selection_policy",
        "stage_mc_gated_inference_seed",
        "saved_stage_mc_dropout_scope",
        "saved_stage_mc_dropout_p",
        "saved_stage_viterbi_weight",
        "saved_stage_gate_confidence_threshold",
        "saved_stage_gate_margin_threshold",
        "saved_stage_gate_min_sequence_gain",
        "eeg_pretrained_model_path",
        "load_full_backbone_pretrain",
        "stage_backbone_checkpoint_source",
        "generation_full_model_path",
        "stage_output_dir",
        "probe_mode",
    ]
    signature = {name: getattr(cfg, name, None) for name in names}
    # Preserve the exact legacy signature when the branch is disabled so
    # existing ordinary sleep-stage checkpoints remain reproducible.
    if bool(getattr(cfg, "use_stage_relative_spectral_branch", False)):
        for name in [
            "use_stage_relative_spectral_branch",
            "stage_spectral_mode",
            "stage_spectral_gate_init",
            "stage_stft_window_samples",
            "stage_stft_hop_samples",
            "stage_stft_fft_size",
            "stage_stft_frequency_min_hz",
            "stage_stft_frequency_max_hz",
        ]:
            signature[name] = getattr(cfg, name, None)
    return signature




def list_morphology_files(processed_dir):
    files = []
    if not os.path.isdir(processed_dir):
        print(f"Warning: processed directory not found: {processed_dir}")
        return files
    for fname in os.listdir(processed_dir):
        if fname.endswith("_morphology.npz"):
            files.append(os.path.join(processed_dir, fname))
    files.sort()
    return files


def load_subject_records_from_dirs(processed_dirs):
    all_records = []
    for d in processed_dirs:
        recs = load_subject_records(d)
        all_records.extend(recs)
    print(f"合并后总被试数: {len(all_records)}")
    return all_records


def load_disorder_training_records(cfg):
    """Load CAP and optional DREAMS HC without changing other task labels.

    Each record is one participant. DREAMS participates in both train and test
    folds of the existing pooled stratified CV, not training-only augmentation.
    """
    records = []
    seen_files = set()
    seen_subjects = set()
    sources = [("CAP", cfg.disorder_train_dirs, None)]
    if getattr(cfg, "disorder_include_dreams_hc", False):
        sources.append(("DREAMS", cfg.disorder_dreams_hc_dirs, 0))
    for source, directories, forced_label in sources:
        source_count = 0
        for directory in directories:
            files = list_morphology_files(directory)
            if not files:
                raise ValueError(f"No morphology files for {source}: {directory}")
            for file_path in files:
                canonical = os.path.normcase(os.path.realpath(file_path))
                if canonical in seen_files:
                    raise ValueError(f"Duplicate disorder input file: {file_path}")
                seen_files.add(canonical)
            for record in load_subject_records(directory):
                original_id = str(record["subject_id"])
                subject_id = f"{source}__{original_id}"
                if subject_id in seen_subjects:
                    raise ValueError(f"Duplicate disorder participant: {subject_id}")
                seen_subjects.add(subject_id)
                original_label = int(record["sleep_disorder_label"])
                if forced_label is not None and original_label not in (-1, forced_label):
                    raise ValueError(f"Conflicting HC label for {subject_id}: {original_label}")
                record["source_dataset"] = source
                record["source_directory"] = os.path.abspath(directory)
                record["original_subject_id"] = original_id
                record["original_disorder_label"] = original_label
                record["subject_id"] = subject_id
                if forced_label is not None:
                    record["sleep_disorder_label"] = forced_label
                if int(record["sleep_disorder_label"]) not in (0, 1, 2):
                    raise ValueError(f"Unsupported disorder label for {subject_id}")
                records.append(record)
                source_count += 1
        if source == "DREAMS" and source_count != cfg.disorder_dreams_hc_expected_count:
            raise ValueError(f"Expected {cfg.disorder_dreams_hc_expected_count} DREAMS HC subjects, got {source_count}")
    counts = {name: sum(int(r["sleep_disorder_label"]) == label for r in records)
              for label, name in enumerate(("HC", "NFLE", "RBD"))}
    print(f"Disorder pooled subject CV: n={len(records)}, counts={counts}")
    return records


def save_disorder_cohort_manifest(records, cfg):
    ensure_dir(cfg.disorder_output_dir)
    keys = ("subject_id", "original_subject_id", "source_dataset",
            "source_directory", "original_disorder_label", "sleep_disorder_label")
    manifest = {
        "evaluation": "pooled_subject_level_stratified_cv",
        "label_mapping": {"0": "HC", "1": "NFLE", "2": "RBD"},
        "subjects": [{**{k: r[k] for k in keys},
                      "n_epochs": len(r["eeg_epochs"])} for r in records],
    }
    with open(os.path.join(cfg.disorder_output_dir, "cohort_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _default_epoch_samples(data):
    sf = 100.0
    if "sampling_rate" in data:
        try:
            sf = float(np.array(data["sampling_rate"]).item())
        except Exception:
            sf = 100.0

    epoch_duration = 30
    if "epoch_duration" in data:
        try:
            epoch_duration = int(np.array(data["epoch_duration"]).item())
        except Exception:
            epoch_duration = 30

    return int(round(sf * epoch_duration))


def _safe_scalar_to_str(x, default=''):
    try:
        arr = np.array(x)
        if arr.shape == ():
            return str(arr.item())
        if arr.size == 1:
            return str(arr.reshape(-1)[0])
        return str(x)
    except Exception:
        return default if default != '' else str(x)


def _infer_disorder_label(disorder_label, file_path, processed_dir, data):
    if disorder_label >= 0:
        return disorder_label

    if "sleep_disorder_name" in data:
        name = _safe_scalar_to_str(data["sleep_disorder_name"], default="").strip().lower()
        if name in {"normal", "healthy", "control", "hc"}:
            return 0
        if name in {"nfle", "nocturnal_frontal_lobe_epilepsy"}:
            return 1
        if name in {"rbd", "rem_sleep_behavior_disorder"}:
            return 2

    path_hint = f"{processed_dir} {file_path}".lower()
    base_name = os.path.basename(file_path).lower()
    if re.match(r"^nfle\d+", base_name):
        return 1
    if re.match(r"^rbd\d+", base_name):
        return 2
    if re.match(r"^n\d+", base_name):
        return 0
    if "patients" in path_hint or "patient" in path_hint:
        return 1
    if "databasesubjects" in path_hint or "healthy" in path_hint:
        return 0

    return disorder_label


def _extract_slow_wave_labels(data, target_len):
    if "slow_wave_labels" in data:
        labels = np.asarray(data["slow_wave_labels"], dtype=np.int64).reshape(-1)
        if len(labels) >= target_len:
            return labels[:target_len].tolist()
        return labels.tolist() + [-100] * (target_len - len(labels))

    if "waveform_features" in data:
        try:
            waveform_features = data["waveform_features"].tolist()
            labels = []
            for feat in waveform_features:
                if isinstance(feat, dict):
                    labels.append(int(feat.get("has_slowwave", -100)))
                else:
                    labels.append(-100)
            if len(labels) >= target_len:
                return labels[:target_len]
            return labels + [-100] * (target_len - len(labels))
        except Exception:
            pass

    return [-100] * target_len


def load_subject_records(processed_dir):
    files = list_morphology_files(processed_dir)
    records = []

    for file_path in files:
        data = np.load(file_path, allow_pickle=True)

        if "descriptions_plain" in data:
            descriptions = data["descriptions_plain"].tolist()
        elif "descriptions" in data:
            descriptions = data["descriptions"].tolist()
        else:
            descriptions = []

        if not isinstance(descriptions, list):
            descriptions = list(descriptions)

        subject_id = _safe_scalar_to_str(data["subject_id"]) if "subject_id" in data else os.path.basename(file_path).replace("_morphology.npz", "")

        if "true_sleep_stages" in data:
            true_stages = data["true_sleep_stages"].tolist()
        elif "sleep_stages" in data:
            true_stages = data["sleep_stages"].tolist()
        else:
            true_stages = [-1] * len(descriptions)

        if not isinstance(true_stages, list):
            true_stages = list(true_stages)

        if "sleep_disorder_label" in data:
            disorder_label = int(np.array(data["sleep_disorder_label"]).item())
        else:
            disorder_label = -1
        disorder_label = _infer_disorder_label(disorder_label, file_path, processed_dir, data)

        if "eeg_epochs" in data:
            eeg_epochs = np.asarray(data["eeg_epochs"], dtype=np.float32)
            if eeg_epochs.ndim == 1:
                eeg_epochs = eeg_epochs.reshape(1, -1)
        else:
            samples_per_epoch = _default_epoch_samples(data)
            base_len = len(true_stages) if len(true_stages) > 0 else len(descriptions)
            eeg_epochs = np.zeros((base_len, samples_per_epoch), dtype=np.float32)
            print(f"Warning: {file_path} 缺少 eeg_epochs，已回退为零输入。")

        eog_loc_epochs = np.asarray(
            data["eog_loc_epochs"] if "eog_loc_epochs" in data else np.zeros_like(eeg_epochs),
            dtype=np.float32,
        )
        if eog_loc_epochs.ndim == 1:
            eog_loc_epochs = eog_loc_epochs.reshape(1, -1)
        eog_roc_epochs = np.asarray(
            data["eog_roc_epochs"] if "eog_roc_epochs" in data else np.zeros_like(eeg_epochs),
            dtype=np.float32,
        )
        if eog_roc_epochs.ndim == 1:
            eog_roc_epochs = eog_roc_epochs.reshape(1, -1)
        eog_available = np.asarray(
            data["eog_available"] if "eog_available" in data else [0.0, 0.0],
            dtype=np.float32,
        ).reshape(-1)
        if eog_available.size != 2:
            eog_available = np.asarray([0.0, 0.0], dtype=np.float32)

        n_candidates = [len(eeg_epochs), len(eog_loc_epochs), len(eog_roc_epochs)]
        if len(true_stages) > 0:
            n_candidates.append(len(true_stages))
        if len(descriptions) > 0:
            n_candidates.append(len(descriptions))
        n = min(n_candidates) if len(n_candidates) > 0 else 0
        if n == 0:
            continue

        slow_wave_labels = _extract_slow_wave_labels(data, n)
        true_stages = true_stages[:n] if len(true_stages) >= n else (true_stages + [-1] * (n - len(true_stages)))
        eeg_epochs = eeg_epochs[:n]
        eog_loc_epochs = eog_loc_epochs[:n]
        eog_roc_epochs = eog_roc_epochs[:n]
        descriptions = descriptions[:n] if len(descriptions) >= n else descriptions

        records.append({
            "subject_id": subject_id,
            "descriptions": descriptions,
            "true_sleep_stages": true_stages,
            "sleep_disorder_label": disorder_label,
            "slow_wave_labels": slow_wave_labels,
            "eeg_epochs": eeg_epochs,
            "eog_loc_epochs": eog_loc_epochs,
            "eog_roc_epochs": eog_roc_epochs,
            "eog_available": eog_available,
        })

    print(f"共加载被试数: {len(records)}")
    return records


def build_window_starts(n, seq_len, stride):
    if n <= seq_len:
        return [0]

    starts = list(range(0, n - seq_len + 1, stride))
    last_start = n - seq_len
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def pad_texts(texts, target_len, pad_text="padding epoch"):
    # 纯 EEG 主线中不再使用，保留函数名兼容。
    if len(texts) >= target_len:
        return texts[:target_len]
    return texts + [pad_text] * (target_len - len(texts))


def pad_labels(labels, target_len, pad_value=-100):
    if len(labels) >= target_len:
        return labels[:target_len]
    return labels + [pad_value] * (target_len - len(labels))


def pad_valid_mask(valid_len, target_len):
    valid_len = int(min(max(valid_len, 0), target_len))
    return [1] * valid_len + [0] * (target_len - valid_len)


def pad_eeg_epochs(eeg_epochs, target_len):
    eeg_epochs = np.asarray(eeg_epochs, dtype=np.float32)
    if eeg_epochs.ndim not in (2, 3):
        raise ValueError(f"eeg_epochs 期望二维数组，实际形状: {eeg_epochs.shape}")
    if len(eeg_epochs) >= target_len:
        return eeg_epochs[:target_len]
    pad = np.zeros((target_len - len(eeg_epochs), *eeg_epochs.shape[1:]), dtype=np.float32)
    return np.concatenate([eeg_epochs, pad], axis=0)


def normalize_eeg_window(window_eeg, valid_mask=None):
    window_eeg = np.asarray(window_eeg, dtype=np.float32).copy()
    if valid_mask is None:
        valid_mask = np.ones(len(window_eeg), dtype=np.int64)
    else:
        valid_mask = np.asarray(valid_mask, dtype=np.int64)

    for i in range(len(window_eeg)):
        if i >= len(valid_mask) or valid_mask[i] == 0:
            continue
        channels = window_eeg[i] if window_eeg.ndim == 3 else window_eeg[i:i + 1]
        for channel_idx, x in enumerate(channels):
            mean = float(np.mean(x))
            std = float(np.std(x))
            normalized = (x - mean) / std if std > 1e-6 else x - mean
            if window_eeg.ndim == 3:
                window_eeg[i, channel_idx] = normalized
            else:
                window_eeg[i] = normalized
    return window_eeg


def build_signal_epochs(rec):
    eeg_epochs = np.asarray(rec["eeg_epochs"], dtype=np.float32)
    loc_epochs = np.asarray(rec.get("eog_loc_epochs", np.zeros_like(eeg_epochs)), dtype=np.float32)
    roc_epochs = np.asarray(rec.get("eog_roc_epochs", np.zeros_like(eeg_epochs)), dtype=np.float32)
    eog_available = np.asarray(rec.get("eog_available", [0.0, 0.0]), dtype=np.float32).reshape(-1)
    if eog_available.size != 2:
        eog_available = np.asarray([0.0, 0.0], dtype=np.float32)

    n = min(len(eeg_epochs), len(loc_epochs), len(roc_epochs))
    signal_epochs = np.stack([eeg_epochs[:n], loc_epochs[:n], roc_epochs[:n]], axis=1)
    signal_epochs[:, 1, :] *= eog_available[0]
    signal_epochs[:, 2, :] *= eog_available[1]
    return signal_epochs, eog_available


def build_stage_samples(records, seq_len=100, stride=50):
    samples = []
    for rec in records:
        raw_labels = rec["true_sleep_stages"]
        eeg_epochs, eog_available = build_signal_epochs(rec)

        n = min(len(raw_labels), len(eeg_epochs))
        if n == 0:
            continue

        labels = []
        for x in raw_labels[:n]:
            x = int(x)
            if x in [0, 1, 2, 3, 4]:
                labels.append(x)
            else:
                labels.append(-100)

        starts = build_window_starts(n, seq_len, stride)
        for s in starts:
            e = min(s + seq_len, n)

            window_labels = pad_labels(labels[s:e], seq_len, pad_value=-100)
            window_valid_mask = pad_valid_mask(e - s, seq_len)
            window_eeg = pad_eeg_epochs(eeg_epochs[s:e], seq_len)
            window_eeg = normalize_eeg_window(window_eeg, window_valid_mask)

            if all(v == -100 for v in window_labels):
                continue

            samples.append({
                "subject_id": rec["subject_id"],
                "start_idx": int(s),
                "valid_len": int(e - s),
                "eeg": window_eeg,
                "eog_available": eog_available,
                "valid_mask": window_valid_mask,
                "labels": window_labels,
            })
    return samples


def build_disorder_samples(records, seq_len=100, stride=50):
    samples = []
    for rec in records:
        disorder_label = int(rec["sleep_disorder_label"])
        if disorder_label < 0:
            continue

        eeg_epochs, eog_available = build_signal_epochs(rec)
        n = len(eeg_epochs)
        if n == 0:
            continue

        starts = build_window_starts(n, seq_len, stride)
        for s in starts:
            e = min(s + seq_len, n)
            window_valid_mask = pad_valid_mask(e - s, seq_len)
            window_eeg = pad_eeg_epochs(eeg_epochs[s:e], seq_len)
            window_eeg = normalize_eeg_window(window_eeg, window_valid_mask)

            samples.append({
                "subject_id": rec["subject_id"],
                "eeg": window_eeg,
                "eog_available": eog_available,
                "valid_mask": window_valid_mask,
                "label": disorder_label
            })
    return samples


def build_slow_wave_samples(records, seq_len=100, stride=50):
    samples = []
    for rec in records:
        raw_labels = rec.get("slow_wave_labels", [])
        eeg_epochs, eog_available = build_signal_epochs(rec)

        n = min(len(raw_labels), len(eeg_epochs))
        if n == 0:
            continue

        labels = []
        for x in raw_labels[:n]:
            x = int(x)
            if x in [0, 1]:
                labels.append(x)
            else:
                labels.append(-100)

        starts = build_window_starts(n, seq_len, stride)
        for s in starts:
            e = min(s + seq_len, n)

            window_labels = pad_labels(labels[s:e], seq_len, pad_value=-100)
            window_valid_mask = pad_valid_mask(e - s, seq_len)
            window_eeg = pad_eeg_epochs(eeg_epochs[s:e], seq_len)
            window_eeg = normalize_eeg_window(window_eeg, window_valid_mask)

            if all(v == -100 for v in window_labels):
                continue

            samples.append({
                "subject_id": rec["subject_id"],
                "eeg": window_eeg,
                "eog_available": eog_available,
                "valid_mask": window_valid_mask,
                "labels": window_labels
            })
    return samples


# =========================
# Dataset
# =========================
class SleepTextWindowDataset(Dataset):
    """
    保留原类名，内部已改为 EEG+EOG 数据集。
    tokenizer / max_text_len 参数仅为兼容旧调用方式，EEG+EOG 主线中会被忽略。
    """
    def __init__(self, samples, tokenizer=None, max_text_len=None, task_name="sleep_stage_classification"):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.task_name = task_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        out = {
            "eeg": torch.tensor(sample["eeg"], dtype=torch.float32),
            "eog_available": torch.tensor(sample["eog_available"], dtype=torch.float32),
            "valid_mask": torch.tensor(sample["valid_mask"], dtype=torch.long),
            "subject_id": sample["subject_id"],
            "start_idx": torch.tensor(sample.get("start_idx", 0), dtype=torch.long),
            "valid_len": torch.tensor(sample.get("valid_len", len(sample["valid_mask"])), dtype=torch.long),
        }

        if self.task_name == "sleep_stage_classification":
            out["labels"] = torch.tensor(sample["labels"], dtype=torch.long)
        elif self.task_name == "slow_wave_detection":
            out["labels"] = torch.tensor(sample["labels"], dtype=torch.long)
        else:
            out["labels"] = torch.tensor(sample["label"], dtype=torch.long)

        return out


# =========================
# 模型
# =========================
class EEGEpochEncoder(nn.Module):
    def __init__(self, out_dim=256, kernel_size=7, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(128, out_dim)

    def forward_temporal(self, x):
        """返回全局池化前的CNN特征[N, 128, L/8]，保留事件时间位置。"""
        return self.net[:-1](x.unsqueeze(1))

    def pool_temporal(self, temporal):
        pooled = self.net[-1](temporal).squeeze(-1)
        return self.proj(pooled)

    def forward(self, x):
        return self.pool_temporal(self.forward_temporal(x))


class SleepTextBackbone(nn.Module):
    """
    保留原类名，内部改成与 03 BERT 预训练一致的 EEG+EOG backbone：
    eeg_encoder/eog_encoder -> fusion -> mask_token/pos_embed -> Transformer seq_encoder
    """
    def __init__(
        self,
        pretrained_model_dir=None,
        seq_len=100,
        dropout=0.1,
        seq_num_layers=2,
        seq_num_heads=8,
        bert_chunk_size=0,
        eeg_hidden_dim=256,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=7
    ):
        super().__init__()
        self.eeg_encoder = EEGEpochEncoder(
            out_dim=eeg_hidden_dim,
            kernel_size=eeg_kernel_size,
            dropout=dropout
        )
        self.eog_encoder = EEGEpochEncoder(
            out_dim=eeg_hidden_dim,
            kernel_size=eeg_kernel_size,
            dropout=dropout
        )
        self.fusion = nn.Sequential(
            nn.Linear(eeg_hidden_dim * 3, eeg_hidden_dim),
            nn.LayerNorm(eeg_hidden_dim),
            nn.GELU(),
        )

        self.hidden_dim = eeg_hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, self.hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=seq_num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=seq_num_layers)
        # Frozen early layers and trainable later layers require dense tensors.
        # These runtime flags do not change parameters or checkpoint keys.
        self.seq_encoder.enable_nested_tensor = False
        self.seq_encoder.use_nested_tensor = False

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def encode_epochs(self, eeg, eog_available=None):
        bsz, seq_len, channels, sig_len = eeg.shape
        if channels != 3:
            raise ValueError(f"expected EEG+LOC+ROC channels, got shape {eeg.shape}")
        flat_eeg = eeg[:, :, 0, :].reshape(bsz * seq_len, sig_len)
        flat_loc = eeg[:, :, 1, :].reshape(bsz * seq_len, sig_len)
        flat_roc = eeg[:, :, 2, :].reshape(bsz * seq_len, sig_len)
        eeg_emb = self.eeg_encoder(flat_eeg)
        loc_emb = self.eog_encoder(flat_loc)
        roc_emb = self.eog_encoder(flat_roc)
        if eog_available is not None:
            availability = eog_available.to(device=eeg.device, dtype=eeg.dtype)
            loc_emb = loc_emb * availability[:, 0].repeat_interleave(seq_len).unsqueeze(-1)
            roc_emb = roc_emb * availability[:, 1].repeat_interleave(seq_len).unsqueeze(-1)
        fused = self.fusion(torch.cat([eeg_emb, loc_emb, roc_emb], dim=-1))
        return fused.view(bsz, seq_len, -1)

    def encode_epochs_with_eeg_temporal(self, eeg, eog_available=None):
        """一次前向同时获得epoch嵌入和中央EEG局部时间特征。

        局部特征直接复用预训练EEG CNN，没有增加新的backbone参数，
        因此原来的上游checkpoint仍可按原参数名加载。
        """
        bsz, seq_len, channels, sig_len = eeg.shape
        if channels != 3:
            raise ValueError(f"expected EEG+LOC+ROC channels, got shape {eeg.shape}")
        flat_eeg = eeg[:, :, 0, :].reshape(bsz * seq_len, sig_len)
        flat_loc = eeg[:, :, 1, :].reshape(bsz * seq_len, sig_len)
        flat_roc = eeg[:, :, 2, :].reshape(bsz * seq_len, sig_len)
        eeg_temporal = self.eeg_encoder.forward_temporal(flat_eeg)
        eeg_emb = self.eeg_encoder.pool_temporal(eeg_temporal)
        loc_emb = self.eog_encoder(flat_loc)
        roc_emb = self.eog_encoder(flat_roc)
        if eog_available is not None:
            availability = eog_available.to(device=eeg.device, dtype=eeg.dtype)
            loc_emb = loc_emb * availability[:, 0].repeat_interleave(seq_len).unsqueeze(-1)
            roc_emb = roc_emb * availability[:, 1].repeat_interleave(seq_len).unsqueeze(-1)
        fused = self.fusion(torch.cat([eeg_emb, loc_emb, roc_emb], dim=-1))
        fused = fused.view(bsz, seq_len, -1)
        eeg_temporal = eeg_temporal.view(
            bsz, seq_len, eeg_temporal.size(1), eeg_temporal.size(2)
        )
        return fused, eeg_temporal

    def contextualize(self, x, valid_mask=None):
        """加入位置编码并用Transformer学习前后epoch上下文。"""
        seq_len = x.size(1)
        x = self.drop(self.norm(x + self.pos_embed[:, :seq_len, :]))
        if valid_mask is not None:
            return self.seq_encoder(x, src_key_padding_mask=(valid_mask == 0))
        return self.seq_encoder(x)

    def forward_with_eeg_temporal(self, eeg, eog_available=None, valid_mask=None):
        epoch_embeddings, eeg_temporal = self.encode_epochs_with_eeg_temporal(
            eeg, eog_available=eog_available
        )
        contextual = self.contextualize(epoch_embeddings, valid_mask=valid_mask)
        return contextual, eeg_temporal

    def forward(self, eeg, eog_available=None, valid_mask=None, epoch_mask=None, random_token_emb=None):
        x = self.encode_epochs(eeg, eog_available=eog_available)

        if epoch_mask is not None:
            mask = epoch_mask.unsqueeze(-1)
            mask_token = self.mask_token.expand(x.size(0), x.size(1), -1)
            replacement = random_token_emb if random_token_emb is not None else mask_token
            x = torch.where(mask, replacement, x)

        return self.contextualize(x, valid_mask=valid_mask)


class DirectGenerationSettings:
    """Task-local settings for the 04 direct masked-patch generator."""

    def __init__(self, cfg):
        self.seq_len = int(cfg.generation_context_seq_len)
        self.sample_length = 3000
        self.sampling_rate = 100.0
        self.eeg_hidden_dim = int(cfg.eeg_hidden_dim)
        self.dropout = float(cfg.dropout)
        self.patch_size_samples = int(cfg.generation_direct_patch_samples)
        self.mask_ratio = float(cfg.generation_direct_mask_ratio)
        self.decoder_channels = int(cfg.generation_direct_decoder_dim)
        self.decoder_num_layers = int(cfg.generation_direct_decoder_layers)
        self.decoder_num_heads = int(cfg.generation_direct_decoder_heads)
        self.decoder_dropout = float(cfg.generation_direct_decoder_dropout)
        self.smooth_l1_beta = float(cfg.generation_direct_smooth_l1_beta)
        self.frequency_loss_weight = float(
            cfg.generation_direct_frequency_loss_weight
        )
        self.frequency_head_fraction = float(
            cfg.generation_direct_frequency_head_fraction
        )
        self.frequency_waveform_fraction = float(
            cfg.generation_direct_frequency_waveform_fraction
        )
        self.frequency_min_hz = float(
            cfg.generation_direct_frequency_min_hz
        )
        self.frequency_max_hz = float(
            cfg.generation_direct_frequency_max_hz
        )
        self.stft_window_samples = int(
            cfg.generation_direct_stft_window_samples
        )
        self.stft_hop_samples = int(
            cfg.generation_direct_stft_hop_samples
        )
        self.stft_fft_size = int(cfg.generation_direct_stft_fft_size)
        self.frequency_fusion_gate_init = float(
            cfg.generation_direct_frequency_fusion_gate_init
        )

    def as_dict(self):
        return dict(vars(self))


def _direct_valid_patch_mask(
    valid_mask, eog_available, sample_length, patch_size
):
    batch_size, seq_len = valid_mask.shape
    n_patches = int(np.ceil(int(sample_length) / max(1, int(patch_size))))
    channel_available = torch.cat(
        [
            torch.ones(
                (batch_size, 1),
                dtype=eog_available.dtype,
                device=eog_available.device,
            ),
            eog_available,
        ],
        dim=1,
    ).bool()
    return (
        valid_mask.bool()[:, :, None, None]
        & channel_available[:, None, :, None]
    ).expand(-1, seq_len, -1, n_patches)


def _direct_patch_mask(
    valid_mask,
    eog_available,
    sample_length,
    patch_size,
    mask_ratio,
    generator=None,
    target_indices=None,
):
    valid_patches = _direct_valid_patch_mask(
        valid_mask, eog_available, sample_length, patch_size
    )
    batch_size, seq_len = valid_mask.shape
    if target_indices is None:
        targets = []
        for index in range(batch_size):
            choices = torch.nonzero(
                valid_mask[index].bool(), as_tuple=False
            ).reshape(-1)
            if choices.numel() == 0:
                raise RuntimeError("Generator sample has no valid target epoch.")
            selected = torch.randint(
                choices.numel(),
                (1,),
                device=valid_mask.device,
                generator=generator,
            )
            targets.append(choices[selected].reshape(()))
        target_indices = torch.stack(targets)
    else:
        target_indices = torch.as_tensor(
            target_indices, dtype=torch.long, device=valid_mask.device
        ).reshape(-1)
        if target_indices.numel() == 1 and batch_size > 1:
            target_indices = target_indices.expand(batch_size)
    if target_indices.numel() != batch_size:
        raise ValueError(
            f"target_indices has {target_indices.numel()} values for "
            f"batch_size={batch_size}."
        )
    if (
        (target_indices < 0).any()
        or (target_indices >= seq_len).any()
        or not valid_mask.bool()[
            torch.arange(batch_size, device=valid_mask.device), target_indices
        ].all()
    ):
        raise ValueError("target_indices must point to valid epochs.")
    ratio = min(max(float(mask_ratio), 0.0), 1.0)
    if ratio <= 0.0:
        return torch.zeros_like(valid_patches)
    n_patches = valid_patches.size(-1)
    exact_count = float(n_patches) * ratio
    lower_count = min(int(np.floor(exact_count)), n_patches)
    fractional_count = exact_count - float(lower_count)
    num_masked = torch.full(
        (batch_size,),
        lower_count,
        dtype=torch.long,
        device=valid_mask.device,
    )
    if fractional_count > 0.0 and lower_count < n_patches:
        round_up = torch.rand(
            (batch_size,),
            device=valid_mask.device,
            generator=generator,
        ) < fractional_count
        num_masked += round_up.long()
    num_masked = num_masked.clamp(min=1, max=n_patches)
    scores = torch.rand(
        (batch_size, n_patches),
        device=valid_mask.device,
        generator=generator,
    )
    order = scores.argsort(dim=-1)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(
        n_patches, device=valid_mask.device, dtype=order.dtype
    ).view(1, n_patches).expand_as(order)
    ranks.scatter_(-1, order, rank_values)
    target_time_mask = ranks < num_masked.unsqueeze(-1)
    patch_mask = torch.zeros_like(valid_patches)
    batch_indices = torch.arange(batch_size, device=valid_mask.device)
    patch_mask[batch_indices, target_indices] = (
        target_time_mask[:, None, :].expand(-1, 3, -1)
        & valid_patches[batch_indices, target_indices]
    )
    return patch_mask


def _direct_sample_target_indices(valid_mask, generator=None):
    """Choose exactly one valid target epoch for each 20-epoch window."""
    targets = []
    for index in range(valid_mask.size(0)):
        choices = torch.nonzero(
            valid_mask[index].bool(), as_tuple=False
        ).reshape(-1)
        if choices.numel() == 0:
            raise RuntimeError("Generator sample has no valid target epoch.")
        selected = torch.randint(
            choices.numel(),
            (1,),
            device=valid_mask.device,
            generator=generator,
        )
        targets.append(choices[selected].reshape(()))
    return torch.stack(targets)


def _direct_patchify(signal, patch_size):
    patch_size = max(1, int(patch_size))
    signal_length = signal.size(-1)
    n_patches = int(np.ceil(signal_length / patch_size))
    padded_length = n_patches * patch_size
    if padded_length != signal_length:
        signal = F.pad(signal, (0, padded_length - signal_length))
    return signal.contiguous().view(
        *signal.shape[:-1], n_patches, patch_size
    )


def _direct_unpatchify(patches, signal_length):
    return patches.flatten(start_dim=-2)[..., : int(signal_length)]


def _direct_expand_patch_mask(patch_mask, patch_size, signal_length):
    return patch_mask.repeat_interleave(
        max(1, int(patch_size)), dim=-1
    )[..., : int(signal_length)]


def _direct_aligned_log_spectral_patches(signal, settings, sample_mask=None):
    if signal.ndim != 4:
        raise ValueError(f"expected [B,S,C,L], got {tuple(signal.shape)}")
    patch_size = int(settings.patch_size_samples)
    hop_size = int(settings.stft_hop_samples)
    window_size = int(settings.stft_window_samples)
    fft_size = max(window_size, int(settings.stft_fft_size))
    if hop_size != patch_size:
        raise ValueError("Direct generator requires STFT hop == patch size.")
    working = signal.float()
    if sample_mask is not None:
        working = working.masked_fill(sample_mask.bool(), 0.0)
    flat = working.reshape(-1, working.size(-1))
    left = max((window_size - patch_size) // 2, 0)
    right = max(window_size - patch_size - left, 0)
    frames = F.pad(flat, (left, right), mode="constant", value=0.0).unfold(
        -1, window_size, hop_size
    )
    expected = int(np.ceil(signal.size(-1) / patch_size))
    if frames.size(-2) != expected:
        raise RuntimeError(
            f"aligned STFT frames mismatch: {frames.size(-2)} vs {expected}"
        )
    window = torch.hamming_window(
        window_size,
        periodic=True,
        dtype=frames.dtype,
        device=frames.device,
    )
    spectrum = torch.fft.rfft(
        frames * window, n=fft_size, dim=-1
    ).abs()
    frequencies = torch.fft.rfftfreq(
        fft_size,
        d=1.0 / float(settings.sampling_rate),
        device=signal.device,
    )
    selected = (
        (frequencies >= float(settings.frequency_min_hz))
        & (frequencies <= float(settings.frequency_max_hz))
    )
    spectrum = torch.log1p(spectrum[..., selected])
    return spectrum.view(
        *signal.shape[:-1], expected, spectrum.size(-1)
    )


class DirectDualDomainPatchGenerator(nn.Module):
    """SleepGPT-like dual-domain head adapted to the existing event Backbone."""

    def __init__(self, backbone, settings):
        super().__init__()
        self.backbone = backbone
        self.settings = settings
        self.patch_size = int(settings.patch_size_samples)
        self.n_patches = int(
            np.ceil(settings.sample_length / self.patch_size)
        )
        self.encoder_dim = int(settings.eeg_hidden_dim)
        self.decoder_dim = int(settings.decoder_channels)
        frequencies = torch.fft.rfftfreq(
            int(settings.stft_fft_size),
            d=1.0 / float(settings.sampling_rate),
        )
        selected = (
            (frequencies >= float(settings.frequency_min_hz))
            & (frequencies <= float(settings.frequency_max_hz))
        )
        self.frequency_patch_dim = int(selected.sum().item())
        self.visible_encoder = copy.deepcopy(backbone.seq_encoder)
        self.encoder_norm = nn.LayerNorm(self.encoder_dim)
        self.encoder_dropout = nn.Dropout(float(settings.dropout))
        self.frequency_input_projection = nn.Sequential(
            nn.LayerNorm(self.frequency_patch_dim),
            nn.Linear(self.frequency_patch_dim, self.encoder_dim),
            nn.GELU(),
            nn.LayerNorm(self.encoder_dim),
        )
        gate = min(
            max(float(settings.frequency_fusion_gate_init), 1e-4),
            1.0 - 1e-4,
        )
        self.frequency_fusion_gate_logit = nn.Parameter(
            torch.logit(torch.tensor(gate, dtype=torch.float32))
        )
        self.encoder_epoch_pos = nn.Parameter(
            torch.zeros(1, settings.seq_len, self.encoder_dim)
        )
        self.encoder_channel_pos = nn.Parameter(
            torch.zeros(1, 3, self.encoder_dim)
        )
        self.encoder_patch_pos = nn.Parameter(
            torch.zeros(1, self.n_patches, self.encoder_dim)
        )
        self.decoder_embed = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.decoder_mask_token = nn.Parameter(
            torch.zeros(1, 1, self.decoder_dim)
        )
        self.decoder_epoch_pos = nn.Parameter(
            torch.zeros(1, settings.seq_len, self.decoder_dim)
        )
        self.decoder_channel_pos = nn.Parameter(
            torch.zeros(1, 3, self.decoder_dim)
        )
        self.decoder_patch_pos = nn.Parameter(
            torch.zeros(1, self.n_patches, self.decoder_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim,
            nhead=int(settings.decoder_num_heads),
            dim_feedforward=self.decoder_dim * 4,
            dropout=float(settings.decoder_dropout),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=int(settings.decoder_num_layers)
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_dim)
        self.time_reconstruction_head = nn.Linear(
            self.decoder_dim, self.patch_size
        )
        self.frequency_reconstruction_head = nn.Linear(
            self.decoder_dim, self.frequency_patch_dim
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.visible_encoder.parameters():
            parameter.requires_grad = False
        self.encoder_adaptation_enabled = False
        for parameter in [
            self.encoder_epoch_pos,
            self.encoder_channel_pos,
            self.encoder_patch_pos,
            self.decoder_mask_token,
            self.decoder_epoch_pos,
            self.decoder_channel_pos,
            self.decoder_patch_pos,
        ]:
            nn.init.normal_(parameter, std=0.02)

    def enable_last_visible_transformer_layer(self):
        layers = list(self.visible_encoder.layers)
        if not layers:
            raise RuntimeError("Direct generator has no visible Transformer layer.")
        for parameter in layers[-1].parameters():
            parameter.requires_grad = True
        self.encoder_adaptation_enabled = True

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.visible_encoder.eval()
        if mode and self.encoder_adaptation_enabled:
            self.visible_encoder.layers[-1].train(True)
        return self

    def _position_ids(self, flat_indices):
        epoch_ids = torch.div(
            flat_indices, 3 * self.n_patches, rounding_mode="floor"
        )
        remainder = flat_indices % (3 * self.n_patches)
        channel_ids = torch.div(
            remainder, self.n_patches, rounding_mode="floor"
        )
        return epoch_ids, channel_ids, remainder % self.n_patches

    def _encoder_positions(self, epoch_ids, channel_ids, patch_ids):
        return (
            self.encoder_epoch_pos[0, epoch_ids]
            + self.encoder_channel_pos[0, channel_ids]
            + self.encoder_patch_pos[0, patch_ids]
        )

    def _decoder_positions(self, device):
        return (
            self.decoder_epoch_pos[:, :, None, None, :]
            + self.decoder_channel_pos[:, None, :, None, :]
            + self.decoder_patch_pos[:, None, None, :, :]
        ).reshape(1, -1, self.decoder_dim).to(device)

    def _embed_visible(
        self,
        flat_patches,
        flat_frequency,
        visible_indices,
        visible_padding,
    ):
        batch_size, max_visible = visible_indices.shape
        time_index = visible_indices.unsqueeze(-1).expand(
            -1, -1, flat_patches.size(-1)
        )
        frequency_index = visible_indices.unsqueeze(-1).expand(
            -1, -1, flat_frequency.size(-1)
        )
        visible_time = torch.gather(flat_patches, 1, time_index)
        visible_frequency = torch.gather(
            flat_frequency, 1, frequency_index
        )
        epoch_ids, channel_ids, patch_ids = self._position_ids(
            visible_indices
        )
        tokens = flat_patches.new_zeros(
            (batch_size, max_visible, self.encoder_dim)
        )
        active = ~visible_padding
        with torch.no_grad():
            eeg_active = active & (channel_ids == 0)
            if eeg_active.any():
                tokens[eeg_active] = self.backbone.eeg_encoder(
                    visible_time[eeg_active]
                ).to(tokens.dtype)
            eog_active = active & (channel_ids != 0)
            if eog_active.any():
                tokens[eog_active] = self.backbone.eog_encoder(
                    visible_time[eog_active]
                ).to(tokens.dtype)
        frequency_tokens = self.frequency_input_projection(
            visible_frequency
        ).to(tokens.dtype)
        gate = torch.sigmoid(self.frequency_fusion_gate_logit).to(tokens.dtype)
        positions = self._encoder_positions(
            epoch_ids.clamp(0, self.settings.seq_len - 1),
            channel_ids.clamp(0, 2),
            patch_ids.clamp(0, self.n_patches - 1),
        )
        tokens = self.encoder_dropout(
            self.encoder_norm(tokens + gate * frequency_tokens + positions)
        )
        return tokens.masked_fill(visible_padding.unsqueeze(-1), 0.0)

    def forward(self, signal, patch_mask, eog_available, valid_mask):
        batch_size, seq_len, channels, signal_length = signal.shape
        if channels != 3 or seq_len != int(self.settings.seq_len):
            raise ValueError(f"unexpected generator input {tuple(signal.shape)}")
        patches = _direct_patchify(signal, self.patch_size)
        valid_patches = _direct_valid_patch_mask(
            valid_mask, eog_available, signal_length, self.patch_size
        )
        active_mask = patch_mask & valid_patches
        sample_mask = _direct_expand_patch_mask(
            active_mask, self.patch_size, signal_length
        )
        frequency = _direct_aligned_log_spectral_patches(
            signal, self.settings, sample_mask=sample_mask
        )
        flat_time = patches.reshape(batch_size, -1, self.patch_size)
        flat_frequency = frequency.reshape(
            batch_size, -1, self.frequency_patch_dim
        )
        flat_valid = valid_patches.reshape(batch_size, -1)
        flat_visible = flat_valid & ~patch_mask.reshape(batch_size, -1)
        visible_lists = [
            torch.nonzero(flat_visible[i], as_tuple=False).reshape(-1)
            for i in range(batch_size)
        ]
        max_visible = max(indices.numel() for indices in visible_lists)
        if max_visible <= 0:
            raise RuntimeError("Direct generator received no visible patches.")
        visible_indices = torch.zeros(
            (batch_size, max_visible), dtype=torch.long, device=signal.device
        )
        visible_padding = torch.ones(
            (batch_size, max_visible), dtype=torch.bool, device=signal.device
        )
        visible_counts = []
        for i, indices in enumerate(visible_lists):
            count = indices.numel()
            visible_counts.append(count)
            visible_indices[i, :count] = indices
            visible_padding[i, :count] = False
        visible_tokens = self._embed_visible(
            flat_time,
            flat_frequency,
            visible_indices,
            visible_padding,
        )
        encoded = self.visible_encoder(
            visible_tokens, src_key_padding_mask=visible_padding
        )
        positions = self._decoder_positions(signal.device)
        decoder_tokens = (
            self.decoder_mask_token.expand(batch_size, flat_time.size(1), -1)
            + positions
        )
        for i, count in enumerate(visible_counts):
            indices = visible_indices[i, :count]
            decoder_tokens[i, indices] = (
                self.decoder_embed(encoded[i, :count]) + positions[0, indices]
            )
        decoded = self.decoder(
            decoder_tokens, src_key_padding_mask=~flat_valid
        )
        decoded = self.decoder_norm(decoded)
        time_patches = self.time_reconstruction_head(decoded).view(
            batch_size,
            seq_len,
            channels,
            self.n_patches,
            self.patch_size,
        )
        frequency_patches = self.frequency_reconstruction_head(decoded).view(
            batch_size,
            seq_len,
            channels,
            self.n_patches,
            self.frequency_patch_dim,
        )
        return {
            "time_patches": time_patches.masked_fill(
                ~patch_mask.unsqueeze(-1), 0.0
            ),
            "frequency_patches": frequency_patches.masked_fill(
                ~patch_mask.unsqueeze(-1), 0.0
            ),
        }


class HierarchicalEventContextGenerator(nn.Module):
    """30-second CNN features + 20-epoch context + target-epoch decoder."""

    def __init__(self, backbone, settings):
        super().__init__()
        self.backbone = backbone
        self.settings = settings
        self.patch_size = int(settings.patch_size_samples)
        self.n_patches = int(
            np.ceil(settings.sample_length / self.patch_size)
        )
        self.encoder_dim = int(settings.eeg_hidden_dim)
        self.decoder_dim = int(settings.decoder_channels)
        frequencies = torch.fft.rfftfreq(
            int(settings.stft_fft_size),
            d=1.0 / float(settings.sampling_rate),
        )
        selected = (
            (frequencies >= float(settings.frequency_min_hz))
            & (frequencies <= float(settings.frequency_max_hz))
        )
        self.frequency_patch_dim = int(selected.sum().item())
        if self.frequency_patch_dim <= 0:
            raise ValueError("Hierarchical generator has no frequency bins.")

        self.local_time_projection = nn.Sequential(
            nn.Linear(128, self.encoder_dim),
            nn.GELU(),
            nn.LayerNorm(self.encoder_dim),
        )
        self.frequency_input_projection = nn.Sequential(
            nn.LayerNorm(self.frequency_patch_dim),
            nn.Linear(self.frequency_patch_dim, self.encoder_dim),
            nn.GELU(),
            nn.LayerNorm(self.encoder_dim),
        )
        gate = min(
            max(float(settings.frequency_fusion_gate_init), 1e-4),
            1.0 - 1e-4,
        )
        self.frequency_fusion_gate_logit = nn.Parameter(
            torch.logit(torch.tensor(gate, dtype=torch.float32))
        )
        self.local_norm = nn.LayerNorm(self.encoder_dim)
        self.local_dropout = nn.Dropout(float(settings.dropout))
        self.local_to_decoder = nn.Linear(
            self.encoder_dim, self.decoder_dim
        )
        self.context_to_decoder = nn.Linear(
            self.encoder_dim, self.decoder_dim
        )
        self.decoder_mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, self.decoder_dim)
        )
        self.decoder_channel_pos = nn.Parameter(
            torch.zeros(1, 3, 1, self.decoder_dim)
        )
        self.decoder_patch_pos = nn.Parameter(
            torch.zeros(1, 1, self.n_patches, self.decoder_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim,
            nhead=int(settings.decoder_num_heads),
            dim_feedforward=self.decoder_dim * 4,
            dropout=float(settings.decoder_dropout),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=int(settings.decoder_num_layers)
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_dim)
        self.time_reconstruction_head = nn.Linear(
            self.decoder_dim, self.patch_size
        )
        self.frequency_reconstruction_head = nn.Linear(
            self.decoder_dim, self.frequency_patch_dim
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone_adaptation_enabled = False
        for parameter in [
            self.decoder_mask_token,
            self.decoder_channel_pos,
            self.decoder_patch_pos,
        ]:
            nn.init.normal_(parameter, std=0.02)

    @staticmethod
    def _last_cnn_convolution(epoch_encoder):
        convolutions = [
            module
            for module in epoch_encoder.net
            if isinstance(module, nn.Conv1d)
        ]
        if not convolutions:
            raise RuntimeError("EEG epoch encoder has no Conv1d layer.")
        return convolutions[-1]

    def adaptation_parameter_groups(self):
        transformer_layers = list(self.backbone.seq_encoder.layers)
        if not transformer_layers:
            raise RuntimeError("Backbone has no context Transformer layer.")
        transformer_parameters = list(transformer_layers[-1].parameters())
        cnn_parameters = list(
            self._last_cnn_convolution(
                self.backbone.eeg_encoder
            ).parameters()
        ) + list(
            self._last_cnn_convolution(
                self.backbone.eog_encoder
            ).parameters()
        )
        return transformer_parameters, cnn_parameters

    def enable_backbone_adaptation(self):
        transformer_parameters, cnn_parameters = (
            self.adaptation_parameter_groups()
        )
        for parameter in transformer_parameters + cnn_parameters:
            parameter.requires_grad = True
        self.backbone_adaptation_enabled = True

    def train(self, mode=True):
        super().train(mode)
        # Frozen pretrained modules and every BatchNorm keep their saved
        # running statistics. Only the last context layer may use training
        # Dropout after the restricted adaptation stage begins.
        self.backbone.eval()
        if mode and self.backbone_adaptation_enabled:
            self.backbone.seq_encoder.layers[-1].train(True)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        return self

    def _encode_channel(self, signal, epoch_encoder):
        batch_size, seq_len, signal_length = signal.shape
        flat = signal.reshape(batch_size * seq_len, signal_length)
        temporal = epoch_encoder.forward_temporal(flat)
        epoch_embedding = epoch_encoder.pool_temporal(temporal)
        local = F.adaptive_avg_pool1d(
            temporal, self.n_patches
        ).transpose(1, 2)
        local = self.local_time_projection(local)
        return (
            epoch_embedding.view(batch_size, seq_len, -1),
            local.view(
                batch_size,
                seq_len,
                self.n_patches,
                self.encoder_dim,
            ),
        )

    @staticmethod
    def _gather_target(tensor, target_indices):
        batch_indices = torch.arange(
            tensor.size(0), device=tensor.device
        )
        return tensor[batch_indices, target_indices]

    def forward(
        self,
        signal,
        patch_mask,
        eog_available,
        valid_mask,
        target_indices,
    ):
        batch_size, seq_len, channels, signal_length = signal.shape
        if channels != 3 or seq_len != int(self.settings.seq_len):
            raise ValueError(
                f"unexpected hierarchical generator input {tuple(signal.shape)}"
            )
        target_indices = torch.as_tensor(
            target_indices, dtype=torch.long, device=signal.device
        ).reshape(-1)
        if target_indices.numel() != batch_size:
            raise ValueError("One target epoch is required per batch sample.")
        valid_patches = _direct_valid_patch_mask(
            valid_mask,
            eog_available,
            signal_length,
            self.patch_size,
        )
        active_mask = patch_mask & valid_patches
        sample_mask = _direct_expand_patch_mask(
            active_mask, self.patch_size, signal_length
        )
        masked_signal = signal.masked_fill(sample_mask, 0.0)

        eeg_epoch, eeg_local = self._encode_channel(
            masked_signal[:, :, 0, :], self.backbone.eeg_encoder
        )
        loc_epoch, loc_local = self._encode_channel(
            masked_signal[:, :, 1, :], self.backbone.eog_encoder
        )
        roc_epoch, roc_local = self._encode_channel(
            masked_signal[:, :, 2, :], self.backbone.eog_encoder
        )
        availability = eog_available.to(
            device=signal.device, dtype=signal.dtype
        )
        loc_scale = availability[:, 0, None, None]
        roc_scale = availability[:, 1, None, None]
        loc_epoch = loc_epoch * loc_scale
        roc_epoch = roc_epoch * roc_scale
        loc_local = loc_local * loc_scale.unsqueeze(-1)
        roc_local = roc_local * roc_scale.unsqueeze(-1)
        epoch_embeddings = self.backbone.fusion(
            torch.cat([eeg_epoch, loc_epoch, roc_epoch], dim=-1)
        )
        epoch_context = self.backbone.contextualize(
            epoch_embeddings, valid_mask=valid_mask
        )
        local_tokens = torch.stack(
            [eeg_local, loc_local, roc_local], dim=2
        )

        target_local = self._gather_target(
            local_tokens, target_indices
        )
        target_context = self._gather_target(
            epoch_context, target_indices
        )
        target_mask = self._gather_target(
            active_mask, target_indices
        )
        target_valid = self._gather_target(
            valid_patches, target_indices
        )
        # Frequency tokens are needed only for the decoded target epoch.
        # Avoid calculating 60 full STFT streams when the other 19 epochs
        # contribute context exclusively through the pretrained CNN/Transformer.
        target_signal = self._gather_target(signal, target_indices)
        target_sample_mask = _direct_expand_patch_mask(
            target_mask,
            self.patch_size,
            signal_length,
        )
        target_frequency = _direct_aligned_log_spectral_patches(
            target_signal.unsqueeze(1),
            self.settings,
            sample_mask=target_sample_mask.unsqueeze(1),
        )[:, 0]
        frequency_tokens = self.frequency_input_projection(
            target_frequency
        ).to(target_local.dtype)
        gate = torch.sigmoid(
            self.frequency_fusion_gate_logit
        ).to(target_local.dtype)
        visible_local = self.local_dropout(
            self.local_norm(target_local + gate * frequency_tokens)
        )
        visible_decoder = self.local_to_decoder(visible_local)
        context_decoder = self.context_to_decoder(
            target_context
        )[:, None, None, :]
        positions = self.decoder_channel_pos + self.decoder_patch_pos
        visible_decoder = visible_decoder + context_decoder + positions
        masked_decoder = (
            self.decoder_mask_token + context_decoder + positions
        )
        decoder_input = torch.where(
            target_mask.unsqueeze(-1),
            masked_decoder,
            visible_decoder,
        )
        decoder_input = decoder_input.reshape(
            batch_size, 3 * self.n_patches, self.decoder_dim
        )
        target_padding = ~target_valid.reshape(
            batch_size, 3 * self.n_patches
        )
        decoded = self.decoder(
            decoder_input, src_key_padding_mask=target_padding
        )
        decoded = self.decoder_norm(decoded).view(
            batch_size, 3, self.n_patches, self.decoder_dim
        )
        time_patches = self.time_reconstruction_head(decoded)
        frequency_patches = self.frequency_reconstruction_head(decoded)
        return {
            "time_patches": time_patches.masked_fill(
                ~target_mask.unsqueeze(-1), 0.0
            ),
            "frequency_patches": frequency_patches.masked_fill(
                ~target_mask.unsqueeze(-1), 0.0
            ),
            "target_mask": target_mask,
            "target_valid": target_valid,
        }


class TCNResidualBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid_mask=None):
        residual = x
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = y.transpose(1, 2)
        y = self.drop(y)
        y = self.norm(residual + y)
        y = self.act(y)
        if valid_mask is not None:
            y = y * valid_mask.unsqueeze(-1).to(dtype=y.dtype, device=y.device)
        return y


class StageTCNHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_classes,
        channels=None,
        num_layers=3,
        kernel_size=3,
        dropout=0.25,
    ):
        super().__init__()
        channels = int(hidden_dim if channels is None else channels)
        if channels <= 0:
            raise ValueError(f"TCN channels must be positive, got {channels}.")

        if channels == int(hidden_dim):
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(hidden_dim, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
            )
        self.blocks = nn.ModuleList([
            TCNResidualBlock(
                channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        self.out = nn.Linear(channels, num_classes)

    def extract_features(self, x, valid_mask=None):
        """Return the final TCN representation immediately before logits."""
        x = self.input_proj(x)
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x, valid_mask=valid_mask)
        return x

    def forward(self, x, valid_mask=None):
        x = self.extract_features(x, valid_mask=valid_mask)
        return self.out(x)


class RelativeSpectralEncoder(nn.Module):
    """Encode per-channel relative band power without absolute amplitude."""
    def __init__(
        self,
        out_dim,
        sampling_rate=100.0,
        bands=None,
        hidden_dim=128,
        dropout=0.1,
        num_channels=3,
    ):
        super().__init__()
        self.sampling_rate = float(sampling_rate)
        self.bands = list(bands or [
            (0.5, 4.0),
            (4.0, 8.0),
            (8.0, 12.0),
            (12.0, 16.0),
            (16.0, 30.0),
        ])
        self.num_channels = int(num_channels)
        in_dim = self.num_channels * len(self.bands)
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )

    def relative_band_power(self, eeg, eog_available=None):
        if eeg.ndim != 4 or eeg.size(2) != self.num_channels:
            raise ValueError(
                f"expected [B,T,{self.num_channels},L] EEG/EOG input, got {tuple(eeg.shape)}"
            )

        # FFT remains float32 under AMP. Absolute spectral scale cancels when
        # every band is divided by total 0.5-30 Hz power.
        signal = eeg.float()
        spectrum = torch.fft.rfft(signal, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        freqs = torch.fft.rfftfreq(
            signal.size(-1),
            d=1.0 / self.sampling_rate,
            device=signal.device,
        )

        low_total = min(float(low) for low, _ in self.bands)
        high_total = max(float(high) for _, high in self.bands)
        total_mask = (freqs >= low_total) & (freqs <= high_total)
        total_power = power[..., total_mask].sum(dim=-1, keepdim=True).clamp_min(1e-8)

        relative_bands = []
        for band_idx, (low, high) in enumerate(self.bands):
            if band_idx == len(self.bands) - 1:
                band_mask = (freqs >= float(low)) & (freqs <= float(high))
            else:
                band_mask = (freqs >= float(low)) & (freqs < float(high))
            band_power = power[..., band_mask].sum(dim=-1, keepdim=True)
            relative_bands.append(band_power / total_power)
        relative = torch.cat(relative_bands, dim=-1)

        # Missing EOG channels must contribute exactly zero spectral features.
        if eog_available is not None:
            availability = torch.cat([
                torch.ones(
                    eog_available.size(0), 1,
                    dtype=relative.dtype,
                    device=relative.device,
                ),
                eog_available.to(dtype=relative.dtype, device=relative.device),
            ], dim=1)
            relative = relative * availability[:, None, :, None]

        # Log relative power improves separation of weak bands while remaining
        # scale invariant. Missing channels remain zero after masking.
        log_relative = torch.log(relative.clamp_min(1e-8))
        if eog_available is not None:
            log_relative = log_relative * availability[:, None, :, None]
        return log_relative.flatten(start_dim=2)

    def forward(self, eeg, eog_available=None, valid_mask=None):
        features = self.relative_band_power(eeg, eog_available=eog_available)
        features = self.encoder(features)
        if valid_mask is not None:
            features = features * valid_mask.unsqueeze(-1).to(features.dtype)
        return features


class FixedSTFTSpectralEncoder(nn.Module):
    """
    Encode frequency information recomputed from the waveform actually seen by
    the classifier. Real, synthetic and held-out test epochs therefore follow
    exactly the same spectral path; an independently generated spectrum can
    never become a synthetic-data shortcut.
    """
    def __init__(
        self,
        out_dim,
        sampling_rate=100.0,
        window_samples=200,
        hop_samples=100,
        fft_size=256,
        frequency_min_hz=0.5,
        frequency_max_hz=30.0,
        hidden_dim=128,
        dropout=0.1,
        num_channels=3,
    ):
        super().__init__()
        self.sampling_rate = float(sampling_rate)
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self.fft_size = int(fft_size)
        self.frequency_min_hz = float(frequency_min_hz)
        self.frequency_max_hz = float(frequency_max_hz)
        self.num_channels = int(num_channels)
        frequencies = torch.fft.rfftfreq(
            self.fft_size,
            d=1.0 / self.sampling_rate,
        )
        selected = (
            (frequencies >= self.frequency_min_hz)
            & (frequencies <= self.frequency_max_hz)
        )
        selected_indices = torch.nonzero(selected, as_tuple=False).reshape(-1)
        if selected_indices.numel() == 0:
            raise ValueError("Fixed STFT branch selected no frequency bins.")
        self.first_frequency_bin = int(selected_indices[0].item())
        self.last_frequency_bin = int(selected_indices[-1].item())
        self.frequency_bins = int(selected.sum().item())
        # Mean and standard deviation across aligned STFT frames retain both
        # spectral composition and its within-epoch temporal variability.
        in_dim = self.num_channels * self.frequency_bins * 2
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )

    def spectral_statistics(self, eeg, eog_available=None):
        if eeg.ndim != 4 or eeg.size(2) != self.num_channels:
            raise ValueError(
                f"expected [B,T,{self.num_channels},L], got {tuple(eeg.shape)}"
            )
        batch_size, seq_len, channels, signal_length = eeg.shape
        flat = eeg.float().reshape(-1, signal_length)
        window = torch.hamming_window(
            self.window_samples,
            periodic=True,
            dtype=flat.dtype,
            device=flat.device,
        )
        spectrum = torch.stft(
            flat,
            n_fft=self.fft_size,
            hop_length=self.hop_samples,
            win_length=self.window_samples,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs()
        log_spectrum = torch.log1p(
            spectrum[
                :,
                self.first_frequency_bin:self.last_frequency_bin + 1,
                :,
            ]
        )
        mean = log_spectrum.mean(dim=-1)
        std = log_spectrum.std(dim=-1, unbiased=False)
        statistics = torch.cat([mean, std], dim=-1).view(
            batch_size,
            seq_len,
            channels,
            self.frequency_bins * 2,
        )

        if eog_available is not None:
            availability = torch.cat([
                torch.ones(
                    batch_size,
                    1,
                    dtype=statistics.dtype,
                    device=statistics.device,
                ),
                eog_available.to(
                    dtype=statistics.dtype,
                    device=statistics.device,
                ),
            ], dim=1)
            statistics = statistics * availability[:, None, :, None]
        return statistics.flatten(start_dim=2)

    def forward(self, eeg, eog_available=None, valid_mask=None):
        features = self.spectral_statistics(
            eeg,
            eog_available=eog_available,
        )
        features = self.encoder(features)
        if valid_mask is not None:
            features = features * valid_mask.unsqueeze(-1).to(features.dtype)
        return features


class MultiTaskSleepModel(nn.Module):
    def __init__(self, backbone, num_stage_classes=5, num_disorder_classes=2, num_slow_wave_classes=2, cfg=None):
        super().__init__()
        self.backbone = backbone
        h = backbone.hidden_dim

        self.use_stage_relative_spectral_branch = bool(
            getattr(cfg, "use_stage_relative_spectral_branch", False)
        ) if cfg is not None else False
        if self.use_stage_relative_spectral_branch:
            spectral_mode = str(
                getattr(cfg, "stage_spectral_mode", "relative_band_power")
            ).lower()
            if spectral_mode == "fixed_stft_log":
                self.stage_spectral_encoder = FixedSTFTSpectralEncoder(
                    out_dim=h,
                    sampling_rate=float(
                        getattr(cfg, "stage_spectral_sampling_rate", 100.0)
                    ),
                    window_samples=int(
                        getattr(cfg, "stage_stft_window_samples", 200)
                    ),
                    hop_samples=int(
                        getattr(cfg, "stage_stft_hop_samples", 100)
                    ),
                    fft_size=int(
                        getattr(cfg, "stage_stft_fft_size", 256)
                    ),
                    frequency_min_hz=float(
                        getattr(cfg, "stage_stft_frequency_min_hz", 0.5)
                    ),
                    frequency_max_hz=float(
                        getattr(cfg, "stage_stft_frequency_max_hz", 30.0)
                    ),
                    hidden_dim=int(
                        getattr(cfg, "stage_spectral_hidden_dim", 128)
                    ),
                    dropout=float(
                        getattr(cfg, "stage_spectral_dropout", 0.1)
                    ),
                )
            elif spectral_mode == "relative_band_power":
                spectral_bands = getattr(cfg, "stage_spectral_bands", None)
                self.stage_spectral_encoder = RelativeSpectralEncoder(
                    out_dim=h,
                    sampling_rate=float(
                        getattr(cfg, "stage_spectral_sampling_rate", 100.0)
                    ),
                    bands=spectral_bands,
                    hidden_dim=int(
                        getattr(cfg, "stage_spectral_hidden_dim", 128)
                    ),
                    dropout=float(
                        getattr(cfg, "stage_spectral_dropout", 0.1)
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown stage_spectral_mode={spectral_mode!r}."
                )
            gate_init = float(
                getattr(cfg, "stage_spectral_gate_init", 0.1)
            )
            gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
            self.stage_spectral_gate_logit = nn.Parameter(
                torch.logit(torch.tensor(gate_init, dtype=torch.float32))
            )
        else:
            self.stage_spectral_encoder = None
            self.register_parameter("stage_spectral_gate_logit", None)

        stage_head_type = str(getattr(cfg, "stage_head_type", "linear")).lower() if cfg is not None else "linear"
        self.stage_head_type = stage_head_type
        if stage_head_type == "mlp":
            stage_hidden = int(getattr(cfg, "stage_mlp_hidden_dim", h))
            stage_dropout = float(getattr(cfg, "stage_mlp_dropout", 0.3))
            self.stage_head = nn.Sequential(
                nn.Linear(h, stage_hidden),
                nn.LayerNorm(stage_hidden),
                nn.GELU(),
                nn.Dropout(stage_dropout),
                nn.Linear(stage_hidden, num_stage_classes),
            )
        elif stage_head_type == "tcn":
            self.stage_head = StageTCNHead(
                hidden_dim=h,
                num_classes=num_stage_classes,
                channels=int(getattr(cfg, "stage_tcn_channels", h)),
                num_layers=int(getattr(cfg, "stage_tcn_layers", 3)),
                kernel_size=int(getattr(cfg, "stage_tcn_kernel_size", 3)),
                dropout=float(getattr(cfg, "stage_tcn_dropout", 0.25)),
            )
        else:
            self.stage_head = nn.Linear(h, num_stage_classes)

        self.slow_wave_head = nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(h, num_slow_wave_classes)
        )

        self.disorder_head = nn.Sequential(
            nn.Linear(h, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_disorder_classes)
        )

    def extract_stage_features(self, eeg, eog_available=None, valid_mask=None):
        features = self.backbone(
            eeg,
            eog_available=eog_available,
            valid_mask=valid_mask,
        )
        if self.stage_spectral_encoder is not None:
            spectral_features = self.stage_spectral_encoder(
                eeg,
                eog_available=eog_available,
                valid_mask=valid_mask,
            ).to(dtype=features.dtype)
            gate = torch.sigmoid(self.stage_spectral_gate_logit).to(features.dtype)
            features = features + gate * spectral_features
        return features

    def forward(self, eeg, task_name, eog_available=None, valid_mask=None):
        if task_name == "sleep_stage_classification":
            x = self.extract_stage_features(
                eeg,
                eog_available=eog_available,
                valid_mask=valid_mask,
            )
            if self.stage_head_type == "tcn":
                return self.stage_head(x, valid_mask=valid_mask)
            return self.stage_head(x)

        x = self.backbone(eeg, eog_available=eog_available, valid_mask=valid_mask)

        if task_name == "slow_wave_detection":
            return self.slow_wave_head(x)

        if task_name == "sleep_disorder_classification":
            if valid_mask is None:
                pooled = x.mean(dim=1)
            else:
                mask = valid_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return self.disorder_head(pooled)

        raise ValueError(f"未知任务: {task_name}")


# =========================
# 训练与评估
# =========================
class SpindleConvResidualBlock(nn.Module):
    """在单个30秒epoch内部提取局部时间模式的残差卷积块。

    dilation依次为1、2、4，在不显著增加参数量的情况下扩大感受野。
    """

    def __init__(self, channels, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=5,
            padding=2 * dilation, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=5,
            padding=2 * dilation, dilation=dilation,
        )
        groups = max(1, min(16, channels // 8))
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(self.norm2(self.conv2(x)))
        return F.gelu(x + residual)


class SpindleTemporalDetectionModel(nn.Module):
    """上游预训练表征 + 下游逐采样点纺锤波检测头。

    输入：
        eeg: [B, S, 3, 3000]
             B为batch，S为连续epoch数，3为EEG/LOC/ROC。

    两种表征：
        local：中央EEG CNN池化前特征，保留30秒内的时间位置；
        contextual：EEG+EOG融合后经过Transformer得到的epoch上下文。

    将contextual沿单个epoch的时间轴广播，与local拼接后进行卷积解码，
    最终上采样回3000点，输出[B, S, 3000]逐点logits。
    """

    def __init__(self, backbone, head_channels=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.local_proj = nn.Conv1d(128, head_channels, kernel_size=1)
        self.context_proj = nn.Sequential(
            nn.Linear(backbone.hidden_dim, head_channels),
            nn.LayerNorm(head_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(head_channels * 2, head_channels, kernel_size=1),
            nn.GroupNorm(8 if head_channels % 8 == 0 else 1, head_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            SpindleConvResidualBlock(
                head_channels,
                dilation=2 ** layer_idx,
                dropout=dropout,
            )
            for layer_idx in range(int(num_layers))
        ])
        self.out = nn.Conv1d(head_channels, 1, kernel_size=1)

    def head_parameters(self):
        """只返回新下游检测头参数，不包含预训练backbone。"""
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                yield parameter

    def forward(self, eeg, eog_available=None, valid_mask=None):
        bsz, seq_len, _, signal_len = eeg.shape
        contextual, local = self.backbone.forward_with_eeg_temporal(
            eeg,
            eog_available=eog_available,
            valid_mask=valid_mask,
        )
        local_len = local.size(-1)
        local = local.reshape(bsz * seq_len, local.size(2), local_len)
        local = self.local_proj(local)
        context = self.context_proj(contextual).reshape(
            bsz * seq_len, -1, 1
        ).expand(-1, -1, local_len)
        x = self.fuse(torch.cat([local, context], dim=1))
        for block in self.blocks:
            x = block(x)
        logits = self.out(x)
        logits = F.interpolate(
            logits, size=signal_len, mode="linear", align_corners=False
        )
        logits = logits.view(bsz, seq_len, signal_len)
        if valid_mask is not None:
            logits = logits.masked_fill(valid_mask.unsqueeze(-1) == 0, 0.0)
        return logits


def set_text_encoder_trainable(model, trainable: bool, unfreeze_last_n_layers=None):
    """
    纯 EEG 主线中不再存在 text_encoder。
    保留函数名兼容旧调用方式，但内部不执行任何操作。
    """
    return None


def set_backbone_trainable(model, trainable):
    for param in model.backbone.parameters():
        param.requires_grad = bool(trainable)


def configure_stage_backbone_finetuning(model, cfg):
    """Apply the restricted sleep-stage fine-tuning scope after warm-up."""
    strategy = str(
        getattr(cfg, "stage_backbone_unfreeze_strategy", "last_transformer_layer")
    ).lower()
    set_backbone_trainable(model, False)

    if strategy == "last_transformer_layer":
        layers = model.backbone.seq_encoder.layers
        if len(layers) == 0:
            raise ValueError("Cannot unfreeze the last Transformer layer: no layers found.")
        for param in layers[-1].parameters():
            param.requires_grad = True
    elif strategy == "full_backbone":
        set_backbone_trainable(model, True)
    else:
        raise ValueError(f"Unknown stage_backbone_unfreeze_strategy: {strategy}")

    trainable = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.backbone.parameters())
    print(
        f"[Stage Finetune Scope] strategy={strategy} | "
        f"trainable_backbone_params={trainable}/{total} "
        f"({100.0 * trainable / max(total, 1):.2f}%)"
    )


def enter_second_transformer_finetuning_stage(model, optimizer, cfg):
    """Continuously unfreeze the second-last Transformer layer.

    Model weights and existing optimizer state are preserved. The existing
    last-layer parameter group gets a lower LR, while the newly trainable lower
    layer is appended as a fresh optimizer parameter group.
    """
    layers = model.backbone.seq_encoder.layers
    if len(layers) < 2:
        raise ValueError(
            "Second Transformer fine-tuning stage requires at least two layers."
        )

    # Keep CNN/EOG/fusion/position parameters frozen and train exactly the last
    # two TransformerEncoderLayers.
    set_backbone_trainable(model, False)
    for layer in layers[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    last_layer_param_ids = {id(p) for p in layers[-1].parameters()}
    last_layer_lr = float(cfg.lr_last_transformer_stage2)
    last_group_found = False
    for group in optimizer.param_groups:
        if any(id(p) in last_layer_param_ids for p in group["params"]):
            group["lr"] = last_layer_lr
            group["name"] = "last_transformer_stage2"
            last_group_found = True
    if not last_group_found:
        raise RuntimeError(
            "The last Transformer layer is missing from the existing optimizer."
        )

    existing_param_ids = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    second_last_params = [
        p for p in layers[-2].parameters() if id(p) not in existing_param_ids
    ]
    if not second_last_params:
        raise RuntimeError(
            "The second-last Transformer layer is already in the optimizer."
        )
    optimizer.add_param_group({
        "params": second_last_params,
        "lr": float(cfg.lr_second_last_transformer),
        "weight_decay": float(cfg.weight_decay),
        "name": "second_last_transformer_stage2",
    })

    trainable = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.backbone.parameters())
    print(
        "[Second Transformer Unfreeze] "
        f"trainable_backbone_params={trainable}/{total} "
        f"({100.0 * trainable / max(total, 1):.2f}%) | "
        f"last_layer_lr={last_layer_lr:.2e} | "
        f"second_last_layer_lr={float(cfg.lr_second_last_transformer):.2e} | "
        "optimizer_state=preserved"
    )


def keep_frozen_backbone_modules_in_eval_mode(model):
    """Prevent frozen BatchNorm/dropout modules from drifting during training."""
    backbone = model.backbone
    modules = [backbone.eeg_encoder, backbone.eog_encoder, backbone.fusion]
    modules.extend(list(backbone.seq_encoder.layers))
    for module in modules:
        params = list(module.parameters())
        if params and not any(p.requires_grad for p in params):
            module.eval()

    if not backbone.pos_embed.requires_grad:
        backbone.drop.eval()


def adapt_batchnorm_to_test_subject(model, loader, device):
    """Recompute EEG/EOG CNN BatchNorm statistics from unlabeled test data.

    Only BatchNorm running buffers are changed. Dropout stays disabled, model
    parameters are not updated, and labels are never read.
    """
    encoders = [model.backbone.eeg_encoder, model.backbone.eog_encoder]
    bn_modules = [
        module
        for encoder in encoders
        for module in encoder.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not bn_modules:
        print("[AdaBN] No BatchNorm modules found; calibration skipped.")
        return 0

    model.eval()
    original_momenta = {module: module.momentum for module in bn_modules}
    for module in bn_modules:
        module.reset_running_stats()
        module.momentum = None  # cumulative moving average across target batches
        module.train()

    n_batches = 0
    n_eeg_epochs = 0
    n_eog_epochs = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="AdaBN calibrate", leave=False):
            eeg = batch["eeg"].to(device)
            valid_mask = batch["valid_mask"].to(device).bool()
            if not bool(valid_mask.any()):
                continue

            # [B, T, 3, L] -> only real (non-padding) epochs.
            valid_signals = eeg[valid_mask]
            model.backbone.eeg_encoder(valid_signals[:, 0, :])
            n_eeg_epochs += int(valid_signals.size(0))

            availability = batch["eog_available"].to(device).bool()
            availability = availability.unsqueeze(1).expand(
                -1, valid_mask.size(1), -1
            )[valid_mask]
            if bool(availability[:, 0].any()):
                model.backbone.eog_encoder(
                    valid_signals[availability[:, 0], 1, :]
                )
                n_eog_epochs += int(availability[:, 0].sum().item())
            if bool(availability[:, 1].any()):
                model.backbone.eog_encoder(
                    valid_signals[availability[:, 1], 2, :]
                )
                n_eog_epochs += int(availability[:, 1].sum().item())
            n_batches += 1

    model.eval()
    for module, momentum in original_momenta.items():
        module.momentum = momentum

    print(
        f"[AdaBN] calibrated {len(bn_modules)} BatchNorm layers | "
        f"loader_batches={n_batches} | EEG epochs={n_eeg_epochs} | "
        f"available EOG channel-epochs={n_eog_epochs} | labels_used=False | "
        "weights_updated=False"
    )
    return n_batches


def compute_stage_class_weights(train_samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for sample in train_samples:
        labels = np.asarray(sample.get("labels", []), dtype=np.int64).reshape(-1)
        valid = labels[(labels >= 0) & (labels < num_classes)]
        if len(valid) == 0:
            continue
        counts += np.bincount(valid, minlength=num_classes)[:num_classes]

    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if not np.any(present):
        return torch.ones(num_classes, dtype=torch.float32)

    weights[present] = 1.0 / np.sqrt(counts[present])
    mean_present = float(weights[present].mean())
    if mean_present > 0:
        weights[present] /= mean_present

    return torch.tensor(weights, dtype=torch.float32)


class FoldMaskedEpochPredictor(nn.Module):
    """Predict fixed-teacher epoch embeddings from masked student context."""
    def __init__(self, backbone, hidden_dim=256):
        super().__init__()
        self.backbone = backbone
        feature_dim = int(backbone.hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), feature_dim),
        )

    def forward(self, eeg, eog_available, valid_mask, epoch_mask):
        contextual = self.backbone(
            eeg,
            eog_available=eog_available,
            valid_mask=valid_mask,
            epoch_mask=epoch_mask,
        )
        return self.predictor(contextual)


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _split_fold_ssl_samples(train_samples, cfg, fold_idx):
    subject_ids = sorted({str(sample["subject_id"]) for sample in train_samples})
    if len(subject_ids) < 2:
        raise ValueError("fold SSL adaptation requires at least two training subjects")

    rng = np.random.RandomState(cfg.seed + 30000 + int(fold_idx))
    shuffled = np.asarray(subject_ids, dtype=object)
    rng.shuffle(shuffled)
    valid_ratio = float(getattr(cfg, "fold_ssl_valid_subject_ratio", 0.1))
    n_valid = max(1, int(round(len(shuffled) * valid_ratio)))
    n_valid = min(n_valid, len(shuffled) - 1)
    valid_subjects = set(str(x) for x in shuffled[:n_valid].tolist())
    train_subjects = set(str(x) for x in shuffled[n_valid:].tolist())
    ssl_train = [
        sample for sample in train_samples
        if str(sample["subject_id"]) in train_subjects
    ]
    ssl_valid = [
        sample for sample in train_samples
        if str(sample["subject_id"]) in valid_subjects
    ]
    return ssl_train, ssl_valid, sorted(train_subjects), sorted(valid_subjects)


def _sample_valid_epoch_mask(valid_mask, mask_ratio, generator):
    epoch_mask = (
        torch.rand(
            valid_mask.shape,
            device=valid_mask.device,
            generator=generator,
        ) < float(mask_ratio)
    ) & valid_mask.bool()

    # Every sequence contributes at least one valid reconstruction target.
    for batch_idx in range(valid_mask.size(0)):
        if bool(epoch_mask[batch_idx].any()):
            continue
        valid_indices = torch.nonzero(
            valid_mask[batch_idx] > 0,
            as_tuple=False,
        ).flatten()
        if valid_indices.numel() == 0:
            continue
        choice = torch.randint(
            valid_indices.numel(),
            (1,),
            device=valid_mask.device,
            generator=generator,
        )
        epoch_mask[batch_idx, valid_indices[choice]] = True
    return epoch_mask


def _run_fold_ssl_epoch(
    student,
    teacher,
    loader,
    device,
    cfg,
    optimizer=None,
    scaler=None,
    mask_seed=0,
):
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(mask_seed))
    use_amp = bool(cfg.use_amp and device.type == "cuda")
    total_loss = 0.0
    total_masked = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(
            loader,
            desc="Fold SSL train" if training else "Fold SSL valid",
            leave=False,
        ):
            eeg = batch["eeg"].to(device)
            eog_available = batch["eog_available"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            epoch_mask = _sample_valid_epoch_mask(
                valid_mask,
                cfg.fold_ssl_mask_ratio,
                generator,
            )
            n_masked = int(epoch_mask.sum().item())
            if n_masked == 0:
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_targets = teacher.encode_epochs(
                    eeg,
                    eog_available=eog_available,
                )

            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = student(
                    eeg,
                    eog_available,
                    valid_mask,
                    epoch_mask,
                )
                predicted_masked = F.normalize(
                    predictions[epoch_mask].float(),
                    dim=-1,
                )
                target_masked = F.normalize(
                    teacher_targets[epoch_mask].float(),
                    dim=-1,
                )
                loss = 1.0 - (
                    predicted_masked * target_masked
                ).sum(dim=-1).mean()

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    float(getattr(cfg, "fold_ssl_grad_clip_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.detach().item()) * n_masked
            total_masked += n_masked

    if total_masked == 0:
        raise RuntimeError("fold SSL adaptation produced no masked valid epochs")
    return total_loss / total_masked, total_masked


def adapt_backbone_on_fold_train_subjects(
    backbone,
    train_samples,
    cfg,
    device,
    fold_idx,
):
    """Domain-adapt a backbone using unlabeled data from this fold's 19 subjects."""
    ssl_train, ssl_valid, train_subjects, valid_subjects = _split_fold_ssl_samples(
        train_samples,
        cfg,
        fold_idx,
    )
    all_subjects = sorted({str(sample["subject_id"]) for sample in train_samples})
    print(
        f"[Fold SSL] Fold {fold_idx} | all_train_subjects={len(all_subjects)} | "
        f"ssl_train_subjects={len(train_subjects)} | "
        f"ssl_valid_subjects={len(valid_subjects)} | labels_used=False | "
        "outer_test_used=False"
    )
    print(f"[Fold SSL] validation subjects: {valid_subjects}")

    batch_size = max(1, int(getattr(cfg, "fold_ssl_batch_size", 4)))
    train_loader = DataLoader(
        SleepTextWindowDataset(
            ssl_train, None, None, "sleep_stage_classification"
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        SleepTextWindowDataset(
            ssl_valid, None, None, "sleep_stage_classification"
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    backbone = backbone.to(device)
    teacher = copy.deepcopy(backbone).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student = FoldMaskedEpochPredictor(
        backbone,
        hidden_dim=int(getattr(cfg, "fold_ssl_predictor_hidden_dim", 256)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(getattr(cfg, "fold_ssl_learning_rate", 1e-5)),
        weight_decay=float(getattr(cfg, "fold_ssl_weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(cfg.use_amp and device.type == "cuda"),
    )

    best_valid = float("inf")
    best_epoch = 0
    best_backbone_state = None
    patience_counter = 0
    history = []
    max_epochs = max(1, int(getattr(cfg, "fold_ssl_epochs", 10)))
    patience = max(1, int(getattr(cfg, "fold_ssl_patience", 3)))
    min_delta = float(getattr(cfg, "fold_ssl_min_delta", 1e-4))

    for epoch in range(max_epochs):
        train_loss, train_masked = _run_fold_ssl_epoch(
            student,
            teacher,
            train_loader,
            device,
            cfg,
            optimizer=optimizer,
            scaler=scaler,
            mask_seed=cfg.seed + fold_idx * 1000 + epoch,
        )
        valid_loss, valid_masked = _run_fold_ssl_epoch(
            student,
            teacher,
            valid_loader,
            device,
            cfg,
            optimizer=None,
            scaler=None,
            mask_seed=cfg.seed + fold_idx * 1000 + 99999,
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
            "train_masked_epochs": int(train_masked),
            "valid_masked_epochs": int(valid_masked),
        })
        print(
            f"[Fold SSL] Fold {fold_idx} | epoch {epoch + 1}/{max_epochs} | "
            f"train_cosine_loss={train_loss:.6f} | "
            f"valid_cosine_loss={valid_loss:.6f}"
        )

        if valid_loss < best_valid - min_delta:
            best_valid = float(valid_loss)
            best_epoch = epoch + 1
            best_backbone_state = {
                key: value.detach().cpu().clone()
                for key, value in student.backbone.state_dict().items()
            }
            patience_counter = 0
            print(
                f"[Fold SSL New Best] Fold {fold_idx} | epoch={best_epoch} | "
                f"valid_loss={best_valid:.6f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[Fold SSL Early Stop] Fold {fold_idx} | "
                    f"epoch={epoch + 1} | best_epoch={best_epoch}"
                )
                break

    if best_backbone_state is None:
        raise RuntimeError("fold SSL adaptation did not produce a checkpoint")
    backbone.load_state_dict(best_backbone_state, strict=True)
    metadata = {
        "enabled": True,
        "labels_used": False,
        "outer_test_used": False,
        "all_train_subjects": all_subjects,
        "ssl_train_subjects": train_subjects,
        "ssl_valid_subjects": valid_subjects,
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid),
        "history": history,
    }

    del teacher, student, optimizer, scaler, train_loader, valid_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return backbone, metadata


def build_optimizer(model, task_name, cfg):
    if task_name == "sleep_stage_classification":
        head_params = list(model.stage_head.parameters())
        if getattr(model, "stage_spectral_encoder", None) is not None:
            head_params.extend(model.stage_spectral_encoder.parameters())
            head_params.append(model.stage_spectral_gate_logit)
    elif task_name == "slow_wave_detection":
        head_params = list(model.slow_wave_head.parameters())
    else:
        head_params = list(model.disorder_head.parameters())

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({
            "params": backbone_params,
            "lr": cfg.lr_backbone,
            "name": "backbone",
        })

    param_groups.append({
        "params": head_params,
        "lr": cfg.lr_head,
        "name": "head",
    })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    return optimizer


def train_one_epoch(model, loader, optimizer, device, task_name, cfg, class_weights=None):
    model.train()
    if task_name == "sleep_stage_classification":
        keep_frozen_backbone_modules_in_eval_mode(model)

    if task_name == "sleep_stage_classification":
        weight = class_weights.to(device) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
    elif task_name == "slow_wave_detection":
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_labels = []

    subject_logits = defaultdict(list)
    subject_labels = {}

    use_amp = (device.type == "cuda" and cfg.use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for batch in tqdm(loader, desc=f"Train {task_name}", leave=False):
        eeg = batch["eeg"].to(device)
        eog_available = batch["eog_available"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        labels = batch["labels"].to(device)
        subject_ids = batch["subject_id"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                eeg,
                task_name,
                eog_available=eog_available,
                valid_mask=valid_mask,
            )

            if task_name in {"sleep_stage_classification", "slow_wave_detection"}:
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if getattr(cfg, "grad_clip_norm", 0.0) and cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_batches += 1

        if task_name in {"sleep_stage_classification", "slow_wave_detection"}:
            preds = logits.argmax(dim=-1).detach().cpu().numpy().reshape(-1)
            labels_np = labels.detach().cpu().numpy().reshape(-1)
            valid_np = labels_np != -100
            if np.any(valid_np):
                all_preds.extend(preds[valid_np].tolist())
                all_labels.extend(labels_np[valid_np].tolist())
        else:
            logits_np = logits.detach().cpu().numpy()
            for sid, logit, label in zip(subject_ids, logits_np, labels.detach().cpu().numpy().tolist()):
                sid = str(sid)
                subject_logits[sid].append(logit)
                subject_labels[sid] = int(label)

    avg_loss = total_loss / max(n_batches, 1)

    if task_name in {"sleep_stage_classification", "slow_wave_detection"}:
        if len(all_labels) == 0:
            acc = 0.0
            kappa = 0.0
            macro_f1 = 0.0
        else:
            acc = accuracy_score(all_labels, all_preds)
            kappa = _safe_kappa(all_labels, all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    else:
        y_true, y_pred = [], []
        for sid, logits_list in subject_logits.items():
            mean_logit = np.mean(np.stack(logits_list, axis=0), axis=0)
            pred = int(np.argmax(mean_logit))
            y_pred.append(pred)
            y_true.append(subject_labels[sid])

        if len(y_true) == 0:
            acc = 0.0
            kappa = 0.0
            macro_f1 = 0.0
        else:
            acc = accuracy_score(y_true, y_pred)
            kappa = _safe_kappa(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "acc": float(acc),
        "kappa": float(kappa),
        "macro_f1": float(macro_f1)
    }


def _stage_metrics_from_predictions(labels, preds, class_names=None):
    if len(labels) == 0:
        return {
            "acc": 0.0,
            "kappa": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class_metrics": {},
            "per_class_f1": {},
            "report": {},
            "n_samples": 0,
        }

    acc = accuracy_score(labels, preds)
    kappa = _safe_kappa(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))) if class_names is not None else None,
        target_names=class_names if class_names is not None else None,
        output_dict=True,
        zero_division=0,
    )
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    per_class_metrics = {}
    if class_names is not None:
        for name in class_names:
            item = report.get(name, {})
            per_class_metrics[name] = {
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1": float(item.get("f1-score", 0.0)),
                "support": int(item.get("support", 0)),
            }
    per_class_f1 = {
        name: float(report.get(name, {}).get("f1-score", 0.0))
        for name in (class_names if class_names is not None else [])
    }

    return {
        "acc": float(acc),
        "kappa": float(kappa),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": per_class_metrics,
        "per_class_f1": per_class_f1,
        "report": report,
        "n_samples": int(len(labels)),
    }


def build_stage_transition_log_probs(samples, num_classes, smoothing=1.0):
    counts = np.full((num_classes, num_classes), float(smoothing), dtype=np.float64)
    by_subject = defaultdict(dict)
    for sample in samples:
        sid = str(sample["subject_id"])
        start = int(sample.get("start_idx", 0))
        labels = np.asarray(sample["labels"], dtype=np.int64)
        valid_mask = np.asarray(sample["valid_mask"], dtype=np.int64)
        for i, label in enumerate(labels):
            if i >= len(valid_mask) or valid_mask[i] <= 0:
                continue
            label = int(label)
            if 0 <= label < num_classes:
                by_subject[sid][start + i] = label

    for epochs in by_subject.values():
        ordered = [epochs[k] for k in sorted(epochs)]
        for prev_label, next_label in zip(ordered[:-1], ordered[1:]):
            counts[prev_label, next_label] += 1.0

    probs = counts / np.clip(counts.sum(axis=1, keepdims=True), 1e-12, None)
    return np.log(probs + 1e-12).astype(np.float32)


def viterbi_decode(log_emissions, log_transitions, transition_weight=0.5):
    log_emissions = np.asarray(log_emissions, dtype=np.float32)
    if log_emissions.ndim != 2 or log_emissions.shape[0] == 0:
        return np.asarray([], dtype=np.int64)

    trans = np.asarray(log_transitions, dtype=np.float32) * float(transition_weight)
    n_steps, n_classes = log_emissions.shape
    dp = np.zeros((n_steps, n_classes), dtype=np.float32)
    backptr = np.zeros((n_steps, n_classes), dtype=np.int64)
    dp[0] = log_emissions[0]

    for t in range(1, n_steps):
        scores = dp[t - 1][:, None] + trans
        backptr[t] = np.argmax(scores, axis=0)
        dp[t] = log_emissions[t] + np.max(scores, axis=0)

    path = np.zeros(n_steps, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_steps - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def _aggregate_stage_logits_by_epoch(
    model,
    loader,
    device,
    preserve_model_mode=False,
):
    if not preserve_model_mode:
        model.eval()
    by_subject = defaultdict(dict)
    all_features = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval sleep_stage", leave=False):
            eeg = batch["eeg"].to(device)
            eog_available = batch["eog_available"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            labels = batch["labels"].to(device)
            subject_ids = batch["subject_id"]
            start_idxs = batch["start_idx"].cpu().numpy().tolist()

            backbone_feats = model.extract_stage_features(
                eeg,
                eog_available=eog_available,
                valid_mask=valid_mask,
            )
            if getattr(model, "stage_head_type", "linear") == "tcn":
                # Store the representation used directly by the classifier so
                # t-SNE/UMAP describe the selected downstream model rather
                # than only the upstream backbone.
                feats = model.stage_head.extract_features(
                    backbone_feats,
                    valid_mask=valid_mask,
                )
                logits = model.stage_head.out(feats)
            else:
                feats = backbone_feats
                logits = model.stage_head(feats)

            logits_np = logits.detach().cpu().numpy()
            log_probs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
            feats_np = feats.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            valid_np = valid_mask.detach().cpu().numpy()

            for b, sid in enumerate(subject_ids):
                sid = str(sid)
                start = int(start_idxs[b])
                for i in range(log_probs.shape[1]):
                    if valid_np[b, i] <= 0:
                        continue
                    label = int(labels_np[b, i])
                    if label < 0:
                        continue
                    epoch_idx = start + i
                    slot = by_subject[sid].setdefault(
                        epoch_idx,
                        {"logits": [], "log_probs": [], "labels": [], "features": []}
                    )
                    slot["logits"].append(logits_np[b, i])
                    slot["log_probs"].append(log_probs[b, i])
                    slot["labels"].append(label)
                    slot["features"].append(feats_np[b, i])

    return by_subject, all_features


def _evaluate_stage_viterbi_from_aggregated(
    by_subject,
    log_transition,
    transition_weight=0.5,
    class_names=None,
    save_dir=None,
    seed=42,
):
    """Evaluate compact subject/epoch emissions without another model forward pass."""
    raw_labels = []
    raw_preds = []
    viterbi_labels = []
    viterbi_preds = []
    all_features = []

    for sid in sorted(by_subject):
        epochs = by_subject[sid]
        ordered_idx = sorted(epochs)
        if not ordered_idx:
            continue
        labels = []
        log_emissions = []
        features = []
        for epoch_idx in ordered_idx:
            item = epochs[epoch_idx]
            if "logits" in item and len(item["logits"]) > 0:
                mean_logits = np.mean(np.stack(item["logits"], axis=0), axis=0)
                emission_probs = _softmax_np(mean_logits)
                log_emissions.append(
                    np.log(np.clip(emission_probs, 1e-12, None)).astype(np.float32)
                )
            else:
                log_emissions.append(
                    np.mean(np.stack(item["log_probs"], axis=0), axis=0)
                )
            labels.append(int(round(float(np.median(item["labels"])))))
            features.append(np.mean(np.stack(item["features"], axis=0), axis=0))

        log_emissions = np.stack(log_emissions, axis=0)
        labels = np.asarray(labels, dtype=np.int64)
        raw_path = np.argmax(log_emissions, axis=1).astype(np.int64)
        viterbi_path = viterbi_decode(log_emissions, log_transition, transition_weight=transition_weight)

        raw_labels.extend(labels.tolist())
        raw_preds.extend(raw_path.tolist())
        viterbi_labels.extend(labels.tolist())
        viterbi_preds.extend(viterbi_path.tolist())
        all_features.append(np.stack(features, axis=0))

    raw_metrics = _stage_metrics_from_predictions(raw_labels, raw_preds, class_names)
    viterbi_metrics = _stage_metrics_from_predictions(viterbi_labels, viterbi_preds, class_names)
    n_unique_epochs = len(viterbi_labels)
    n_window_predictions = sum(
        len(item.get("logits", item.get("log_probs", [])))
        for epochs in by_subject.values()
        for item in epochs.values()
    )
    aggregation_metadata = {
        "evaluation_level": "unique_epoch",
        "aggregation": "mean_logits_by_subject_and_epoch",
        "n_unique_epochs": int(n_unique_epochs),
        "n_window_predictions": int(n_window_predictions),
        "mean_predictions_per_epoch": float(
            n_window_predictions / max(n_unique_epochs, 1)
        ),
    }
    raw_metrics.update(aggregation_metadata)
    viterbi_metrics.update(aggregation_metadata)
    viterbi_metrics["raw_metrics"] = raw_metrics
    viterbi_metrics["viterbi_transition_weight"] = float(transition_weight)

    if save_dir is not None and class_names is not None and len(viterbi_labels) > 0:
        ensure_dir(save_dir)
        save_confusion_matrix_plot(
            viterbi_labels, viterbi_preds, class_names,
            os.path.join(save_dir, "confusion_matrix_stage_viterbi.png"),
            "Sleep Stage Confusion Matrix Viterbi"
        )
        save_confusion_matrix_plot(
            raw_labels, raw_preds, class_names,
            os.path.join(save_dir, "confusion_matrix_stage_raw_epoch_agg.png"),
            "Sleep Stage Confusion Matrix Raw Aggregated"
        )
        if len(all_features) > 0:
            feat_mat = np.concatenate(all_features, axis=0)
            save_tsne_plot(
                feat_mat, np.array(viterbi_labels), class_names,
                os.path.join(save_dir, "tsne_stage_viterbi.png"),
                "Sleep Stage t-SNE Viterbi", seed=seed
            )

    return viterbi_metrics


def enable_stage_mc_dropout(model, scope="tcn", dropout_p=0.02):
    """Enable inference Dropout without allowing BatchNorm updates."""
    model.eval()
    enabled_names = []
    scope = str(scope).lower()
    if scope == "none":
        return enabled_names
    if scope not in {"tcn", "all"}:
        raise ValueError(f"Unsupported stage MC Dropout scope: {scope}")

    if scope == "all":
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                module.train()
                enabled_names.append(name)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Dropout):
            continue
        if scope == "all" or name.startswith("stage_head"):
            module.p = float(dropout_p)
            module.train()
            enabled_names.append(name)

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            raise RuntimeError("BatchNorm entered train mode during MC Dropout test.")
    if not enabled_names:
        raise RuntimeError(f"No Dropout modules were enabled for scope={scope}.")
    return sorted(set(enabled_names))


def _stage_mean_epoch_log_emissions(item):
    if "logits" in item and len(item["logits"]) > 0:
        mean_logits = np.mean(np.stack(item["logits"], axis=0), axis=0)
        probs = _softmax_np(mean_logits)
        return np.log(np.clip(probs, 1e-12, None)).astype(np.float32)
    return np.mean(np.stack(item["log_probs"], axis=0), axis=0).astype(
        np.float32
    )


def _stage_path_segment_gain(
    log_emissions,
    log_transition,
    transition_weight,
    base_path,
    candidate_path,
    start,
    end,
):
    trial_path = base_path.copy()
    trial_path[start:end + 1] = candidate_path[start:end + 1]
    positions = np.arange(start, end + 1)
    emission_gain = np.sum(
        log_emissions[positions, trial_path[positions]]
        - log_emissions[positions, base_path[positions]]
    )
    transition_gain = 0.0
    first_edge = max(1, start)
    last_edge = min(len(base_path) - 1, end + 1)
    for right in range(first_edge, last_edge + 1):
        left = right - 1
        transition_gain += (
            log_transition[trial_path[left], trial_path[right]]
            - log_transition[base_path[left], base_path[right]]
        )
    return float(emission_gain + float(transition_weight) * transition_gain)


def evaluate_stage_confidence_gated_viterbi_from_aggregated(
    by_subject,
    log_transition,
    transition_weight,
    class_names,
    confidence_threshold=0.85,
    margin_threshold=0.40,
    min_sequence_gain=-0.40,
):
    """Create a fixed raw/Viterbi hybrid without reading labels for decisions."""
    all_labels = []
    all_preds = []
    n_viterbi_changes = 0
    n_eligible_changes = 0
    n_accepted_changes = 0
    n_candidate_segments = 0
    n_accepted_segments = 0
    accepted_score_gain = 0.0

    for subject_id in sorted(by_subject):
        epochs = by_subject[subject_id]
        ordered_idx = sorted(epochs)
        if not ordered_idx:
            continue
        labels = np.asarray(
            [
                int(round(float(np.median(epochs[idx]["labels"]))))
                for idx in ordered_idx
            ],
            dtype=np.int64,
        )
        log_emissions = np.stack(
            [_stage_mean_epoch_log_emissions(epochs[idx]) for idx in ordered_idx],
            axis=0,
        )
        probs = np.exp(log_emissions)
        raw_path = np.argmax(log_emissions, axis=1).astype(np.int64)
        viterbi_path = viterbi_decode(
            log_emissions,
            log_transition,
            transition_weight=transition_weight,
        )
        sorted_probs = np.sort(probs, axis=1)
        confidence = sorted_probs[:, -1]
        margin = (
            sorted_probs[:, -1] - sorted_probs[:, -2]
            if sorted_probs.shape[1] > 1
            else confidence.copy()
        )
        changed = viterbi_path != raw_path
        eligible = (
            changed
            & (confidence <= float(confidence_threshold))
            & (margin <= float(margin_threshold))
        )
        gated_path = raw_path.copy()
        n_viterbi_changes += int(changed.sum())
        n_eligible_changes += int(eligible.sum())

        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size > 0:
            split_points = np.where(np.diff(eligible_indices) > 1)[0] + 1
            for segment in np.split(eligible_indices, split_points):
                start = int(segment[0])
                end = int(segment[-1])
                n_candidate_segments += 1
                gain = _stage_path_segment_gain(
                    log_emissions,
                    log_transition,
                    transition_weight,
                    gated_path,
                    viterbi_path,
                    start,
                    end,
                )
                if gain >= float(min_sequence_gain):
                    gated_path[start:end + 1] = viterbi_path[start:end + 1]
                    n_accepted_segments += 1
                    n_accepted_changes += int(end - start + 1)
                    accepted_score_gain += gain

        all_labels.extend(labels.tolist())
        all_preds.extend(gated_path.tolist())

    metrics = _stage_metrics_from_predictions(all_labels, all_preds, class_names)
    metrics.update(
        {
            "evaluation_level": "unique_epoch",
            "aggregation": "mean_logits_by_subject_and_epoch",
            "gate_confidence_threshold": float(confidence_threshold),
            "gate_margin_threshold": float(margin_threshold),
            "gate_min_sequence_gain": float(min_sequence_gain),
            "viterbi_transition_weight": float(transition_weight),
            "n_viterbi_changed_epochs": int(n_viterbi_changes),
            "n_gate_eligible_epochs": int(n_eligible_changes),
            "n_gate_accepted_epochs": int(n_accepted_changes),
            "n_gate_candidate_segments": int(n_candidate_segments),
            "n_gate_accepted_segments": int(n_accepted_segments),
            "accepted_sequence_score_gain": float(accepted_score_gain),
        }
    )
    return metrics


def evaluate_stage_fixed_method_from_model(
    model,
    loader,
    cfg,
    device,
    log_transition,
    inference_seed,
    return_aggregated=False,
):
    """Evaluate one fixed decoder while preserving the training RNG stream."""
    primary_method = str(cfg.saved_stage_primary_method).strip().lower()
    if primary_method not in {"raw", "viterbi", "gated_viterbi"}:
        raise ValueError(f"Unsupported fixed stage decoder: {primary_method!r}")
    if primary_method != "raw" and log_transition is None:
        raise ValueError("The fixed decoder requires training-only transitions.")
    rng_state = _capture_rng_state()
    module_training = {
        module: bool(module.training) for module in model.modules()
    }
    dropout_probabilities = {
        module: float(module.p)
        for module in model.modules()
        if isinstance(module, nn.Dropout)
    }
    try:
        set_seed(int(inference_seed))
        enabled_modules = enable_stage_mc_dropout(
            model,
            scope=cfg.saved_stage_mc_dropout_scope,
            dropout_p=cfg.saved_stage_mc_dropout_p,
        )
        aggregated, _ = _aggregate_stage_logits_by_epoch(
            model,
            loader,
            device,
            preserve_model_mode=True,
        )
    finally:
        for module, dropout_p in dropout_probabilities.items():
            module.p = dropout_p
        for module, was_training in module_training.items():
            module.training = was_training
        _restore_rng_state(rng_state)

    class_names = cfg.tasks["sleep_stage_classification"]["class_names"]
    if primary_method == "raw" and log_transition is None:
        num_classes = int(cfg.tasks["sleep_stage_classification"]["num_classes"])
        log_transition = np.zeros((num_classes, num_classes), dtype=np.float32)
    if primary_method == "gated_viterbi":
        primary_metrics = evaluate_stage_confidence_gated_viterbi_from_aggregated(
            aggregated, log_transition=log_transition,
            transition_weight=cfg.saved_stage_viterbi_weight, class_names=class_names,
            confidence_threshold=cfg.saved_stage_gate_confidence_threshold,
            margin_threshold=cfg.saved_stage_gate_margin_threshold,
            min_sequence_gain=cfg.saved_stage_gate_min_sequence_gain,
        )
    else:
        primary_metrics = _evaluate_stage_viterbi_from_aggregated(
            aggregated, log_transition=log_transition,
            transition_weight=(cfg.saved_stage_viterbi_weight if primary_method == "viterbi" else 0.0),
            class_names=class_names, save_dir=None, seed=int(inference_seed),
        )
        primary_metrics.pop("raw_metrics", None)
    result = {
        "primary_metrics": primary_metrics,
        "primary_method": primary_method,
        "selection_rule": "fixed_before_training",
        "uses_test_labels_for_method_selection": False,
        "inference_seed": int(inference_seed),
        "enabled_mc_modules": enabled_modules,
    }
    if return_aggregated:
        result["_aggregated"] = aggregated
    return result


def _selected_stage_visualization_arrays_from_aggregated(
    by_subject,
    primary_method,
    log_transition,
    transition_weight,
    confidence_threshold,
    margin_threshold,
    min_sequence_gain,
):
    """Collect the exact unique-epoch labels/predictions used by one decoder."""
    if primary_method not in {"raw", "viterbi", "gated_viterbi"}:
        raise ValueError(f"Unsupported visualization decoder: {primary_method}")

    all_labels = []
    all_preds = []
    all_features = []
    all_subject_ids = []
    all_epoch_indices = []

    for subject_id in sorted(by_subject):
        epochs = by_subject[subject_id]
        ordered_idx = sorted(epochs)
        if not ordered_idx:
            continue
        labels = np.asarray(
            [
                int(round(float(np.median(epochs[idx]["labels"]))))
                for idx in ordered_idx
            ],
            dtype=np.int64,
        )
        features = np.stack(
            [
                np.mean(np.stack(epochs[idx]["features"], axis=0), axis=0)
                for idx in ordered_idx
            ],
            axis=0,
        ).astype(np.float32)
        log_emissions = np.stack(
            [_stage_mean_epoch_log_emissions(epochs[idx]) for idx in ordered_idx],
            axis=0,
        )
        raw_path = np.argmax(log_emissions, axis=1).astype(np.int64)
        viterbi_path = viterbi_decode(
            log_emissions,
            log_transition,
            transition_weight=transition_weight,
        )

        if primary_method == "raw":
            selected_path = raw_path
        elif primary_method == "viterbi":
            selected_path = viterbi_path
        else:
            probs = np.exp(log_emissions)
            sorted_probs = np.sort(probs, axis=1)
            confidence = sorted_probs[:, -1]
            margin = (
                sorted_probs[:, -1] - sorted_probs[:, -2]
                if sorted_probs.shape[1] > 1
                else confidence.copy()
            )
            eligible = (
                (viterbi_path != raw_path)
                & (confidence <= float(confidence_threshold))
                & (margin <= float(margin_threshold))
            )
            selected_path = raw_path.copy()
            eligible_indices = np.flatnonzero(eligible)
            if eligible_indices.size > 0:
                split_points = np.where(np.diff(eligible_indices) > 1)[0] + 1
                for segment in np.split(eligible_indices, split_points):
                    start = int(segment[0])
                    end = int(segment[-1])
                    gain = _stage_path_segment_gain(
                        log_emissions,
                        log_transition,
                        transition_weight,
                        selected_path,
                        viterbi_path,
                        start,
                        end,
                    )
                    if gain >= float(min_sequence_gain):
                        selected_path[start:end + 1] = viterbi_path[start:end + 1]

        all_labels.extend(labels.tolist())
        all_preds.extend(selected_path.tolist())
        all_features.append(features)
        all_subject_ids.extend([str(subject_id)] * len(ordered_idx))
        all_epoch_indices.extend([int(idx) for idx in ordered_idx])

    return {
        "labels": np.asarray(all_labels, dtype=np.int64),
        "preds": np.asarray(all_preds, dtype=np.int64),
        "features": (
            np.concatenate(all_features, axis=0)
            if all_features
            else np.empty((0, 0), dtype=np.float32)
        ),
        "subject_ids": np.asarray(all_subject_ids, dtype=str),
        "epoch_indices": np.asarray(all_epoch_indices, dtype=np.int64),
    }


def _transformer_layer_with_attention(layer, x, key_padding_mask=None):
    """Reproduce TransformerEncoderLayer forward and expose per-head weights."""
    if layer.norm_first:
        normalized = layer.norm1(x)
        attention_output, attention_weights = layer.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + layer.dropout1(attention_output)
        normalized = layer.norm2(x)
        feedforward = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(normalized)))
        )
        x = x + layer.dropout2(feedforward)
    else:
        attention_output, attention_weights = layer.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = layer.norm1(x + layer.dropout1(attention_output))
        feedforward = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(x)))
        )
        x = layer.norm2(x + layer.dropout2(feedforward))
    return x, attention_weights


def collect_best_transformer_attention(
    model,
    loader,
    device,
    num_classes,
    layer_index=-1,
):
    """Average real multi-head self-attention over valid test windows."""
    layers = model.backbone.seq_encoder.layers
    n_layers = len(layers)
    resolved_layer = int(layer_index)
    if resolved_layer < 0:
        resolved_layer += n_layers
    if resolved_layer < 0 or resolved_layer >= n_layers:
        raise ValueError(
            f"stage_attention_layer={layer_index} is invalid for {n_layers} layers."
        )

    original_training = {
        module: bool(module.training) for module in model.modules()
    }
    model.eval()
    position_sum = None
    position_count = None
    stage_sum = None
    stage_count = None

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="Best attention", leave=False):
                eeg = batch["eeg"].to(device)
                eog_available = batch["eog_available"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                labels = batch["labels"].to(device)

                x = model.backbone.encode_epochs(
                    eeg,
                    eog_available=eog_available,
                )
                seq_len = x.size(1)
                x = model.backbone.drop(
                    model.backbone.norm(
                        x + model.backbone.pos_embed[:, :seq_len, :]
                    )
                )
                key_padding_mask = valid_mask == 0
                target_attention = None
                for current_idx, layer in enumerate(layers):
                    x, attention_weights = _transformer_layer_with_attention(
                        layer,
                        x,
                        key_padding_mask=key_padding_mask,
                    )
                    if current_idx == resolved_layer:
                        target_attention = attention_weights.detach().cpu().numpy()
                        break
                if target_attention is None:
                    raise RuntimeError("Failed to collect Transformer attention weights.")

                valid_np = valid_mask.detach().cpu().numpy().astype(bool)
                labels_np = labels.detach().cpu().numpy()
                _, n_heads, n_positions, _ = target_attention.shape
                if position_sum is None:
                    position_sum = np.zeros(
                        (n_heads, n_positions, n_positions), dtype=np.float64
                    )
                    position_count = np.zeros(
                        (n_positions, n_positions), dtype=np.float64
                    )
                    stage_sum = np.zeros(
                        (n_heads, num_classes, num_classes), dtype=np.float64
                    )
                    stage_count = np.zeros(
                        (num_classes, num_classes), dtype=np.float64
                    )

                for batch_idx in range(target_attention.shape[0]):
                    valid_positions = np.flatnonzero(valid_np[batch_idx])
                    for query_pos in valid_positions:
                        query_label = int(labels_np[batch_idx, query_pos])
                        if not 0 <= query_label < num_classes:
                            continue
                        for key_pos in valid_positions:
                            key_label = int(labels_np[batch_idx, key_pos])
                            if not 0 <= key_label < num_classes:
                                continue
                            values = target_attention[
                                batch_idx, :, query_pos, key_pos
                            ]
                            position_sum[:, query_pos, key_pos] += values
                            position_count[query_pos, key_pos] += 1.0
                            stage_sum[:, query_label, key_label] += values
                            stage_count[query_label, key_label] += 1.0
    finally:
        for module, was_training in original_training.items():
            module.training = was_training

    if position_sum is None:
        raise RuntimeError("No valid windows were available for attention visualization.")
    mean_position = position_sum / np.where(
        position_count[None, :, :] > 0,
        position_count[None, :, :],
        np.nan,
    )
    mean_stage = stage_sum / np.where(
        stage_count[None, :, :] > 0,
        stage_count[None, :, :],
        np.nan,
    )
    return {
        "position_attention_by_head": mean_position.astype(np.float32),
        "stage_attention_by_head": mean_stage.astype(np.float32),
        "layer_index": int(resolved_layer),
        "n_layers": int(n_layers),
    }


def save_selected_best_stage_visualizations(
    model,
    loader,
    aggregated,
    cfg,
    device,
    log_transition,
    best_epoch,
    primary_method,
    primary_metrics,
    fold_idx,
    save_dir,
    inference_seed,
):
    """Save four visualization types tied to the selected best checkpoint."""
    ensure_dir(save_dir)
    class_names = cfg.tasks["sleep_stage_classification"]["class_names"]
    artifacts = _selected_stage_visualization_arrays_from_aggregated(
        aggregated,
        primary_method=primary_method,
        log_transition=log_transition,
        transition_weight=cfg.saved_stage_viterbi_weight,
        confidence_threshold=cfg.saved_stage_gate_confidence_threshold,
        margin_threshold=cfg.saved_stage_gate_margin_threshold,
        min_sequence_gain=cfg.saved_stage_gate_min_sequence_gain,
    )
    labels = artifacts["labels"]
    preds = artifacts["preds"]
    features = artifacts["features"]
    reproduced_metrics = _stage_metrics_from_predictions(
        labels.tolist(), preds.tolist(), class_names
    )
    checked_metrics = ("acc", "kappa", "macro_f1", "weighted_f1")
    metric_delta = {
        name: float(reproduced_metrics[name] - primary_metrics[name])
        for name in checked_metrics
    }
    if any(abs(delta) > 1e-10 for delta in metric_delta.values()):
        raise RuntimeError(
            "Best-model visualization predictions do not reproduce the selected "
            f"metrics: deltas={metric_delta}"
        )

    caption = _stage_visualization_caption(
        best_epoch,
        primary_method,
        primary_metrics,
    )
    save_best_stage_confusion_matrix(
        labels,
        preds,
        class_names,
        os.path.join(save_dir, "cm_best.png"),
        caption,
    )

    max_points = max(1, int(cfg.stage_visualization_max_points))
    if len(labels) > max_points:
        selected_indices = np.sort(
            np.random.RandomState(int(inference_seed)).choice(
                len(labels), max_points, replace=False
            )
        )
    else:
        selected_indices = np.arange(len(labels), dtype=np.int64)
    plot_features = features[selected_indices]
    plot_labels = labels[selected_indices]
    tsne_embedding = compute_stage_tsne(
        plot_features,
        seed=int(inference_seed),
    )
    umap_embedding = compute_stage_umap(
        plot_features,
        seed=int(inference_seed),
        n_neighbors=cfg.stage_umap_n_neighbors,
        min_dist=cfg.stage_umap_min_dist,
    )
    save_stage_embedding_plot(
        tsne_embedding,
        plot_labels,
        class_names,
        os.path.join(save_dir, "tsne_best.png"),
        "t-SNE",
        caption,
    )
    save_stage_embedding_plot(
        umap_embedding,
        plot_labels,
        class_names,
        os.path.join(save_dir, "umap_best.png"),
        "UMAP",
        caption,
    )

    attention = collect_best_transformer_attention(
        model,
        loader,
        device,
        num_classes=len(class_names),
        layer_index=cfg.stage_attention_layer,
    )
    save_stage_attention_plots(
        attention["position_attention_by_head"],
        attention["stage_attention_by_head"],
        class_names,
        os.path.join(save_dir, "attn_best.png"),
        os.path.join(save_dir, "attn_heads_best.png"),
        caption,
        attention["layer_index"],
    )

    np.savez_compressed(
        os.path.join(save_dir, "viz_data.npz"),
        labels=labels,
        predictions=preds,
        features=features,
        subject_ids=artifacts["subject_ids"],
        epoch_indices=artifacts["epoch_indices"],
        embedding_indices=selected_indices,
        tsne=tsne_embedding,
        umap=umap_embedding,
        attention_by_head=attention["position_attention_by_head"],
        stage_attention_by_head=attention["stage_attention_by_head"],
    )
    metadata = {
        "fold_idx": int(fold_idx),
        "best_epoch": int(best_epoch),
        "primary_method": str(primary_method),
        "metrics": {
            name: float(primary_metrics[name]) for name in checked_metrics
        },
        "metric_reproduction_delta": metric_delta,
        "n_unique_epochs": int(len(labels)),
        "feature_source": "TCN final hidden representation before classifier",
        "confusion_prediction_source": str(primary_method),
        "embedding_color_source": "ground-truth sleep stage",
        "attention_source": (
            "selected checkpoint Transformer self-attention; averaged over "
            "valid held-out-subject windows"
        ),
        "attention_is_channel_attention": False,
        "attention_layer_index": int(attention["layer_index"]),
        "attention_num_layers": int(attention["n_layers"]),
        "inference_seed": int(inference_seed),
        "uses_outer_test_for_checkpoint_and_decoder_selection": False,
        "files": {
            "confusion_matrix": "cm_best.png",
            "tsne": "tsne_best.png",
            "umap": "umap_best.png",
            "attention": "attn_best.png",
            "attention_heads": "attn_heads_best.png",
            "numeric_data": "viz_data.npz",
        },
    }
    with open(
        os.path.join(save_dir, "viz_meta.json"),
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)
    print(
        f"[Best Visualizations] Fold {fold_idx} | epoch={best_epoch} | "
        f"method={primary_method} | acc={primary_metrics['acc']:.4f} | "
        f"verified_delta_acc={metric_delta['acc']:+.1e} | dir={save_dir}"
    )
    return metadata


def evaluate_stage_viterbi(
    model,
    loader,
    device,
    log_transition,
    transition_weight=0.5,
    class_names=None,
    save_dir=None,
    seed=42,
):
    by_subject, _ = _aggregate_stage_logits_by_epoch(model, loader, device)
    return _evaluate_stage_viterbi_from_aggregated(
        by_subject,
        log_transition,
        transition_weight=transition_weight,
        class_names=class_names,
        save_dir=save_dir,
        seed=seed,
    )


def evaluate_stage_window_level(model, loader, device, class_names=None, save_dir=None, seed=42):
    model.eval()

    all_preds = []
    all_labels = []
    all_features = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval sleep_stage", leave=False):
            eeg = batch["eeg"].to(device)
            eog_available = batch["eog_available"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            labels = batch["labels"].to(device)

            feats = model.extract_stage_features(
                eeg,
                eog_available=eog_available,
                valid_mask=valid_mask,
            )
            if getattr(model, "stage_head_type", "linear") == "tcn":
                logits = model.stage_head(feats, valid_mask=valid_mask)
            else:
                logits = model.stage_head(feats)
            preds = logits.argmax(dim=-1)

            labels_np = labels.cpu().numpy().reshape(-1)
            preds_np = preds.cpu().numpy().reshape(-1)
            feats_np = feats.cpu().numpy().reshape(-1, feats.size(-1))

            valid_np = labels_np != -100
            labels_np = labels_np[valid_np]
            preds_np = preds_np[valid_np]
            feats_np = feats_np[valid_np]

            if len(labels_np) > 0:
                all_labels.extend(labels_np.tolist())
                all_preds.extend(preds_np.tolist())
                all_features.append(feats_np)

    if len(all_labels) == 0:
        return {
            "acc": 0.0,
            "kappa": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class_metrics": {},
            "per_class_f1": {},
            "report": {},
            "n_samples": 0,
        }

    acc = accuracy_score(all_labels, all_preds)
    kappa = _safe_kappa(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))) if class_names is not None else None,
        target_names=class_names if class_names is not None else None,
        output_dict=True,
        zero_division=0,
    )
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    per_class_metrics = {}
    if class_names is not None:
        for name in class_names:
            item = report.get(name, {})
            per_class_metrics[name] = {
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1": float(item.get("f1-score", 0.0)),
                "support": int(item.get("support", 0)),
            }
    per_class_f1 = {
        name: float(report.get(name, {}).get("f1-score", 0.0))
        for name in (class_names if class_names is not None else [])
    }

    if save_dir is not None and class_names is not None:
        ensure_dir(save_dir)
        save_confusion_matrix_plot(
            all_labels, all_preds, class_names,
            os.path.join(save_dir, "confusion_matrix_stage.png"),
            "Sleep Stage Confusion Matrix"
        )
        if len(all_features) > 0:
            feat_mat = np.concatenate(all_features, axis=0)
            save_tsne_plot(
                feat_mat, np.array(all_labels), class_names,
                os.path.join(save_dir, "tsne_stage.png"),
                "Sleep Stage t-SNE", seed=seed
            )

    return {
        "acc": float(acc),
        "kappa": float(kappa),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": per_class_metrics,
        "per_class_f1": per_class_f1,
        "report": report,
        "n_samples": int(len(all_labels)),
    }


def _evaluate_stage_epoch_aggregated_from_aggregated(
    by_subject,
    class_names=None,
    save_dir=None,
    seed=42,
):
    """Average overlapping-window logits and score every original epoch once."""
    all_labels = []
    all_preds = []
    all_features = []
    n_window_predictions = 0

    for sid in sorted(by_subject):
        epochs = by_subject[sid]
        for epoch_idx in sorted(epochs):
            item = epochs[epoch_idx]
            if len(item["logits"]) == 0:
                continue

            epoch_logits = np.mean(np.stack(item["logits"], axis=0), axis=0)
            epoch_label = int(round(float(np.median(item["labels"]))))
            epoch_feature = np.mean(np.stack(item["features"], axis=0), axis=0)

            all_labels.append(epoch_label)
            all_preds.append(int(np.argmax(epoch_logits)))
            all_features.append(epoch_feature)
            n_window_predictions += len(item["logits"])

    metrics = _stage_metrics_from_predictions(all_labels, all_preds, class_names)
    n_unique_epochs = len(all_labels)
    metrics["evaluation_level"] = "unique_epoch"
    metrics["aggregation"] = "mean_logits_by_subject_and_epoch"
    metrics["n_unique_epochs"] = int(n_unique_epochs)
    metrics["n_window_predictions"] = int(n_window_predictions)
    metrics["mean_predictions_per_epoch"] = float(
        n_window_predictions / max(n_unique_epochs, 1)
    )

    if save_dir is not None and class_names is not None and n_unique_epochs > 0:
        ensure_dir(save_dir)
        save_confusion_matrix_plot(
            all_labels,
            all_preds,
            class_names,
            os.path.join(save_dir, "confusion_matrix_stage_epoch_aggregated.png"),
            "Sleep Stage Confusion Matrix (Unique Epochs)",
        )
        if len(all_features) > 0:
            save_tsne_plot(
                np.stack(all_features, axis=0),
                np.asarray(all_labels, dtype=np.int64),
                class_names,
                os.path.join(save_dir, "tsne_stage_epoch_aggregated.png"),
                "Sleep Stage t-SNE (Unique Epochs)",
                seed=seed,
            )

    return metrics


def evaluate_stage(model, loader, device, class_names=None, save_dir=None, seed=42):
    """Default sleep-stage evaluation: one prediction per original epoch."""
    by_subject, _ = _aggregate_stage_logits_by_epoch(model, loader, device)
    return _evaluate_stage_epoch_aggregated_from_aggregated(
        by_subject,
        class_names=class_names,
        save_dir=save_dir,
        seed=seed,
    )


def evaluate_slow_wave(model, loader, device, class_names=None, save_dir=None, seed=42):
    model.eval()

    all_preds = []
    all_labels = []
    all_features = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval slow_wave", leave=False):
            eeg = batch["eeg"].to(device)
            eog_available = batch["eog_available"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            labels = batch["labels"].to(device)

            feats = model.backbone(eeg, eog_available=eog_available, valid_mask=valid_mask)
            logits = model.slow_wave_head(feats)
            preds = logits.argmax(dim=-1)

            labels_np = labels.cpu().numpy().reshape(-1)
            preds_np = preds.cpu().numpy().reshape(-1)
            feats_np = feats.cpu().numpy().reshape(-1, feats.size(-1))

            valid_np = labels_np != -100
            labels_np = labels_np[valid_np]
            preds_np = preds_np[valid_np]
            feats_np = feats_np[valid_np]

            if len(labels_np) > 0:
                all_labels.extend(labels_np.tolist())
                all_preds.extend(preds_np.tolist())
                all_features.append(feats_np)

    if len(all_labels) == 0:
        return {
            "acc": 0.0,
            "kappa": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class_metrics": {},
            "per_class_f1": {},
            "report": {},
            "n_samples": 0,
        }

    acc = accuracy_score(all_labels, all_preds)
    kappa = _safe_kappa(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))) if class_names is not None else None,
        target_names=class_names if class_names is not None else None,
        output_dict=True,
        zero_division=0,
    )
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    per_class_metrics = {}
    if class_names is not None:
        for name in class_names:
            item = report.get(name, {})
            per_class_metrics[name] = {
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1": float(item.get("f1-score", 0.0)),
                "support": int(item.get("support", 0)),
            }
    per_class_f1 = {
        name: float(report.get(name, {}).get("f1-score", 0.0))
        for name in (class_names if class_names is not None else [])
    }

    if save_dir is not None and class_names is not None:
        ensure_dir(save_dir)
        save_confusion_matrix_plot(
            all_labels, all_preds, class_names,
            os.path.join(save_dir, "confusion_matrix_slow_wave.png"),
            "Slow Wave Confusion Matrix"
        )
        if len(all_features) > 0:
            feat_mat = np.concatenate(all_features, axis=0)
            save_tsne_plot(
                feat_mat, np.array(all_labels), class_names,
                os.path.join(save_dir, "tsne_slow_wave.png"),
                "Slow Wave t-SNE", seed=seed
            )

    return {
        "acc": float(acc),
        "kappa": float(kappa),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": per_class_metrics,
        "per_class_f1": per_class_f1,
        "report": report,
        "n_samples": int(len(all_labels)),
    }


def evaluate_disorder(model, loader, device, class_names=None, save_dir=None, seed=42):
    model.eval()

    subject_logits = defaultdict(list)
    subject_features = defaultdict(list)
    subject_labels = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval sleep_disorder", leave=False):
            eeg = batch["eeg"].to(device)
            eog_available = batch["eog_available"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            labels = batch["labels"].cpu().numpy().tolist()
            subject_ids = batch["subject_id"]

            feats = model.backbone(eeg, eog_available=eog_available, valid_mask=valid_mask)
            mask = valid_mask.unsqueeze(-1).float()
            pooled = (feats * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            logits = model.disorder_head(pooled)

            logits_np = logits.cpu().numpy()
            pooled_np = pooled.cpu().numpy()

            for sid, logit, feat, label in zip(subject_ids, logits_np, pooled_np, labels):
                sid = str(sid)
                subject_logits[sid].append(logit)
                subject_features[sid].append(feat)
                subject_labels[sid] = int(label)

    y_true = []
    y_pred = []
    y_score = []
    all_features = []

    for sid, logits_list in subject_logits.items():
        mean_logit = np.mean(np.stack(logits_list, axis=0), axis=0)
        mean_prob = _softmax_np(mean_logit)
        pred = int(np.argmax(mean_logit))
        y_pred.append(pred)
        y_true.append(subject_labels[sid])
        y_score.append(float(mean_prob[1]) if len(mean_prob) > 1 else 0.0)
        all_features.append(np.mean(np.stack(subject_features[sid], axis=0), axis=0))

    if len(y_true) == 0:
        return {
            "acc": 0.0,
            "kappa": 0.0,
            "balanced_acc": 0.0,
            "auroc": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class_metrics": {},
            "per_class_f1": {},
            "report": {},
            "n_subjects": 0,
        }

    acc = accuracy_score(y_true, y_pred)
    kappa = _safe_kappa(y_true, y_pred)
    balanced_acc = _safe_balanced_accuracy(y_true, y_pred)
    auroc = _safe_binary_auroc(y_true, y_score)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names if class_names is not None else None,
        output_dict=True,
        zero_division=0,
    )
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class_metrics = {}
    if class_names is not None:
        for name in class_names:
            item = report.get(name, {})
            per_class_metrics[name] = {
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1": float(item.get("f1-score", 0.0)),
                "support": int(item.get("support", 0)),
            }
    per_class_f1 = {
        name: float(report.get(name, {}).get("f1-score", 0.0))
        for name in (class_names if class_names is not None else [])
    }

    if save_dir is not None and class_names is not None:
        ensure_dir(save_dir)
        save_confusion_matrix_plot(
            y_true, y_pred, class_names,
            os.path.join(save_dir, "confusion_matrix_disorder.png"),
            "Sleep Disorder Confusion Matrix"
        )
        if len(all_features) > 1:
            feat_mat = np.stack(all_features, axis=0)
            save_tsne_plot(
                feat_mat, np.array(y_true), class_names,
                os.path.join(save_dir, "tsne_disorder.png"),
                "Sleep Disorder t-SNE", seed=seed, max_points=1000
            )

    return {
        "acc": float(acc),
        "kappa": float(kappa),
        "balanced_acc": float(balanced_acc),
        "auroc": float(auroc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": per_class_metrics,
        "per_class_f1": per_class_f1,
        "report": report,
        "n_subjects": int(len(y_true)),
    }


def pick_best_metric(metrics, cfg):
    metric_name = cfg.early_stopping_metric.lower()
    if metric_name == "kappa":
        return float(metrics.get("kappa", 0.0))
    if metric_name == "macro_f1":
        return float(metrics.get("macro_f1", 0.0))
    if metric_name == "weighted_f1":
        return float(metrics.get("weighted_f1", 0.0))
    if metric_name == "balanced_acc":
        return float(metrics.get("balanced_acc", 0.0))
    if metric_name == "auroc":
        return float(metrics.get("auroc", 0.0))
    return float(metrics.get("acc", 0.0))


def is_better_checkpoint(current_metrics, best_metrics, cfg, primary_min_delta=0.0):
    """Compare checkpoints by the configured metric, then macro-F1 on an ACC tie."""
    if best_metrics is None:
        return True, "initial"

    metric_name = str(getattr(cfg, "early_stopping_metric", "acc")).lower()
    current_score = pick_best_metric(current_metrics, cfg)
    best_score = pick_best_metric(best_metrics, cfg)
    if current_score > best_score + float(primary_min_delta):
        return True, "primary"

    # Accuracy is computed from integer counts. Use a very small tolerance only
    # to protect against harmless floating-point representation differences.
    if metric_name == "acc" and np.isclose(
        current_score,
        best_score,
        rtol=0.0,
        atol=1e-12,
    ):
        tiebreak_name = str(
            getattr(cfg, "selection_tiebreak_metric", "macro_f1")
        ).lower()
        current_tiebreak = float(current_metrics.get(tiebreak_name, 0.0))
        best_tiebreak = float(best_metrics.get(tiebreak_name, 0.0))
        if current_tiebreak > best_tiebreak + 1e-12:
            return True, f"{metric_name}_tie_{tiebreak_name}"

    return False, "none"

def print_best_metrics(task_name, best_epoch, best_metrics):
    print(f"[Best] task={task_name} | epoch={best_epoch}")
    print(f"  acc={best_metrics.get('acc', 0.0):.4f}")
    print(f"  kappa={best_metrics.get('kappa', 0.0):.4f}")
    if "balanced_acc" in best_metrics:
        print(f"  balanced_acc={best_metrics.get('balanced_acc', 0.0):.4f}")
    if "auroc" in best_metrics:
        print(f"  auroc={best_metrics.get('auroc', 0.0):.4f}")
    print(f"  macro_f1={best_metrics.get('macro_f1', 0.0):.4f}")
    print(f"  weighted_f1={best_metrics.get('weighted_f1', 0.0):.4f}")

    if "n_samples" in best_metrics:
        print(f"  n_samples={best_metrics['n_samples']}")
    if "n_subjects" in best_metrics:
        print(f"  n_subjects={best_metrics['n_subjects']}")

    per_class = best_metrics.get("per_class_metrics", {})
    if per_class:
        print("[Best] per-class metrics:")
        for cls_name, m in per_class.items():
            print(
                f"  {cls_name:<8} | "
                f"P={m['precision']:.4f}  "
                f"R={m['recall']:.4f}  "
                f"F1={m['f1']:.4f}  "
                f"support={m['support']}"
            )


def print_test_metrics_detail(test_metrics):
    per_class = test_metrics.get("per_class_metrics", {})
    if not per_class:
        return
    print("[Test] per-class metrics:")
    for cls_name, m in per_class.items():
        print(
            f"  {cls_name:<8} | "
            f"P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"support={m['support']}"
        )


def should_stop_early(best_score, current_score, patience_counter, cfg):
    improved = current_score > (best_score + cfg.early_stopping_min_delta)
    if improved:
        return current_score, 0, False
    patience_counter += 1
    stop = patience_counter >= cfg.early_stopping_patience
    return best_score, patience_counter, stop


def load_eeg_pretrained_weights(
    backbone,
    eeg_pretrained_model_path,
    load_full_backbone_pretrain=True,
    interpolate_pos_embed=True,
):
    if eeg_pretrained_model_path is None:
        return False
    if not os.path.exists(eeg_pretrained_model_path):
        print(f"[EEG pretrain] 未找到预训练权重: {eeg_pretrained_model_path}，将使用随机初始化。")
        return False

    try:
        ckpt = torch.load(
            eeg_pretrained_model_path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        ckpt = torch.load(eeg_pretrained_model_path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] weights_only=True 加载失败，尝试 weights_only=False: {e}")
        ckpt = torch.load(
            eeg_pretrained_model_path,
            map_location="cpu",
            weights_only=False
        )

    interpolated_keys = {}

    def _filter_compatible_state_dict(module, state_dict):
        module_state = module.state_dict()
        filtered = {}
        skipped = {}
        for key, value in state_dict.items():
            if key not in module_state:
                continue
            if module_state[key].shape != value.shape:
                target = module_state[key]
                can_interpolate_pos_embed = (
                    bool(interpolate_pos_embed)
                    and key == "pos_embed"
                    and value.ndim == 3
                    and target.ndim == 3
                    and value.shape[0] == target.shape[0]
                    and value.shape[2] == target.shape[2]
                )
                if can_interpolate_pos_embed:
                    source_shape = list(value.shape)
                    resized = F.interpolate(
                        value.float().transpose(1, 2),
                        size=int(target.shape[1]),
                        mode="linear",
                        align_corners=True,
                    ).transpose(1, 2)
                    filtered[key] = resized.to(dtype=target.dtype)
                    interpolated_keys[key] = {
                        "source": source_shape,
                        "target": list(target.shape),
                        "mode": "linear",
                        "align_corners": True,
                    }
                    continue
                skipped[key] = {
                    "expected": list(module_state[key].shape),
                    "loaded": list(value.shape),
                }
                continue
            filtered[key] = value
        return filtered, skipped

    if load_full_backbone_pretrain and isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
        state_dict = ckpt["backbone_state_dict"]
        state_dict, skipped = _filter_compatible_state_dict(backbone, state_dict)
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if interpolated_keys:
            print(
                "[EEG pretrain] interpolated position embedding: "
                f"{interpolated_keys}"
            )
        print(f"[EEG pretrain] 已加载 BERT contextual backbone 权重: {eeg_pretrained_model_path}")
        if skipped:
            print(f"[EEG pretrain] skipped shape-mismatched keys: {skipped}")
        if missing:
            print(f"[EEG pretrain] missing keys: {missing}")
        if unexpected:
            print(f"[EEG pretrain] unexpected keys: {unexpected}")
        return True

    if isinstance(ckpt, dict) and "eeg_encoder_state_dict" in ckpt:
        state_dict = ckpt["eeg_encoder_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict, skipped = _filter_compatible_state_dict(backbone.eeg_encoder, state_dict)
    missing, unexpected = backbone.eeg_encoder.load_state_dict(state_dict, strict=False)
    print(f"[EEG pretrain] 已加载 EEG encoder 权重: {eeg_pretrained_model_path}")
    if skipped:
        print(f"[EEG pretrain] skipped shape-mismatched keys: {skipped}")
    if missing:
        print(f"[EEG pretrain] missing keys: {missing}")
    if unexpected:
        print(f"[EEG pretrain] unexpected keys: {unexpected}")
    return True


def load_generation_decoder_backbone_weights(
    backbone,
    generation_model_path,
    interpolate_pos_embed=True,
):
    """
    Initialize SleepTextBackbone from a v8 03generation_decoder checkpoint.

    The patch-MAE keeps channel CNNs under ``backbone.*`` and trains a separate
    ``visible_encoder`` Transformer copy. Downstream staging therefore loads:
    1) all compatible ``backbone.*`` tensors;
    2) ``visible_encoder.*`` remapped onto ``seq_encoder.*``.
    """
    if generation_model_path is None:
        return False
    generation_model_path = os.path.abspath(generation_model_path)
    if not os.path.exists(generation_model_path):
        print(
            "[EEG pretrain] generation decoder checkpoint missing: "
            f"{generation_model_path}；将使用随机初始化。"
        )
        return False

    try:
        payload = torch.load(
            generation_model_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(generation_model_path, map_location="cpu")

    if not isinstance(payload, dict) or int(payload.get("format_version", -1)) != 8:
        raise RuntimeError(
            "stage_backbone_checkpoint_source='generation_decoder' requires a "
            f"v8 generation_model.pt, got: {generation_model_path}"
        )

    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise RuntimeError(
            "generation checkpoint is missing model_state_dict: "
            f"{generation_model_path}"
        )

    backbone_state = {}
    for key, value in model_state.items():
        if key.startswith("backbone."):
            backbone_state[key[len("backbone."):]] = value

    visible_mapped = 0
    for key, value in model_state.items():
        if key.startswith("visible_encoder."):
            suffix = key[len("visible_encoder."):]
            backbone_state[f"seq_encoder.{suffix}"] = value
            visible_mapped += 1

    if not backbone_state:
        raise RuntimeError(
            "generation checkpoint contained no backbone/visible_encoder "
            f"weights: {generation_model_path}"
        )

    interpolated_keys = {}

    def _filter_compatible_state_dict(module, state_dict):
        module_state = module.state_dict()
        filtered = {}
        skipped = {}
        for key, value in state_dict.items():
            if key not in module_state:
                continue
            if module_state[key].shape != value.shape:
                target = module_state[key]
                can_interpolate_pos_embed = (
                    bool(interpolate_pos_embed)
                    and key == "pos_embed"
                    and value.ndim == 3
                    and target.ndim == 3
                    and value.shape[0] == target.shape[0]
                    and value.shape[2] == target.shape[2]
                )
                if can_interpolate_pos_embed:
                    source_shape = list(value.shape)
                    resized = F.interpolate(
                        value.float().transpose(1, 2),
                        size=int(target.shape[1]),
                        mode="linear",
                        align_corners=True,
                    ).transpose(1, 2)
                    filtered[key] = resized.to(dtype=target.dtype)
                    interpolated_keys[key] = {
                        "source": source_shape,
                        "target": list(target.shape),
                        "mode": "linear",
                        "align_corners": True,
                    }
                    continue
                skipped[key] = {
                    "expected": list(module_state[key].shape),
                    "loaded": list(value.shape),
                }
                continue
            filtered[key] = value
        return filtered, skipped

    filtered, skipped = _filter_compatible_state_dict(backbone, backbone_state)
    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    source_meta = payload.get("source_backbone", {})
    print(
        "[EEG pretrain] loaded generation-decoder backbone: "
        f"{generation_model_path} | format_version="
        f"{payload.get('format_version')} | model_type="
        f"{payload.get('model_type')} | visible_encoder->seq_encoder="
        f"{visible_mapped} tensors | source_event_backbone="
        f"{source_meta.get('path')}"
    )
    if interpolated_keys:
        print(
            "[EEG pretrain] interpolated position embedding: "
            f"{interpolated_keys}"
        )
    if skipped:
        print(f"[EEG pretrain] skipped shape-mismatched keys: {skipped}")
    if missing:
        print(f"[EEG pretrain] missing keys: {missing}")
    if unexpected:
        print(f"[EEG pretrain] unexpected keys: {unexpected}")
    return True


def maybe_load_eeg_pretrained_weights(backbone, cfg):
    if not getattr(cfg, "use_pretrained_backbone", True):
        print("[EEG pretrain] use_pretrained_backbone=False; using random initialized backbone.")
        return False

    source = str(
        getattr(cfg, "stage_backbone_checkpoint_source", "event")
    ).strip().lower()
    if source in {"generation_decoder", "generation", "mae", "03generation"}:
        return load_generation_decoder_backbone_weights(
            backbone,
            getattr(cfg, "generation_full_model_path", None),
            interpolate_pos_embed=bool(
                getattr(cfg, "interpolate_pretrained_pos_embed", True)
            ),
        )
    if source not in {"event", "event_backbone", "03noneteacher", "none_teacher"}:
        raise ValueError(
            "stage_backbone_checkpoint_source must be 'event' or "
            f"'generation_decoder', got {source!r}."
        )
    return load_eeg_pretrained_weights(
        backbone,
        cfg.eeg_pretrained_model_path,
        cfg.load_full_backbone_pretrain,
        interpolate_pos_embed=bool(
            getattr(cfg, "interpolate_pretrained_pos_embed", True)
        ),
    )


def is_linear_probe(cfg):
    return str(getattr(cfg, "probe_mode", "finetune")).lower() == "linear_probe"


def experiment_metadata(cfg):
    return {
        "probe_mode": getattr(cfg, "probe_mode", "finetune"),
        "stage_backbone_checkpoint_source": str(
            getattr(cfg, "stage_backbone_checkpoint_source", "event")
        ),
        "generation_full_model_path": str(
            getattr(cfg, "generation_full_model_path", "") or ""
        ),
        "stage_output_dir": str(
            getattr(cfg, "stage_output_dir", "") or ""
        ),
        "seq_len": int(getattr(cfg, "seq_len", 0)),
        "stride": int(getattr(cfg, "stride", 0)),
        "interpolate_pretrained_pos_embed": bool(
            getattr(cfg, "interpolate_pretrained_pos_embed", False)
        ),
        "use_test_time_adabn": bool(
            getattr(cfg, "use_test_time_adabn", False)
        ),
        "adabn_scope": str(getattr(cfg, "adabn_scope", "disabled")),
        "adabn_timing": "selected_checkpoint_only",
        "use_stage_relative_spectral_branch": bool(
            getattr(cfg, "use_stage_relative_spectral_branch", False)
        ),
        "stage_spectral_sampling_rate": float(
            getattr(cfg, "stage_spectral_sampling_rate", 100.0)
        ),
        "stage_spectral_bands": [
            [float(low), float(high)]
            for low, high in getattr(cfg, "stage_spectral_bands", [])
        ],
        "stage_spectral_hidden_dim": int(
            getattr(cfg, "stage_spectral_hidden_dim", 0)
        ),
        "stage_spectral_dropout": float(
            getattr(cfg, "stage_spectral_dropout", 0.0)
        ),
        "stage_spectral_gate_init": float(
            getattr(cfg, "stage_spectral_gate_init", 0.0)
        ),
        "use_fold_self_supervised_adaptation": bool(
            getattr(cfg, "use_fold_self_supervised_adaptation", False)
        ),
        "fold_ssl_objective": "masked_epoch_fixed_teacher_cosine_prediction",
        "fold_ssl_epochs": int(getattr(cfg, "fold_ssl_epochs", 0)),
        "fold_ssl_patience": int(getattr(cfg, "fold_ssl_patience", 0)),
        "fold_ssl_learning_rate": float(
            getattr(cfg, "fold_ssl_learning_rate", 0.0)
        ),
        "fold_ssl_weight_decay": float(
            getattr(cfg, "fold_ssl_weight_decay", 0.0)
        ),
        "fold_ssl_mask_ratio": float(
            getattr(cfg, "fold_ssl_mask_ratio", 0.0)
        ),
        "fold_ssl_valid_subject_ratio": float(
            getattr(cfg, "fold_ssl_valid_subject_ratio", 0.0)
        ),
        "fold_ssl_batch_size": int(
            getattr(cfg, "fold_ssl_batch_size", 0)
        ),
        "fold_ssl_predictor_hidden_dim": int(
            getattr(cfg, "fold_ssl_predictor_hidden_dim", 0)
        ),
        "fold_ssl_labels_used": False,
        "fold_ssl_outer_test_used": False,
        "stage_head_type": getattr(cfg, "stage_head_type", "linear"),
        "stage_mlp_hidden_dim": int(getattr(cfg, "stage_mlp_hidden_dim", 0)),
        "stage_mlp_dropout": float(getattr(cfg, "stage_mlp_dropout", 0.0)),
        "stage_tcn_channels": int(getattr(cfg, "stage_tcn_channels", 0)),
        "stage_tcn_layers": int(getattr(cfg, "stage_tcn_layers", 0)),
        "stage_tcn_kernel_size": int(getattr(cfg, "stage_tcn_kernel_size", 0)),
        "stage_tcn_dropout": float(getattr(cfg, "stage_tcn_dropout", 0.0)),
        "freeze_backbone_epochs": int(getattr(cfg, "freeze_backbone_epochs", 0)),
        "stage_backbone_unfreeze_strategy": str(
            getattr(cfg, "stage_backbone_unfreeze_strategy", "full_backbone")
        ),
        "reset_early_stopping_on_unfreeze": bool(
            getattr(cfg, "reset_early_stopping_on_unfreeze", False)
        ),
        "enable_plateau_second_transformer_stage": bool(
            getattr(cfg, "enable_plateau_second_transformer_stage", False)
        ),
        "lr_last_transformer_stage2": float(
            getattr(cfg, "lr_last_transformer_stage2", 0.0)
        ),
        "lr_second_last_transformer": float(
            getattr(cfg, "lr_second_last_transformer", 0.0)
        ),
        "reset_early_stopping_on_second_unfreeze": bool(
            getattr(cfg, "reset_early_stopping_on_second_unfreeze", False)
        ),
        "use_pretrained_backbone": bool(getattr(cfg, "use_pretrained_backbone", True)),
        "use_stage_class_weights": bool(getattr(cfg, "use_stage_class_weights", False)),
        "early_stopping_metric": getattr(cfg, "early_stopping_metric", "acc"),
        "stage_loso_protocol": "all_but_one_train_one_test_fixed_final_epoch",
        "checkpoint_selection_uses_test_labels": False,
        "plateau_monitor": "training_loss",
        "stage_loso_use_inner_validation": bool(
            getattr(cfg, "stage_loso_use_inner_validation", False)
        ),
        "stage_loso_epochs": int(getattr(cfg, "stage_loso_epochs", 0)),
        "validation_epoch_aggregation": False,
        "test_epoch_aggregation": "mean_logits_by_subject_and_epoch",
        "refit_on_all_trainval": bool(getattr(cfg, "refit_on_all_trainval", False)),
        "use_viterbi_postprocess": bool(
            getattr(cfg, "use_viterbi_postprocess", False)
        ),
        "viterbi_transition_source": "training_subjects_only",
        "viterbi_transition_weight": float(
            getattr(cfg, "viterbi_transition_weight", 0.0)
        ),
        "viterbi_transition_weight_grid": [
            float(x) for x in getattr(cfg, "viterbi_transition_weight_grid", [])
        ],
        "stage_primary_method": str(cfg.saved_stage_primary_method),
        "stage_method_selection_rule": "fixed_before_training",
        "stage_method_selection_uses_test_labels": False,
        "stage_mc_gated_inference_seed": int(
            getattr(cfg, "stage_mc_gated_inference_seed", 0)
        ),
        "stage_mc_dropout_scope": str(
            getattr(cfg, "saved_stage_mc_dropout_scope", "none")
        ),
        "stage_mc_dropout_p": float(
            getattr(cfg, "saved_stage_mc_dropout_p", 0.0)
        ),
        "fixed_decoder_viterbi_weight": float(
            getattr(cfg, "saved_stage_viterbi_weight", 0.0)
        ),
        "fixed_decoder_gate_confidence_threshold": float(
            getattr(cfg, "saved_stage_gate_confidence_threshold", 0.0)
        ),
        "fixed_decoder_gate_margin_threshold": float(
            getattr(cfg, "saved_stage_gate_margin_threshold", 0.0)
        ),
        "fixed_decoder_gate_min_sequence_gain": float(
            getattr(cfg, "saved_stage_gate_min_sequence_gain", 0.0)
        ),
        "save_best_stage_visualizations": bool(
            getattr(cfg, "save_best_stage_visualizations", False)
        ),
        "stage_visualization_feature_source": (
            "TCN final hidden representation before classifier"
        ),
        "stage_attention_layer": int(
            getattr(cfg, "stage_attention_layer", -1)
        ),
        "experiment_name": str(getattr(cfg, "experiment_name", "") or ""),
    }


def _print_best_metrics(task_name, best_epoch, best_metrics, class_names):
    msg = (
        f"[Best] task={task_name} | epoch={best_epoch} | "
        f"acc={best_metrics.get('acc', 0.0):.4f}, "
        f"kappa={best_metrics.get('kappa', 0.0):.4f}, "
        f"macro_f1={best_metrics.get('macro_f1', 0.0):.4f}"
    )
    if "balanced_acc" in best_metrics:
        msg += f", balanced_acc={best_metrics.get('balanced_acc', 0.0):.4f}"
    if "auroc" in best_metrics:
        msg += f", auroc={best_metrics.get('auroc', 0.0):.4f}"
    print(msg)
    per_class_f1 = best_metrics.get("per_class_f1", {}) or {}
    if class_names is not None and len(per_class_f1) > 0:
        print("[Best] per-class F1:")
        for name in class_names:
            print(f"  {name}: {per_class_f1.get(name, 0.0):.4f}")


def run_one_fold(task_name, fold_idx, train_samples, val_samples, cfg, device, fold_dir):
    train_ds = SleepTextWindowDataset(train_samples, None, None, task_name)
    val_ds = SleepTextWindowDataset(val_samples, None, None, task_name)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size
    )
    maybe_load_eeg_pretrained_weights(backbone, cfg)

    model = MultiTaskSleepModel(backbone, cfg=cfg).to(device)
    stage_class_weights = None
    if task_name == "sleep_stage_classification" and cfg.use_stage_class_weights:
        stage_class_weights = compute_stage_class_weights(
            train_samples,
            cfg.tasks[task_name]["num_classes"]
        )
        print(f"[Stage Weights] {stage_class_weights.tolist()}")
    linear_probe = is_linear_probe(cfg)
    freeze_epochs = max(int(getattr(cfg, "freeze_backbone_epochs", 0)), 0)
    if linear_probe:
        set_backbone_trainable(model, False)
        print("[Linear Probe] Backbone frozen for the whole run; training task head only.")
    elif freeze_epochs > 0:
        set_backbone_trainable(model, False)
        print(f"[Freeze] Backbone frozen for epochs 1-{freeze_epochs}.")

    optimizer = build_optimizer(model, task_name, cfg)

    history = []
    best_state_dict = None
    best_metrics = None
    best_inference_seed = None
    best_score = -1e18
    best_epoch = -1
    patience_counter = 0

    class_names = cfg.tasks[task_name]["class_names"]
    validation_transition = None
    if task_name == "sleep_stage_classification":
        # A zero transition matrix and zero weight produce raw predictions after
        # aggregating every overlapping window back to one prediction per epoch.
        num_classes = cfg.tasks[task_name]["num_classes"]
        validation_transition = np.zeros((num_classes, num_classes), dtype=np.float32)

    for epoch in range(cfg.num_epochs):
        if (not linear_probe) and freeze_epochs > 0 and epoch == freeze_epochs:
            if task_name == "sleep_stage_classification":
                configure_stage_backbone_finetuning(model, cfg)
                if getattr(cfg, "reset_early_stopping_on_unfreeze", True):
                    patience_counter = 0
            else:
                set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, task_name, cfg)
            print(f"[Unfreeze] Fine-tuning scope changed from epoch {epoch + 1}.")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            task_name,
            cfg,
            class_weights=stage_class_weights,
        )

        if task_name == "sleep_stage_classification":
            val_metrics = evaluate_stage_viterbi(
                model, val_loader, device,
                log_transition=validation_transition,
                transition_weight=0.0,
                class_names=class_names,
                save_dir=fold_dir if epoch == cfg.num_epochs - 1 else None,
                seed=cfg.seed + fold_idx
            )
        elif task_name == "slow_wave_detection":
            val_metrics = evaluate_slow_wave(
                model, val_loader, device,
                class_names=class_names,
                save_dir=fold_dir if epoch == cfg.num_epochs - 1 else None,
                seed=cfg.seed + fold_idx
            )
        else:
            val_metrics = evaluate_disorder(
                model, val_loader, device,
                class_names=class_names,
                save_dir=fold_dir if epoch == cfg.num_epochs - 1 else None,
                seed=cfg.seed + fold_idx
            )

        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics
        }
        history.append(record)

        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}, kappa={train_metrics['kappa']:.4f}, macro_f1={train_metrics.get('macro_f1', 0.0):.4f} | "
            f"val acc={val_metrics['acc']:.4f}, kappa={val_metrics['kappa']:.4f}, macro_f1={val_metrics.get('macro_f1', 0.0):.4f}"
        )

        current_score = pick_best_metric(val_metrics, cfg)
        is_better, selection_reason = is_better_checkpoint(
            val_metrics,
            best_metrics,
            cfg,
            primary_min_delta=cfg.early_stopping_min_delta,
        )
        if is_better:
            best_score = current_score
            best_epoch = epoch + 1
            best_metrics = val_metrics
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(
                f"[New Best] epoch={best_epoch} | reason={selection_reason} | "
                f"acc={best_metrics['acc']:.4f}, kappa={best_metrics['kappa']:.4f}, "
                f"macro_f1={best_metrics.get('macro_f1', 0.0):.4f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}")
                break

    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_metrics = history[-1]["val"] if len(history) > 0 else {"acc": 0.0, "kappa": 0.0, "macro_f1": 0.0, "per_class_f1": {}, "report": {}}
        best_epoch = len(history)

    # 用最佳模型重新评估一次，保存最佳模型的混淆矩阵 / TSNE，并拿到最终最佳指标
    best_backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size
    )
    best_model = MultiTaskSleepModel(best_backbone, cfg=cfg).to(device)
    best_model.load_state_dict(best_state_dict, strict=True)

    if task_name == "sleep_stage_classification":
        best_metrics = evaluate_stage_viterbi(
            best_model,
            val_loader,
            device,
            log_transition=validation_transition,
            transition_weight=0.0,
            class_names=class_names,
            save_dir=fold_dir,
            seed=cfg.seed + fold_idx,
        )
    elif task_name == "slow_wave_detection":
        best_metrics = evaluate_slow_wave(best_model, val_loader, device, class_names=class_names, save_dir=fold_dir, seed=cfg.seed + fold_idx)
    else:
        best_metrics = evaluate_disorder(best_model, val_loader, device, class_names=class_names, save_dir=fold_dir, seed=cfg.seed + fold_idx)

    # _print_best_metrics(task_name, best_epoch, best_metrics, class_names)

    with open(os.path.join(fold_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    print_best_metrics(task_name, best_epoch, best_metrics)

    return {
        "model": best_state_dict,
        "metrics": best_metrics,
        "history": history,
        "best_epoch": best_epoch,
        **experiment_metadata(cfg),
    }


def train_stage_fixed_final(
    train_samples,
    test_samples,
    cfg,
    device,
    num_epochs,
    fold_idx,
    test_dir,
    log_transition=None,
):
    """Train all remaining subjects for fixed epochs; evaluate the test subject once."""
    num_epochs = max(int(num_epochs), 1)
    primary_method = str(cfg.saved_stage_primary_method).strip().lower()
    if primary_method not in {"raw", "viterbi", "gated_viterbi"}:
        raise ValueError(f"Unsupported fixed stage decoder: {primary_method!r}")
    if primary_method != "raw" and log_transition is None:
        raise ValueError(
            "Viterbi is enabled, but no training-set transition matrix was provided."
        )
    set_seed(cfg.seed + 10000 + int(fold_idx))

    train_ds = SleepTextWindowDataset(
        train_samples, None, None, "sleep_stage_classification"
    )
    test_ds = SleepTextWindowDataset(
        test_samples, None, None, "sleep_stage_classification"
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size,
    )
    maybe_load_eeg_pretrained_weights(backbone, cfg)
    fold_ssl_metadata = {"enabled": False}
    if getattr(cfg, "use_fold_self_supervised_adaptation", False):
        # Restore all RNG streams afterwards so adding the SSL phase does not
        # silently change TCN/head initialization or downstream shuffling.
        downstream_rng_state = _capture_rng_state()
        backbone, fold_ssl_metadata = adapt_backbone_on_fold_train_subjects(
            backbone,
            train_samples,
            cfg,
            device,
            fold_idx,
        )
        _restore_rng_state(downstream_rng_state)
        adapted_backbone_path = os.path.join(
            os.path.dirname(test_dir),
            "fold_ssl_adapted_backbone.pt",
        )
        torch.save(
            {
                "backbone_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in backbone.state_dict().items()
                },
                "fold_ssl_adaptation": fold_ssl_metadata,
                "source_pretrained_checkpoint": str(
                    getattr(cfg, "eeg_pretrained_model_path", "")
                ),
            },
            adapted_backbone_path,
        )
        fold_ssl_metadata["checkpoint_path"] = adapted_backbone_path
        print(
            f"[Fold SSL Complete] Fold {fold_idx} | "
            f"best_epoch={fold_ssl_metadata['best_epoch']} | "
            f"best_valid_loss={fold_ssl_metadata['best_valid_loss']:.6f} | "
            f"checkpoint={adapted_backbone_path} | "
            "downstream_rng_restored=True"
        )
    model = MultiTaskSleepModel(backbone, cfg=cfg).to(device)
    if getattr(model, "stage_spectral_encoder", None) is not None:
        spectral_params = sum(
            p.numel() for p in model.stage_spectral_encoder.parameters()
        ) + model.stage_spectral_gate_logit.numel()
        spectral_mode = str(
            getattr(cfg, "stage_spectral_mode", "relative_band_power")
        )
        print(
            f"[Stage Spectral] mode={spectral_mode} | "
            f"sampling_rate={cfg.stage_spectral_sampling_rate:.1f}Hz | "
            f"gate_init={torch.sigmoid(model.stage_spectral_gate_logit).item():.4f} | "
            f"trainable_params={spectral_params} | "
            "source=waveform_seen_by_classifier"
        )

    class_weights = None
    if cfg.use_stage_class_weights:
        class_weights = compute_stage_class_weights(
            train_samples,
            cfg.tasks["sleep_stage_classification"]["num_classes"],
        )

    linear_probe = is_linear_probe(cfg)
    freeze_epochs = max(int(getattr(cfg, "freeze_backbone_epochs", 0)), 0)
    if linear_probe or freeze_epochs > 0:
        set_backbone_trainable(model, False)
    optimizer = build_optimizer(model, "sleep_stage_classification", cfg)

    class_names = cfg.tasks["sleep_stage_classification"]["class_names"]
    history = []
    best_train_loss = float("inf")
    patience_counter = 0
    finetune_phase = "head_only" if freeze_epochs > 0 else "last_transformer"
    second_stage_started = False

    for epoch in range(num_epochs):
        if (not linear_probe) and freeze_epochs > 0 and epoch == freeze_epochs:
            configure_stage_backbone_finetuning(model, cfg)
            optimizer = build_optimizer(model, "sleep_stage_classification", cfg)
            finetune_phase = "last_transformer"
            best_train_loss = float("inf")
            patience_counter = 0
            print(f"[Restricted Unfreeze] Fold {fold_idx} | epoch={epoch + 1}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            "sleep_stage_classification", cfg, class_weights=class_weights,
        )
        train_loss = float(train_metrics["loss"])
        if train_loss < best_train_loss - float(cfg.early_stopping_min_delta):
            best_train_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
        current_spectral_gate = None
        if getattr(model, "stage_spectral_gate_logit", None) is not None:
            current_spectral_gate = float(
                torch.sigmoid(model.stage_spectral_gate_logit).detach().cpu().item()
            )
        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "finetune_phase": finetune_phase,
            "spectral_gate": current_spectral_gate,
            "plateau_monitor": "training_loss",
            "patience_counter": int(patience_counter),
        })
        print(
            f"[LOSO Train] Fold {fold_idx} | epoch {epoch + 1}/{num_epochs} | "
            f"phase={finetune_phase} | train loss={train_loss:.4f}, "
            f"acc={train_metrics['acc']:.4f}"
        )
        if (
            not linear_probe
            and bool(getattr(cfg, "enable_plateau_second_transformer_stage", False))
            and not second_stage_started
            and finetune_phase == "last_transformer"
            and patience_counter >= int(cfg.early_stopping_patience)
            and epoch + 1 < num_epochs
        ):
            enter_second_transformer_finetuning_stage(model, optimizer, cfg)
            second_stage_started = True
            finetune_phase = "last_two_transformers"
            best_train_loss = float("inf")
            patience_counter = 0
            print(
                f"[Train-Loss Plateau Transition] Fold {fold_idx} | "
                f"trigger_epoch={epoch + 1} | continue_to_epoch={num_epochs}"
            )

    # The checkpoint is fixed before any test labels are scored.
    final_epoch = num_epochs
    inference_seed = int(cfg.stage_mc_gated_inference_seed)
    if getattr(cfg, "use_test_time_adabn", False):
        adapt_batchnorm_to_test_subject(model, test_loader, device)
    final_evaluation = evaluate_stage_fixed_method_from_model(
        model, test_loader, cfg, device, log_transition, inference_seed,
        return_aggregated=bool(getattr(cfg, "save_best_stage_visualizations", False)),
    )
    final_metrics = final_evaluation["primary_metrics"]
    final_state_dict = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    visualization_aggregated = final_evaluation.pop("_aggregated", None)
    visualization_metadata = None
    if visualization_aggregated is not None:
        visualization_metadata = save_selected_best_stage_visualizations(
            model=model, loader=test_loader, aggregated=visualization_aggregated,
            cfg=cfg, device=device, log_transition=log_transition,
            best_epoch=final_epoch, primary_method=primary_method,
            primary_metrics=final_metrics, fold_idx=fold_idx,
            save_dir=os.path.join(os.path.dirname(test_dir), "viz"),
            inference_seed=inference_seed,
        )
        del visualization_aggregated
    print(
        f"[LOSO Final Test] Fold {fold_idx} | epoch={final_epoch} | "
        f"method={primary_method} | acc={final_metrics['acc']:.4f} | "
        f"macro_f1={final_metrics.get('macro_f1', 0.0):.4f}"
    )
    inference_profile = {
        "primary_method": primary_method,
        "selection_rule": "fixed_before_training",
        "uses_test_labels_for_method_selection": False,
        "mc_seed": inference_seed,
        "dropout_scope": str(cfg.saved_stage_mc_dropout_scope),
        "dropout_p": float(cfg.saved_stage_mc_dropout_p),
        "viterbi_weight": float(cfg.saved_stage_viterbi_weight),
        "gate_confidence_threshold": float(cfg.saved_stage_gate_confidence_threshold),
        "gate_margin_threshold": float(cfg.saved_stage_gate_margin_threshold),
        "gate_min_sequence_gain": float(cfg.saved_stage_gate_min_sequence_gain),
    }
    del model, backbone, train_loader, test_loader, train_ds, test_ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model": final_state_dict,
        "metrics": final_metrics,
        "inference_seed": inference_seed,
        "primary_method": primary_method,
        "inference_profile": inference_profile,
        "history": history,
        # Preserve the checkpoint key expected by existing loaders.
        "best_epoch": final_epoch,
        "final_epoch": final_epoch,
        "selection_metric": "fixed_final_epoch",
        "selection_tiebreak_metric": "none",
        "selection_policy": "fixed_final_epoch",
        "test_used_for_model_selection": False,
        "best_selection_score": None,
        "second_stage_started": bool(second_stage_started),
        "fold_ssl_adaptation": fold_ssl_metadata,
        "best_visualizations": visualization_metadata,
    }


def logo_run(task_name, samples, cfg, device):
    if len(samples) == 0:
        print(f"{task_name}: 没有可用样本，跳过")
        return None

    groups = np.array([s["subject_id"] for s in samples])
    dummy_x = np.arange(len(samples))
    dummy_y = np.zeros(len(samples))

    logo = LeaveOneGroupOut()
    unique_subjects = np.unique(groups)
    total_folds = len(unique_subjects)

    fold_results = []
    task_out_dir = task_output_dir(cfg, task_name)
    ensure_dir(task_out_dir)

    for fold_idx, (train_idx, val_idx) in enumerate(
        logo.split(dummy_x, dummy_y, groups=groups)
    ):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        val_subjects = sorted(list(set(groups[val_idx])))

        print(f"\n{'=' * 80}")
        print(f"任务: {task_name} | LOSO Fold {fold_idx + 1}/{total_folds}")
        print(f"留出的验证被试: {val_subjects}")
        print(f"训练窗口数: {len(train_samples)} | 验证窗口数: {len(val_samples)}")
        print(f"{'=' * 80}")

        if len(val_subjects) == 1:
            fold_name = f"subject_{val_subjects[0]}"
        else:
            fold_name = f"fold_{fold_idx + 1}"

        fold_dir = os.path.join(task_out_dir, fold_name)
        ensure_dir(fold_dir)

        best_state = run_one_fold(
            task_name, fold_idx, train_samples, val_samples, cfg, device, fold_dir
        )

        model_path = os.path.join(fold_dir, "best_model.pth")
        torch.save(best_state["model"], model_path)

        result_json = {
            "task_name": task_name,
            **experiment_metadata(cfg),
            "fold_idx": fold_idx + 1,
            "val_subjects": val_subjects,
            "metrics": best_state["metrics"],
            "history": best_state["history"],
            "best_epoch": best_state.get("best_epoch", None),
        }
        with open(os.path.join(fold_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        fold_results.append(result_json)

    summary = {
        "task_name": task_name,
        **experiment_metadata(cfg),
        "n_folds": len(fold_results),
        "fold_results": fold_results
    }

    if len(fold_results) > 0:
        accs = [fr["metrics"].get("acc", 0.0) for fr in fold_results]
        kappas = [fr["metrics"].get("kappa", 0.0) for fr in fold_results]
        macro_f1s = [fr["metrics"].get("macro_f1", 0.0) for fr in fold_results]
        weighted_f1s = [fr["metrics"].get("weighted_f1", 0.0) for fr in fold_results]

        summary["mean_acc"] = float(np.mean(accs))
        summary["std_acc"] = float(np.std(accs))
        summary["mean_kappa"] = float(np.mean(kappas))
        summary["std_kappa"] = float(np.std(kappas))
        summary["mean_macro_f1"] = float(np.mean(macro_f1s))
        summary["std_macro_f1"] = float(np.std(macro_f1s))
        summary["mean_weighted_f1"] = float(np.mean(weighted_f1s))
        summary["std_weighted_f1"] = float(np.std(weighted_f1s))

    with open(os.path.join(task_out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{task_name} 完成。")
    if "mean_acc" in summary:
        print(
            f"Mean acc={summary['mean_acc']:.4f}±{summary['std_acc']:.4f}, "
            f"mean kappa={summary['mean_kappa']:.4f}±{summary['std_kappa']:.4f}, "
            f"mean macro_f1={summary['mean_macro_f1']:.4f}±{summary['std_macro_f1']:.4f}, "
            f"mean weighted_f1={summary['mean_weighted_f1']:.4f}±{summary['std_weighted_f1']:.4f}"
        )

    return summary


def train_full_task(task_name, train_samples, cfg, device, save_dir):
    ensure_dir(save_dir)

    train_ds = SleepTextWindowDataset(train_samples, None, None, task_name)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )

    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size
    )
    maybe_load_eeg_pretrained_weights(backbone, cfg)
    model = MultiTaskSleepModel(backbone, cfg=cfg).to(device)

    stage_class_weights = None
    if task_name == "sleep_stage_classification" and cfg.use_stage_class_weights:
        stage_class_weights = compute_stage_class_weights(
            train_samples,
            cfg.tasks[task_name]["num_classes"]
        )
        print(f"[Stage Weights] {stage_class_weights.tolist()}")
    linear_probe = is_linear_probe(cfg)
    freeze_epochs = max(int(getattr(cfg, "freeze_backbone_epochs", 0)), 0)
    if linear_probe:
        set_backbone_trainable(model, False)
        print("[Linear Probe] Backbone frozen for the whole run; training task head only.")
    elif freeze_epochs > 0:
        set_backbone_trainable(model, False)
        print(f"[Freeze] Backbone frozen for epochs 1-{freeze_epochs}.")

    optimizer = build_optimizer(model, task_name, cfg)

    best_loss = 1e18
    best_state = None
    history = []

    for epoch in range(cfg.num_epochs):
        if (not linear_probe) and freeze_epochs > 0 and epoch == freeze_epochs:
            if task_name == "sleep_stage_classification":
                configure_stage_backbone_finetuning(model, cfg)
            else:
                set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, task_name, cfg)
            print(f"[Unfreeze] Fine-tuning scope changed from epoch {epoch + 1}.")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            task_name,
            cfg,
            class_weights=stage_class_weights,
        )
        history.append({"epoch": epoch + 1, "train": train_metrics})
        print(
            f"[Full Train] Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}, kappa={train_metrics['kappa']:.4f}"
        )

        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model_path = os.path.join(save_dir, "best_model_full_train.pth")
    torch.save(best_state, model_path)

    with open(os.path.join(save_dir, "full_train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return best_state


def evaluate_external_stage(state_dict, test_samples, cfg, device):
    test_ds = SleepTextWindowDataset(
        test_samples, None, None, "sleep_stage_classification"
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size
    )
    maybe_load_eeg_pretrained_weights(backbone, cfg)
    model = MultiTaskSleepModel(backbone, cfg=cfg).to(device)
    model.load_state_dict(state_dict, strict=True)

    metrics = evaluate_stage(
        model,
        test_loader,
        device,
        class_names=cfg.tasks["sleep_stage_classification"]["class_names"],
    )
    return metrics


def evaluate_stage_from_state_dict(
    state_dict,
    test_samples,
    cfg,
    device,
    save_dir=None,
    seed=None,
    log_transition=None,
    transition_weight=None,
):
    test_ds = SleepTextWindowDataset(
        test_samples, None, None, "sleep_stage_classification"
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size
    )
    maybe_load_eeg_pretrained_weights(backbone, cfg)
    model = MultiTaskSleepModel(backbone, cfg=cfg).to(device)
    model.load_state_dict(state_dict, strict=True)

    class_names = cfg.tasks["sleep_stage_classification"]["class_names"]
    if getattr(cfg, "use_viterbi_postprocess", False) and log_transition is not None:
        if transition_weight is None:
            transition_weight = getattr(cfg, "viterbi_transition_weight", 0.5)
        return evaluate_stage_viterbi(
            model,
            test_loader,
            device,
            log_transition=log_transition,
            transition_weight=transition_weight,
            class_names=class_names,
            save_dir=save_dir,
            seed=cfg.seed if seed is None else seed,
        )

    return evaluate_stage(
        model,
        test_loader,
        device,
        class_names=class_names,
        save_dir=save_dir,
        seed=cfg.seed if seed is None else seed,
    )


def evaluate_saved_stage_mc_gated_from_state_dict(
    state_dict,
    test_samples,
    cfg,
    device,
    log_transition,
    inference_seed=None,
):
    """Evaluate a saved final-epoch model with the configured single decoder."""
    dataset = SleepTextWindowDataset(
        test_samples, None, None, "sleep_stage_classification"
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size,
    )
    model = MultiTaskSleepModel(backbone, cfg=cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    result = evaluate_stage_fixed_method_from_model(
        model, loader, cfg, device, log_transition,
        int(cfg.saved_stage_mc_seed if inference_seed is None else inference_seed),
    )
    del model, backbone, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_saved_model_checkpoint(model_path):
    """Load model weights plus optional reproducible inference metadata."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    metadata = {}
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        metadata = {
            key: value for key, value in checkpoint.items() if key != "model"
        }
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint content: {model_path}")
    return state_dict, metadata


def load_saved_model_state_dict(model_path):
    """Load only a saved task model state dict."""
    state_dict, _ = load_saved_model_checkpoint(model_path)
    return state_dict


def saved_stage_loso_test_run(
    records,
    cfg,
    device,
    checkpoint_root_override=None,
):
    """Re-test each DREAMS subject with only its matching held-out checkpoint."""
    if len(records) == 0:
        print("saved sleep-stage test: no valid subject records, skip.")
        return None

    task_name = "sleep_stage_classification"
    checkpoint_root = checkpoint_root_override
    if not checkpoint_root:
        checkpoint_root = getattr(cfg, "saved_stage_checkpoint_dir", None)
    if not checkpoint_root:
        checkpoint_root = task_output_dir(cfg, task_name)
    checkpoint_root = os.path.abspath(checkpoint_root)

    subject_ids = sorted({str(r["subject_id"]) for r in records})
    max_folds = getattr(cfg, "saved_stage_max_folds", None)
    if max_folds is not None:
        subject_ids = subject_ids[:max(1, int(max_folds))]
    checkpoint_paths = {
        subject_id: os.path.join(
            checkpoint_root,
            f"subject_{subject_id}",
            "best_model.pth",
        )
        for subject_id in subject_ids
    }
    missing = [path for path in checkpoint_paths.values() if not os.path.isfile(path)]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Saved LOSO test cannot start because checkpoints are missing:\n"
            f"{missing_text}"
        )

    output_name = str(
        getattr(cfg, "saved_stage_test_output_name", "test_only_recheck")
        or "test_only_recheck"
    ).strip()
    test_root = getattr(cfg, "saved_stage_test_output_dir", None)
    if not test_root:
        test_root = os.path.join(checkpoint_root, output_name)
    test_root = os.path.abspath(test_root)
    ensure_dir(test_root)

    print(f"[Saved LOSO Test] checkpoint root: {checkpoint_root}")
    print(f"[Saved LOSO Test] subjects/checkpoints: {len(subject_ids)}")
    print("[Saved LOSO Test] training is disabled; model parameters will not be updated.")

    use_mc_gated = True
    print(f"[Saved LOSO Test] fixed decoder={cfg.saved_stage_primary_method}")

    fold_results = []
    num_classes = cfg.tasks[task_name]["num_classes"]
    for fold_idx, subject_id in enumerate(subject_ids, 1):
        train_records = [r for r in records if str(r["subject_id"]) != subject_id]
        test_records = [r for r in records if str(r["subject_id"]) == subject_id]
        train_samples = build_stage_samples(
            train_records,
            seq_len=cfg.seq_len,
            stride=cfg.stride,
        )
        test_samples = build_stage_samples(
            test_records,
            seq_len=cfg.seq_len,
            stride=cfg.stride,
        )

        log_transition = None
        if str(cfg.saved_stage_primary_method).lower() != "raw":
            log_transition = build_stage_transition_log_probs(
                train_samples,
                num_classes,
                smoothing=cfg.viterbi_transition_smoothing,
            )

        model_path = checkpoint_paths[subject_id]
        subject_output_dir = os.path.join(test_root, f"subject_{subject_id}")
        ensure_dir(subject_output_dir)
        print(
            f"\n[Saved LOSO Test] {fold_idx}/{len(subject_ids)} | "
            f"test subject={subject_id} | model={model_path}"
        )

        state_dict, checkpoint_metadata = load_saved_model_checkpoint(model_path)
        # Refuse old test-selected checkpoints; their bias cannot be removed by re-testing.
        if (
            checkpoint_metadata.get("selection_policy") != "fixed_final_epoch"
            or checkpoint_metadata.get("test_used_for_model_selection") is not False
        ):
            raise ValueError(
                f"{model_path} is not a fixed-final-epoch LOSO checkpoint. "
                "Re-train with sleep_stage_run_mode='train'."
            )
        saved_inference_profile = checkpoint_metadata.get("inference_profile")
        postprocess = evaluate_saved_stage_mc_gated_from_state_dict(
            state_dict, test_samples, cfg, device, log_transition,
            inference_seed=int(cfg.stage_mc_gated_inference_seed),
        )
        metrics = postprocess["primary_metrics"]
        selected_primary_method = postprocess["primary_method"]
        print(
            f"[Saved LOSO Test] subject={subject_id} | method={selected_primary_method} | "
            f"acc={metrics['acc']:.4f} | kappa={metrics['kappa']:.4f} | "
            f"macro_f1={metrics.get('macro_f1', 0.0):.4f}"
        )

        result = {
            "fold_idx": fold_idx,
            "test_subject": subject_id,
            "model_path": model_path,
            "checkpoint_inference_profile": saved_inference_profile,
            "primary_method": selected_primary_method,
            "test_metrics": metrics,
        }
        if postprocess is not None:
            result.update(postprocess)
        with open(
            os.path.join(subject_output_dir, "test_metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        fold_results.append(result)

        del state_dict, train_samples, test_samples
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "task_name": "saved_sleep_stage_loso_test",
        **experiment_metadata(cfg),
        "checkpoint_root": checkpoint_root,
        "use_saved_stage_mc_gated_postprocess": use_mc_gated,
        "primary_method": str(cfg.saved_stage_primary_method),
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }
    for metric_name in ["acc", "kappa", "macro_f1", "weighted_f1"]:
        values = [
            float(result["test_metrics"].get(metric_name, 0.0))
            for result in fold_results
        ]
        summary[f"mean_{metric_name}"] = float(np.mean(values))
        summary[f"std_{metric_name}"] = float(np.std(values))

    if use_mc_gated:
        summary["fixed_inference_profile"] = {
            "mc_seed": int(cfg.saved_stage_mc_seed),
            "dropout_scope": str(cfg.saved_stage_mc_dropout_scope),
            "dropout_p": float(cfg.saved_stage_mc_dropout_p),
            "viterbi_weight": float(cfg.saved_stage_viterbi_weight),
            "gate_confidence_threshold": float(
                cfg.saved_stage_gate_confidence_threshold
            ),
            "gate_margin_threshold": float(
                cfg.saved_stage_gate_margin_threshold
            ),
            "gate_min_sequence_gain": float(
                cfg.saved_stage_gate_min_sequence_gain
            ),
            "subject_order": subject_ids,
        }
    summary_path = os.path.join(test_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "\n[Saved LOSO Test] completed | "
        f"method={summary['primary_method']} | "
        f"Mean acc={summary['mean_acc']:.4f}+-{summary['std_acc']:.4f} | "
        f"mean kappa={summary['mean_kappa']:.4f}+-{summary['std_kappa']:.4f} | "
        f"mean macro_f1={summary['mean_macro_f1']:.4f}+-{summary['std_macro_f1']:.4f}"
    )
    print(f"[Saved LOSO Test] summary: {summary_path}")
    return summary


def aggregate_stage_from_state_dict(state_dict, samples, cfg, device):
    """Run one forward pass and retain compact unique-epoch emissions."""
    dataset = SleepTextWindowDataset(
        samples, None, None, "sleep_stage_classification"
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size,
    )
    model = MultiTaskSleepModel(backbone, cfg=cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    by_subject, _ = _aggregate_stage_logits_by_epoch(model, loader, device)
    del model, backbone, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return by_subject


def _split_stage_train_val_records(trainval_records, cfg, fold_idx):
    if len(trainval_records) < 2:
        return trainval_records, trainval_records

    inner_idx = np.arange(len(trainval_records))
    val_size = max(1, int(round(len(trainval_records) * 0.2)))
    val_size = min(val_size, len(trainval_records) - 1)

    tr_idx, val_idx = train_test_split(
        inner_idx,
        test_size=val_size,
        random_state=cfg.seed + fold_idx,
        shuffle=True,
    )

    train_records = [trainval_records[i] for i in tr_idx]
    val_records = [trainval_records[i] for i in val_idx]
    return train_records, val_records


def stage_subject_loso_cv_run(records, cfg, device):
    if len(records) == 0:
        print("sleep_stage_classification: no valid subject records, skip.")
        return None

    task_name = "sleep_stage_classification"
    subject_ids = np.array([str(r["subject_id"]) for r in records])
    dummy_x = np.arange(len(records))
    dummy_y = np.zeros(len(records), dtype=np.int64)

    logo = LeaveOneGroupOut()
    total_folds = len(np.unique(subject_ids))
    task_out_dir = task_output_dir(cfg, task_name)
    ensure_dir(task_out_dir)
    fixed_epochs = max(int(getattr(cfg, "stage_loso_epochs", 20)), 1)

    if getattr(cfg, "stage_loso_use_inner_validation", False):
        raise ValueError(
            "This LOSO path is configured for 19-train/1-test without inner validation."
        )
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(dummy_x, dummy_y, groups=subject_ids), 1
    ):
        max_folds = getattr(cfg, "stage_loso_max_folds", None)
        if max_folds is not None and fold_idx > max(1, int(max_folds)):
            break
        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]
        train_samples = build_stage_samples(train_records, seq_len=cfg.seq_len, stride=cfg.stride)
        test_samples = build_stage_samples(
            test_records, seq_len=cfg.seq_len, stride=cfg.stride
        )
        log_transition = None
        if str(cfg.saved_stage_primary_method).lower() != "raw":
            log_transition = build_stage_transition_log_probs(
                train_samples,
                cfg.tasks[task_name]["num_classes"],
                smoothing=cfg.viterbi_transition_smoothing,
            )

        test_subjects = sorted({str(r["subject_id"]) for r in test_records})
        train_subjects = sorted({str(r["subject_id"]) for r in train_records})

        print(f"\n{'=' * 80}")
        print(f"Task: {task_name} | LOSO all-but-one-train/1-test Fold {fold_idx}/{total_folds}")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print("Val subjects: [] (disabled)")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
        print(f"Train windows: {len(train_samples)} | epochs: {fixed_epochs}")
        print(f"Test windows: {len(test_samples)} (evaluated once after fixed training)")
        print(f"{'=' * 80}")

        fold_name = f"subject_{test_subjects[0]}" if len(test_subjects) == 1 else f"fold_{fold_idx}"
        fold_dir = os.path.join(task_out_dir, fold_name)
        test_dir = os.path.join(fold_dir, "outer_test")
        ensure_dir(fold_dir)
        ensure_dir(test_dir)

        print(
            f"[LOSO Train] Fold {fold_idx} | subjects={len(train_subjects)} | "
            f"windows={len(train_samples)} | epochs={fixed_epochs} | "
            f"checkpoint=fixed_final_epoch | decoder={cfg.saved_stage_primary_method}"
        )
        best_state = train_stage_fixed_final(
            train_samples,
            test_samples,
            cfg,
            device,
            fixed_epochs,
            fold_idx,
            test_dir,
            log_transition=log_transition,
        )

        model_path = os.path.join(fold_dir, "best_model.pth")
        checkpoint_payload = {
            "model": best_state["model"],
            "final_epoch": int(best_state["final_epoch"]),
            "selection_policy": "fixed_final_epoch",
            "test_used_for_model_selection": False,
            "best_epoch": int(best_state["best_epoch"]),
            "primary_method": str(best_state.get("primary_method", "viterbi")),
            "selection_metric": str(best_state["selection_metric"]),
            "selection_tiebreak_metric": str(
                best_state["selection_tiebreak_metric"]
            ),
            "inference_profile": best_state.get("inference_profile"),
            "visualization_metadata": best_state.get("best_visualizations"),
            "saved_metrics": {
                "primary": best_state.get("metrics"),
            },
        }
        torch.save(checkpoint_payload, model_path)
        print(
            f"[Checkpoint Saved] Fold {fold_idx} | path={model_path} | "
            f"best_epoch={best_state['best_epoch']} | "
            f"primary={best_state.get('primary_method')} | "
            f"inference_seed={best_state.get('inference_seed')} | "
            "contains=final_model+fixed_inference_profile+test_metrics"
        )
        del train_samples
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        test_metrics = best_state["metrics"]
        print(
            f"[Test] Fold {fold_idx} | "
            f"acc={test_metrics['acc']:.4f}, "
            f"kappa={test_metrics['kappa']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"weighted_f1={test_metrics.get('weighted_f1', 0.0):.4f}"
        )
        print_test_metrics_detail(test_metrics)
        saved_test_metrics = {
            "primary_method": best_state.get("primary_method", "viterbi"),
            "primary_metrics": test_metrics,
            "inference_profile": best_state.get("inference_profile"),
        }
        with open(
            os.path.join(fold_dir, "test_metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(saved_test_metrics, f, ensure_ascii=False, indent=2)

        result_json = {
            "task_name": task_name,
            **experiment_metadata(cfg),
            "fold_idx": fold_idx,
            "train_subjects": train_subjects,
            "val_subjects": [],
            "final_train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "val_metrics": None,
            "test_metrics": test_metrics,
            "primary_test_method": best_state.get(
                "primary_method", "viterbi"
            ),
            "saved_inference_profile": best_state.get("inference_profile"),
            "best_visualizations": best_state.get("best_visualizations"),
            "test_time_adaptation_dataset": (
                "outer_test_unlabeled"
                if getattr(cfg, "use_test_time_adabn", False)
                else None
            ),
            "test_used_for_model_selection": False,
            "selection_dataset": "none_fixed_final_epoch",
            "test_evaluation_level": test_metrics.get("evaluation_level", "unknown"),
            "test_epoch_aggregation": test_metrics.get("aggregation", None),
            "selection_metric": best_state["selection_metric"],
            "best_selection_score": best_state["best_selection_score"],
            "selected_viterbi_transition_weight": (
                float(best_state["inference_profile"]["viterbi_weight"])
                if (
                    isinstance(best_state.get("inference_profile"), dict)
                    and best_state.get("primary_method")
                    in {"viterbi", "gated_viterbi"}
                )
                else 0.0
            ),
            "viterbi_transition_search": [],
            "history": best_state["history"],
            "fold_ssl_adaptation": best_state.get(
                "fold_ssl_adaptation", {"enabled": False}
            ),
            "best_epoch": best_state["best_epoch"],
            "final_epoch": best_state["final_epoch"],
            "training_epochs": len(best_state["history"]),
            "refit_on_all_trainval": False,
        }
        with open(os.path.join(fold_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        fold_results.append(result_json)

        del test_samples, best_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "task_name": task_name,
        **experiment_metadata(cfg),
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }

    if len(fold_results) > 0:
        metric_names = ["acc", "kappa", "macro_f1", "weighted_f1"]
        for metric_name in metric_names:
            metric_values = [
                fr["test_metrics"].get(metric_name, 0.0) for fr in fold_results
            ]
            summary[f"mean_{metric_name}"] = float(np.mean(metric_values))
            summary[f"std_{metric_name}"] = float(np.std(metric_values))

    with open(os.path.join(task_out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{task_name} completed.")
    if "mean_acc" in summary:
        print(
            f"Mean acc={summary['mean_acc']:.4f}+-{summary['std_acc']:.4f}, "
            f"mean kappa={summary['mean_kappa']:.4f}+-{summary['std_kappa']:.4f}, "
            f"mean macro_f1={summary['mean_macro_f1']:.4f}+-{summary['std_macro_f1']:.4f}, "
            f"mean weighted_f1={summary['mean_weighted_f1']:.4f}+-{summary['std_weighted_f1']:.4f}"
        )
    return summary


def stage_external_generalization_run(stage_records, external_samples, cfg, device):
    if len(stage_records) == 0:
        print("sleep_stage_external_generalization: no training subjects, skip.")
        return None
    if len(external_samples) == 0:
        print("sleep_stage_external_generalization: no external samples, skip.")
        return None

    task_name = "sleep_stage_external_generalization"
    task_out_dir = task_output_dir(cfg, task_name)
    ensure_dir(task_out_dir)

    train_samples = build_stage_samples(
        stage_records,
        seq_len=cfg.seq_len,
        stride=cfg.stride
    )
    if len(train_samples) == 0:
        print("sleep_stage_external_generalization: no train windows after stage sample build, skip.")
        return None

    print(f"\n{'=' * 80}")
    print("Task: sleep_stage_external_generalization | DREAMS train -> ISRUC test")
    print(f"Train subjects: {len(stage_records)} | Train windows: {len(train_samples)}")
    print(f"External windows: {len(external_samples)}")
    print(f"{'=' * 80}")

    best_state = train_full_task(
        "sleep_stage_classification",
        train_samples,
        cfg,
        device,
        task_out_dir
    )

    external_metrics = evaluate_external_stage(
        best_state,
        external_samples,
        cfg,
        device
    )

    result = {
        "task_name": task_name,
        "external_acc": float(external_metrics.get("acc", 0.0)),
        "external_kappa": float(external_metrics.get("kappa", 0.0)),
        "external_macro_f1": float(external_metrics.get("macro_f1", 0.0)),
        "external_weighted_f1": float(external_metrics.get("weighted_f1", 0.0)),
        "external_metrics": external_metrics,
    }

    with open(os.path.join(task_out_dir, "external_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[External ISRUC] acc={result['external_acc']:.4f}, "
        f"kappa={result['external_kappa']:.4f}, "
        f"macro_f1={result['external_macro_f1']:.4f}, "
        f"weighted_f1={result['external_weighted_f1']:.4f}"
    )

    return result


def collect_stage_generation_candidates(
    train_records,
    target_stages=(0, 1, 2, 3, 4),
    max_per_stage=None,
    seed=42,
):
    """Select reproducible, subject-balanced source epochs within each stage."""
    by_stage_subject = {
        int(stage): defaultdict(list) for stage in target_stages
    }
    for rec in train_records:
        sid = str(rec["subject_id"])
        labels = rec["true_sleep_stages"]
        eeg_epochs = rec["eeg_epochs"]
        n = min(len(eeg_epochs), len(labels))
        for epoch_idx in range(n):
            stage = int(labels[epoch_idx])
            if stage in by_stage_subject:
                by_stage_subject[stage][sid].append(
                    {
                        "subject_id": sid,
                        "epoch_index": int(epoch_idx),
                        "stage": stage,
                    }
                )

    rng = np.random.default_rng(int(seed))
    selected = []
    for stage in target_stages:
        stage = int(stage)
        subject_buckets = by_stage_subject[stage]
        subject_order = sorted(subject_buckets)
        for sid in subject_order:
            rng.shuffle(subject_buckets[sid])

        total_available = sum(len(values) for values in subject_buckets.values())
        target_count = (
            total_available
            if max_per_stage is None
            else min(max(int(max_per_stage), 0), total_available)
        )
        stage_selected = []
        cursor = {sid: 0 for sid in subject_order}
        while len(stage_selected) < target_count:
            active = [
                sid
                for sid in subject_order
                if cursor[sid] < len(subject_buckets[sid])
            ]
            if not active:
                break
            rng.shuffle(active)
            for sid in active:
                if len(stage_selected) >= target_count:
                    break
                stage_selected.append(subject_buckets[sid][cursor[sid]])
                cursor[sid] += 1
        selected.extend(stage_selected)
    return selected


def extract_centered_generation_window(rec, epoch_index, seq_len):
    eeg_epochs, eog_available = build_signal_epochs(rec)
    n = len(eeg_epochs)

    half = seq_len // 2
    start = max(0, epoch_index - half)
    end = min(start + seq_len, n)
    start = max(0, end - seq_len)

    center_idx = epoch_index - start

    window_eeg = pad_eeg_epochs(eeg_epochs[start:end], seq_len)
    valid_mask = np.asarray(pad_valid_mask(end - start, seq_len), dtype=np.int64)
    window_eeg = normalize_eeg_window(window_eeg, valid_mask)

    return window_eeg, valid_mask, center_idx, eog_available


def build_generated_stage_records(generated_items):
    synthetic_records = []
    for k, item in enumerate(generated_items):
        generated_signal = np.asarray(item["generated_eeg"], dtype=np.float32)
        synthetic_records.append({
            "subject_id": f"gen_{item['subject_id']}_{item['epoch_index']}_{k}",
            "descriptions": ["generated stage epoch"],
            "true_sleep_stages": [int(item["stage"])],
            "sleep_disorder_label": -1,
            "slow_wave_labels": [-100],
            "eeg_epochs": generated_signal[0].reshape(1, -1),
            "eog_loc_epochs": generated_signal[1].reshape(1, -1),
            "eog_roc_epochs": generated_signal[2].reshape(1, -1),
            "eog_available": np.asarray(item.get("eog_available", [1.0, 1.0]), dtype=np.float32),
        })
    return synthetic_records


def _fix_generated_epoch_length(x, target_len):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if len(x) == target_len:
        return x

    if len(x) > target_len:
        return x[:target_len]

    out = np.zeros(target_len, dtype=np.float32)
    out[:len(x)] = x
    return out


def build_generated_subject_copy_records(train_records, generated_items):
    """
    SleepGPT-style 加入方式：
    1) 原始 train_records 保留；
    2) 对每个原始 subject 复制一份 synthetic subject；
    3) 在 synthetic subject 中，把被生成的 epoch 替换为 generated_eeg；
    4) synthetic subject 仍然保持完整连续 epoch 序列。
    """
    gen_map = defaultdict(dict)

    for item in generated_items:
        sid = str(item["subject_id"])
        epoch_idx = int(item["epoch_index"])
        gen_map[sid][epoch_idx] = np.asarray(item["generated_eeg"], dtype=np.float32)

    synthetic_records = []

    for rec in train_records:
        sid = str(rec["subject_id"])
        if sid not in gen_map:
            continue

        eeg_epochs = np.asarray(rec["eeg_epochs"], dtype=np.float32).copy()
        eog_loc_epochs = np.asarray(rec.get("eog_loc_epochs", np.zeros_like(eeg_epochs)), dtype=np.float32).copy()
        eog_roc_epochs = np.asarray(rec.get("eog_roc_epochs", np.zeros_like(eeg_epochs)), dtype=np.float32).copy()
        labels = list(rec["true_sleep_stages"])

        n = min(len(eeg_epochs), len(labels))
        if n <= 0:
            continue

        eeg_epochs = eeg_epochs[:n].copy()
        labels = labels[:n]

        sample_len = eeg_epochs.shape[1]
        replaced = 0

        for epoch_idx, gen_epoch in gen_map[sid].items():
            if epoch_idx < 0 or epoch_idx >= n:
                continue

            if int(labels[epoch_idx]) not in [0, 1, 2, 3, 4]:
                continue

            if gen_epoch.ndim != 2 or gen_epoch.shape[0] != 3:
                continue
            eeg_epochs[epoch_idx] = _fix_generated_epoch_length(gen_epoch[0], sample_len)
            eog_loc_epochs[epoch_idx] = _fix_generated_epoch_length(gen_epoch[1], sample_len)
            eog_roc_epochs[epoch_idx] = _fix_generated_epoch_length(gen_epoch[2], sample_len)
            replaced += 1

        if replaced == 0:
            continue

        descriptions = rec.get("descriptions", [])
        if isinstance(descriptions, list):
            descriptions = descriptions[:n]
        else:
            descriptions = []

        slow_wave_labels = rec.get("slow_wave_labels", [-100] * n)
        if isinstance(slow_wave_labels, list):
            slow_wave_labels = slow_wave_labels[:n]
        else:
            slow_wave_labels = list(slow_wave_labels)[:n]

        synthetic_records.append({
            "subject_id": f"{sid}_synthetic_gen",
            "descriptions": descriptions,
            "true_sleep_stages": labels,
            "sleep_disorder_label": rec.get("sleep_disorder_label", -1),
            "slow_wave_labels": slow_wave_labels,
            "eeg_epochs": eeg_epochs,
            "eog_loc_epochs": eog_loc_epochs,
            "eog_roc_epochs": eog_roc_epochs,
            "eog_available": np.asarray(rec.get("eog_available", [0.0, 0.0]), dtype=np.float32),
            "synthetic_epoch_count": replaced,
        })

    return synthetic_records






def _stage_checkpoint_payload(best_state, split_payload, cfg, generation=None):
    payload = {
        "model": best_state["model"],
        "selection_policy": "fixed_final_epoch",
        "test_used_for_model_selection": False,
        "final_epoch": int(best_state["final_epoch"]),
        "fold_split": split_payload,
        "config_snapshot": dict(cfg.__dict__),
        "protocol_signature": stage_protocol_signature(cfg),
        "best_epoch": int(best_state["best_epoch"]),
        "primary_method": str(best_state.get("primary_method", "viterbi")),
        "selection_metric": str(best_state["selection_metric"]),
        "selection_tiebreak_metric": str(
            best_state["selection_tiebreak_metric"]
        ),
        "inference_profile": best_state.get("inference_profile"),
        "visualization_metadata": best_state.get("best_visualizations"),
        "saved_metrics": {
            "primary": best_state.get("metrics"),
        },
    }
    if generation is not None:
        payload["generation"] = generation
    return payload


def _saved_stage_metrics_payload(best_state):
    primary = best_state["metrics"]
    return {
        "primary_method": best_state.get("primary_method", "viterbi"),
        "primary_metrics": primary,
        "inference_profile": best_state.get("inference_profile"),
    }






def _stage_generation_augmentation_loso_legacy(records, cfg, device):
    """Independent LOSO sleep-stage run using real plus generated training data."""
    # Keep the ordinary sleep-stage baseline untouched. Only this augmentation
    # task recomputes fixed-STFT features from the final waveform supplied to
    # the classifier.
    cfg = copy.deepcopy(cfg)
    cfg.use_stage_relative_spectral_branch = bool(
        getattr(cfg, "generation_stage_use_fixed_stft_branch", True)
    )
    if cfg.use_stage_relative_spectral_branch:
        cfg.stage_spectral_mode = "fixed_stft_log"
    print(f"{'=' * 80}")
    print("Task: sleep_stage_generation_augmentation")
    if not records:
        print("sleep_stage_generation_augmentation: no valid records, skip.")
        return None

    source_path = os.path.abspath(cfg.generation_source_script)
    model_path = os.path.abspath(cfg.generation_full_model_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Generation source script not found: {source_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "generation_model.pt is missing. Run 03generation_decoder.py once: "
            f"{model_path}"
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "patch_generation_module", source_path
    )
    gen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_mod)
    generator, gen_cfg, generator_checkpoint = gen_mod.load_generation_model(
        model_path,
        device,
        expected_backbone_checkpoint=cfg.eeg_pretrained_model_path,
    )
    generator_provenance = {
        "path": model_path,
        "format_version": generator_checkpoint.get("format_version"),
        "model_type": generator_checkpoint.get("model_type"),
        "source_backbone": generator_checkpoint.get("source_backbone"),
        "best_epoch": generator_checkpoint.get("best_epoch"),
        "best_valid_loss": generator_checkpoint.get("best_valid_loss"),
        "fixed_during_loso": True,
        "sleep_stage_labels_used_to_train_generator": False,
        "input_domains": "time+frequency",
        "reconstruction_heads": ["time_linear", "frequency_linear"],
        "stft": {
            "window_samples": int(gen_cfg.stft_window_samples),
            "hop_samples": int(gen_cfg.stft_hop_samples),
            "fft_size": int(gen_cfg.stft_fft_size),
            "window_type": str(gen_cfg.stft_window_type),
            "frequency_min_hz": float(gen_cfg.frequency_min_hz),
            "frequency_max_hz": float(gen_cfg.frequency_max_hz),
        },
    }

    task_name = "sleep_stage_generation_augmentation"
    task_out_dir = task_output_dir(cfg, task_name)
    ensure_dir(task_out_dir)
    subject_ids = np.asarray(
        [str(record["subject_id"]) for record in records], dtype=object
    )
    dummy_x = np.arange(len(records))
    dummy_y = np.zeros(len(records), dtype=np.int64)
    logo = LeaveOneGroupOut()
    fold_splits = list(logo.split(dummy_x, dummy_y, groups=subject_ids))
    total_folds = len(fold_splits)
    fixed_epochs = max(int(getattr(cfg, "stage_loso_epochs", 20)), 1)
    class_names = cfg.tasks["sleep_stage_classification"]["class_names"]
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits, 1):
        max_folds = getattr(cfg, "stage_loso_max_folds", None)
        if max_folds is not None and fold_idx > max(1, int(max_folds)):
            break
        train_records = [records[index] for index in train_idx]
        test_records = [records[index] for index in test_idx]
        train_subjects = sorted(
            {str(record["subject_id"]) for record in train_records}
        )
        test_subjects = sorted(
            {str(record["subject_id"]) for record in test_records}
        )
        if set(train_subjects) & set(test_subjects):
            raise RuntimeError(f"Fold {fold_idx} has train/test subject overlap.")

        fold_name = (
            f"subject_{test_subjects[0]}"
            if len(test_subjects) == 1
            else f"fold_{fold_idx}"
        )
        fold_dir = os.path.join(task_out_dir, fold_name)
        generation_dir = os.path.join(fold_dir, "generation_task")
        test_dir = os.path.join(fold_dir, "outer_test")
        ensure_dir(fold_dir)
        ensure_dir(generation_dir)
        ensure_dir(test_dir)

        print(f"\n{'=' * 80}")
        print(
            f"Task: {task_name} | LOSO Fold {fold_idx}/{total_folds}"
        )
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print("Val subjects: [] (disabled)")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

        candidate_seed = int(cfg.generation_mask_seed + fold_idx * 1000000)
        candidates = collect_stage_generation_candidates(
            train_records,
            target_stages=cfg.generation_target_stages,
            max_per_stage=cfg.generation_samples_per_stage,
            seed=candidate_seed,
        )
        train_subject_set = set(train_subjects)
        test_subject_set = set(test_subjects)
        candidate_subject_set = {
            str(candidate["subject_id"]) for candidate in candidates
        }
        if not candidate_subject_set.issubset(train_subject_set):
            raise RuntimeError(
                f"Fold {fold_idx} generation candidates contain a non-training "
                "subject."
            )
        if candidate_subject_set & test_subject_set:
            raise RuntimeError(
                f"Fold {fold_idx} generation candidates contain a test subject."
            )
        record_map = {
            str(record["subject_id"]): record for record in train_records
        }
        generated_items = []
        for candidate_index, candidate in enumerate(
            tqdm(
                candidates,
                desc=f"Fold {fold_idx} patch generation",
                leave=False,
            )
        ):
            record = record_map[candidate["subject_id"]]
            window_eeg, valid_mask, center_idx, eog_available = (
                extract_centered_generation_window(
                    record,
                    candidate["epoch_index"],
                    cfg.generation_context_seq_len,
                )
            )
            generation_seed = candidate_seed + candidate_index
            generated_domains = gen_mod.generate_center_epoch(
                generator,
                window_eeg,
                valid_mask,
                center_idx,
                gen_cfg,
                device,
                eog_available=eog_available,
                seed=generation_seed,
            )
            if not isinstance(generated_domains, dict):
                raise RuntimeError(
                    "Generation checkpoint returned the obsolete time-only "
                    "interface. Retrain 03generation_decoder.py."
                )
            generated_epoch = np.asarray(
                generated_domains["waveform"],
                dtype=np.float32,
            )
            generated_spectrum = np.asarray(
                generated_domains["spectrum"],
                dtype=np.float16,
            )
            generated_mask = np.asarray(
                generated_domains["mask"],
                dtype=np.bool_,
            )
            generated_items.append(
                {
                    "subject_id": candidate["subject_id"],
                    "epoch_index": int(candidate["epoch_index"]),
                    "stage": int(candidate["stage"]),
                    "generated_eeg": generated_epoch,
                    "generated_spectrum": generated_spectrum,
                    "generation_mask": generated_mask,
                    "spectrum_consistency_l1": float(
                        generated_domains["spectrum_consistency_l1"]
                    ),
                    "eog_available": eog_available,
                    "generation_seed": int(generation_seed),
                }
            )

        synthetic_records = build_generated_subject_copy_records(
            train_records, generated_items
        )
        real_train_samples = build_stage_samples(
            train_records, seq_len=cfg.seq_len, stride=cfg.stride
        )
        synthetic_samples = build_stage_samples(
            synthetic_records, seq_len=cfg.seq_len, stride=cfg.stride
        )
        train_samples = real_train_samples + synthetic_samples
        test_samples = build_stage_samples(
            test_records, seq_len=cfg.seq_len, stride=cfg.stride
        )
        if not real_train_samples or not synthetic_samples or not test_samples:
            raise RuntimeError(
                f"Fold {fold_idx} produced an empty real/synthetic/test set."
            )

        # Generated samples affect classifier optimization only. Temporal
        # post-processing and optional class weights use real training data.
        log_transition = None
        if getattr(cfg, "use_viterbi_postprocess", False):
            log_transition = build_stage_transition_log_probs(
                real_train_samples,
                cfg.tasks["sleep_stage_classification"]["num_classes"],
                smoothing=cfg.viterbi_transition_smoothing,
            )

        split_payload = {
            "fold_idx": int(fold_idx),
            "fold_name": fold_name,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "val_subjects": [],
            "seed": int(cfg.seed),
            "fold_training_seed": int(cfg.seed + 10000 + fold_idx),
        }
        generation_manifest = {
            "generator": generator_provenance,
            "fold_idx": int(fold_idx),
            "candidate_seed": candidate_seed,
            "target_stages": [int(x) for x in cfg.generation_target_stages],
            "max_per_stage": cfg.generation_samples_per_stage,
            "train_subjects_only": True,
            "test_subjects_seen": False,
            "num_generated_epochs": len(generated_items),
            "num_synthetic_subjects": len(synthetic_records),
            "num_real_train_windows": len(real_train_samples),
            "num_synthetic_train_windows": len(synthetic_samples),
            "num_train_windows": len(train_samples),
            "generated_domains_saved": ["waveform", "spectrum", "mask"],
            "spectrum_consistency_l1_mean": (
                float(np.mean([
                    item["spectrum_consistency_l1"]
                    for item in generated_items
                ]))
                if generated_items
                else None
            ),
            "spectrum_consistency_l1_std": (
                float(np.std([
                    item["spectrum_consistency_l1"]
                    for item in generated_items
                ]))
                if generated_items
                else None
            ),
            "classifier_spectrum_source": (
                "fixed_stft_recomputed_from_final_waveform"
                if cfg.use_stage_relative_spectral_branch
                else "disabled"
            ),
            "independent_generated_spectrum_used_by_classifier": False,
            "viterbi_transition_source": "real_train_only",
            "class_weight_source": (
                "combined_real_synthetic"
                if cfg.use_stage_class_weights
                else "disabled"
            ),
        }
        with open(
            os.path.join(fold_dir, "split.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(split_payload, f, ensure_ascii=False, indent=2)
        with open(
            os.path.join(generation_dir, "generation_manifest.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(generation_manifest, f, ensure_ascii=False, indent=2)

        if cfg.generation_save_synthetic_npz:
            np.savez_compressed(
                os.path.join(generation_dir, "generated_stage_epochs.npz"),
                signals=np.asarray(
                    [item["generated_eeg"] for item in generated_items],
                    dtype=np.float32,
                ),
                spectra=np.asarray(
                    [
                        item["generated_spectrum"]
                        for item in generated_items
                    ],
                    dtype=np.float16,
                ),
                masks=np.asarray(
                    [item["generation_mask"] for item in generated_items],
                    dtype=np.bool_,
                ),
                spectrum_consistency_l1=np.asarray(
                    [
                        item["spectrum_consistency_l1"]
                        for item in generated_items
                    ],
                    dtype=np.float32,
                ),
                stages=np.asarray(
                    [item["stage"] for item in generated_items], dtype=np.int64
                ),
                subject_ids=np.asarray(
                    [str(item["subject_id"]) for item in generated_items]
                ),
                epoch_indices=np.asarray(
                    [item["epoch_index"] for item in generated_items],
                    dtype=np.int64,
                ),
                generation_seeds=np.asarray(
                    [item["generation_seed"] for item in generated_items],
                    dtype=np.int64,
                ),
            )

        print(
            f"Real windows={len(real_train_samples)} | "
            f"synthetic windows={len(synthetic_samples)} | "
            f"combined windows={len(train_samples)} | "
            f"generated epochs={len(generated_items)} | "
            "generated spectrum consistency L1="
            f"{generation_manifest['spectrum_consistency_l1_mean']:.6f}"
        )
        best_state = train_stage_fixed_final(
            train_samples,
            test_samples,
            cfg,
            device,
            fixed_epochs,
            fold_idx,
            test_dir,
            log_transition=log_transition,
        )

        checkpoint_payload = _stage_checkpoint_payload(
            best_state, split_payload, cfg, generation=generation_manifest
        )
        model_save_path = os.path.join(fold_dir, "best_model.pth")
        torch.save(checkpoint_payload, model_save_path)
        saved_metrics = _saved_stage_metrics_payload(best_state)
        with open(
            os.path.join(fold_dir, "test_metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(saved_metrics, f, ensure_ascii=False, indent=2)

        test_metrics = best_state["metrics"]
        fold_result = {
            "task_name": task_name,
            **experiment_metadata(cfg),
            "fold_idx": int(fold_idx),
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "val_subjects": [],
            "test_metrics": test_metrics,
            "primary_test_method": best_state.get("primary_method"),
            "best_epoch": int(best_state["best_epoch"]),
            "training_epochs": len(best_state["history"]),
            "generation": generation_manifest,
        }
        with open(
            os.path.join(fold_dir, "result.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(fold_result, f, ensure_ascii=False, indent=2)
        fold_results.append(fold_result)
        print(
            f"[Test] Fold {fold_idx} | "
            f"acc={test_metrics['acc']:.4f} | "
            f"kappa={test_metrics['kappa']:.4f} | "
            f"macro_f1={test_metrics['macro_f1']:.4f} | "
            f"weighted_f1={test_metrics.get('weighted_f1', 0.0):.4f}"
        )

        del (
            generated_items,
            synthetic_records,
            real_train_samples,
            synthetic_samples,
            train_samples,
            test_samples,
            best_state,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "task_name": task_name,
        **experiment_metadata(cfg),
        "generator": generator_provenance,
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }
    for metric_name in ["acc", "kappa", "macro_f1", "weighted_f1"]:
        values = [
            result["test_metrics"].get(metric_name, 0.0)
            for result in fold_results
        ]
        summary[f"mean_{metric_name}"] = (
            float(np.mean(values)) if values else 0.0
        )
        summary[f"std_{metric_name}"] = (
            float(np.std(values)) if values else 0.0
        )
    summary["mean_per_class_f1"] = {
        class_name: float(
            np.mean(
                [
                    result["test_metrics"]
                    .get("per_class_f1", {})
                    .get(class_name, 0.0)
                    for result in fold_results
                ]
            )
        )
        if fold_results
        else 0.0
        for class_name in class_names
    }
    with open(
        os.path.join(task_out_dir, "summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"\n{task_name} completed | folds={len(fold_results)} | "
        f"mean_acc={summary['mean_acc']:.4f} | "
        f"mean_kappa={summary['mean_kappa']:.4f} | "
        f"mean_macro_f1={summary['mean_macro_f1']:.4f}"
    )
    return summary


def _small_sample_atomic_json(payload, path):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _small_sample_atomic_torch(payload, path):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    temporary = f"{path}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _small_sample_file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _small_sample_state_sha256(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class _DirectGenerationWindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        item = self.windows[index]
        return {
            "eeg": torch.tensor(item["eeg"], dtype=torch.float32),
            "eog_available": torch.tensor(
                item["eog_available"], dtype=torch.float32
            ),
            "valid_mask": torch.tensor(item["valid_mask"], dtype=torch.long),
        }


def _direct_generation_windows(records, cfg):
    """Build unlabeled windows; sleep-stage labels are never inspected."""
    seq_len = int(cfg.generation_context_seq_len)
    stride = int(cfg.stride)
    validation_fraction = min(
        max(float(cfg.generation_direct_validation_fraction), 0.01), 0.40
    )
    train_windows = []
    valid_windows = []
    split_manifest = {}
    for record in records:
        subject_id = str(record["subject_id"])
        signal_epochs, eog_available = build_signal_epochs(record)
        n_epochs = int(len(signal_epochs))
        if n_epochs < seq_len * 2:
            raise RuntimeError(
                f"Direct generator needs at least {seq_len * 2} epochs for "
                f"a leakage-free temporal split: {subject_id} has {n_epochs}."
            )
        boundary = int(np.floor(n_epochs * (1.0 - validation_fraction)))
        boundary = min(max(boundary, seq_len), n_epochs - seq_len)
        subject_train = 0
        subject_valid = 0
        for start in build_window_starts(n_epochs, seq_len, stride):
            end = min(start + seq_len, n_epochs)
            # Drop windows crossing the boundary so fit/validation samples do
            # not share any epoch, even though ordinary windows overlap.
            if end <= boundary:
                destination = train_windows
                subject_train += 1
            elif start >= boundary:
                destination = valid_windows
                subject_valid += 1
            else:
                continue
            valid_mask = pad_valid_mask(end - start, seq_len)
            window = pad_eeg_epochs(signal_epochs[start:end], seq_len)
            window = normalize_eeg_window(window, valid_mask)
            destination.append({
                "subject_id": subject_id,
                "start_idx": int(start),
                "eeg": window,
                "eog_available": eog_available,
                "valid_mask": valid_mask,
            })
        if subject_train == 0 or subject_valid == 0:
            raise RuntimeError(
                f"Could not construct temporal generator split for {subject_id}."
            )
        split_manifest[subject_id] = {
            "epoch_boundary": int(boundary),
            "train_windows": int(subject_train),
            "validation_windows": int(subject_valid),
        }
    if not train_windows or not valid_windows:
        raise RuntimeError("Direct generator fit/validation windows are empty.")
    return train_windows, valid_windows, split_manifest


def _direct_generator_model_path(output_dir):
    return os.path.join(
        output_dir,
        "generator",
        "fixed_12_subject_generator.pt",
    )


def _direct_build_generator(cfg, device):
    settings = DirectGenerationSettings(cfg)
    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=settings.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size,
    )
    checkpoint_path = os.path.abspath(cfg.generation_direct_backbone_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "backbone_state_dict" not in checkpoint:
        raise RuntimeError(
            "Direct generator requires an event checkpoint containing "
            f"backbone_state_dict: {checkpoint_path}"
        )
    missing, unexpected = backbone.load_state_dict(
        checkpoint["backbone_state_dict"], strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Direct generator Backbone mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model = HierarchicalEventContextGenerator(backbone, settings).to(device)
    return model, settings, checkpoint


def _direct_reconstruction_losses(
    predictions,
    target,
    patch_mask,
    valid_mask,
    eog_available,
    target_indices,
    settings,
):
    batch_indices = torch.arange(target.size(0), device=target.device)
    target_indices = torch.as_tensor(
        target_indices, dtype=torch.long, device=target.device
    ).reshape(-1)
    target_epoch = target[batch_indices, target_indices]
    target_patches = _direct_patchify(
        target_epoch, settings.patch_size_samples
    )
    valid_patches = _direct_valid_patch_mask(
        valid_mask,
        eog_available,
        target.size(-1),
        settings.patch_size_samples,
    )
    active_mask = (patch_mask & valid_patches)[
        batch_indices, target_indices
    ]
    predicted_time = predictions["time_patches"]
    predicted_frequency = predictions["frequency_patches"]
    if not active_mask.any():
        zero = predicted_time.sum() * 0.0
        return {"total": zero, "wave": zero, "frequency": zero}
    wave_loss = F.smooth_l1_loss(
        predicted_time[active_mask],
        target_patches[active_mask],
        beta=float(settings.smooth_l1_beta),
    )
    target_frequency = _direct_aligned_log_spectral_patches(
        target_epoch.unsqueeze(1), settings
    )[:, 0]
    frequency_head_loss = F.l1_loss(
        predicted_frequency[active_mask], target_frequency[active_mask]
    )
    final_patches = torch.where(
        active_mask.unsqueeze(-1), predicted_time, target_patches
    )
    final_waveform = _direct_unpatchify(
        final_patches, target_epoch.size(-1)
    )
    final_frequency = _direct_aligned_log_spectral_patches(
        final_waveform.unsqueeze(1), settings
    )[:, 0]
    waveform_frequency_loss = F.l1_loss(
        final_frequency[active_mask], target_frequency[active_mask]
    )
    head_fraction = float(settings.frequency_head_fraction)
    waveform_fraction = float(settings.frequency_waveform_fraction)
    denominator = head_fraction + waveform_fraction
    if denominator <= 0.0:
        raise ValueError("Direct generator frequency fractions must be positive.")
    frequency_loss = (
        head_fraction * frequency_head_loss
        + waveform_fraction * waveform_frequency_loss
    ) / denominator
    total = wave_loss + float(settings.frequency_loss_weight) * frequency_loss
    return {"total": total, "wave": wave_loss, "frequency": frequency_loss}


def _direct_run_generation_epoch(
    model, loader, optimizer, scaler, device, settings, training, mask_seed
):
    model.train(training)
    totals = {"total": 0.0, "wave": 0.0, "frequency": 0.0}
    examples = 0
    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(int(mask_seed))
    iterator = tqdm(
        loader,
        desc="direct generator train" if training else "direct generator valid",
        leave=False,
    )
    for batch in iterator:
        target = batch["eeg"].to(device)
        eog_available = batch["eog_available"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        target_indices = _direct_sample_target_indices(
            valid_mask, generator=mask_generator
        )
        patch_mask = _direct_patch_mask(
            valid_mask,
            eog_available,
            target.size(-1),
            settings.patch_size_samples,
            settings.mask_ratio,
            generator=mask_generator,
            target_indices=target_indices,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda")
            ):
                predictions = model(
                    target,
                    patch_mask,
                    eog_available,
                    valid_mask,
                    target_indices,
                )
                losses = _direct_reconstruction_losses(
                    {name: value.float() for name, value in predictions.items()},
                    target.float(),
                    patch_mask,
                    valid_mask,
                    eog_available,
                    target_indices,
                    settings,
                )
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        batch_size = int(target.size(0))
        for name in totals:
            totals[name] += float(losses[name].detach().item()) * batch_size
        examples += batch_size
    return {name: value / max(examples, 1) for name, value in totals.items()}


def _direct_generator_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _direct_load_saved_generator(
    model_path, fixed_training_subjects, cfg, device
):
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 10:
        raise RuntimeError(f"Unsupported direct generator: {model_path}")
    if payload.get("model_type") != "04_hierarchical_event_context_generator":
        raise RuntimeError(f"Unexpected generator architecture: {model_path}")
    expected_subjects = sorted(str(x) for x in fixed_training_subjects)
    if sorted(payload.get("training_subjects", [])) != expected_subjects:
        raise RuntimeError(
            f"Direct generator subject scope mismatch: {model_path}"
        )
    backbone_path = os.path.abspath(cfg.generation_direct_backbone_path)
    source = payload.get("source_backbone", {})
    if os.path.normcase(os.path.abspath(source.get("path", ""))) != (
        os.path.normcase(backbone_path)
    ):
        raise RuntimeError(f"Direct generator Backbone path mismatch: {model_path}")
    if source.get("sha256") != _small_sample_file_sha256(backbone_path):
        raise RuntimeError(f"Direct generator Backbone hash mismatch: {model_path}")
    model, settings, _ = _direct_build_generator(cfg, device)
    if payload.get("settings") != settings.as_dict():
        raise RuntimeError(
            "Fixed generator settings differ from the current hierarchical "
            f"protocol: {model_path}"
        )
    missing, unexpected = model.load_state_dict(
        payload["generation_state_dict"], strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Direct generator state mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.eval()
    return model, settings, payload


def _direct_train_or_load_generator(
    fixed_training_records,
    fixed_training_subjects,
    cfg,
    device,
    output_dir,
):
    model_path = _direct_generator_model_path(output_dir)
    if os.path.isfile(model_path):
        model, settings, payload = _direct_load_saved_generator(
            model_path, fixed_training_subjects, cfg, device
        )
        print(
            "[Fixed Generator Reuse] fixed_train_pool=12 | "
            f"best_epoch={payload['best_epoch']} | path={model_path}"
        )
        return model, settings, payload, model_path

    seed = int(cfg.generation_mask_seed)
    set_seed(seed)
    train_windows, valid_windows, temporal_split = (
        _direct_generation_windows(fixed_training_records, cfg)
    )
    model, settings, source_checkpoint = _direct_build_generator(cfg, device)
    transformer_parameters, cnn_parameters = model.adaptation_parameter_groups()
    excluded_ids = {id(parameter) for parameter in model.backbone.parameters()}
    new_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in excluded_ids and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": new_parameters,
                "lr": float(cfg.generation_direct_head_lr),
                "name": "new_generation_head",
            },
            {
                "params": transformer_parameters,
                "lr": 0.0,
                "name": "last_context_transformer",
            },
            {
                "params": cnn_parameters,
                "lr": 0.0,
                "name": "last_eeg_eog_convolutions",
            },
        ],
        weight_decay=float(cfg.generation_direct_weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    train_loader = DataLoader(
        _DirectGenerationWindowDataset(train_windows),
        batch_size=int(cfg.generation_direct_batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        _DirectGenerationWindowDataset(valid_windows),
        batch_size=int(cfg.generation_direct_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    backbone_path = os.path.abspath(cfg.generation_direct_backbone_path)
    source_meta = {
        "path": backbone_path,
        "sha256": _small_sample_file_sha256(backbone_path),
        "pretrain_type": source_checkpoint.get("pretrain_type"),
        "best_epoch": source_checkpoint.get("best_epoch"),
        "event_names": source_checkpoint.get("event_names"),
    }
    print(
        "[Fixed Generator Train] fixed_train_pool=12 | "
        f"subjects={sorted(fixed_training_subjects)} | "
        f"train_windows={len(train_windows)} | "
        f"valid_windows={len(valid_windows)} | mask_ratio={settings.mask_ratio:.2f} | "
        "target=random_one_epoch | other_19_epochs=visible_context | "
        "full_30s_cnn=True | local_tokens=3x30 | decoder_tokens=90 | "
        "heads=time_linear+frequency_linear | "
        "labels_used=False | downstream_val_test_used=False"
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    head_only_epochs = max(int(cfg.generation_direct_head_only_epochs), 0)
    max_epochs = max(int(cfg.generation_direct_epochs), 1)
    for epoch in range(1, max_epochs + 1):
        if epoch == head_only_epochs + 1:
            model.enable_backbone_adaptation()
            for group in optimizer.param_groups:
                if group.get("name") == "last_context_transformer":
                    group["lr"] = float(cfg.generation_direct_encoder_lr)
                elif group.get("name") == "last_eeg_eog_convolutions":
                    group["lr"] = float(cfg.generation_direct_cnn_lr)
            patience = 0
            print(
                "[Fixed Generator Unfreeze] final context Transformer layer "
                "+ final EEG/EOG Conv1d | "
                f"transformer_lr={float(cfg.generation_direct_encoder_lr):.2e} | "
                f"cnn_lr={float(cfg.generation_direct_cnn_lr):.2e} | "
                "all_BatchNorm=eval"
            )
        train_loss = _direct_run_generation_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            settings,
            True,
            seed + epoch,
        )
        valid_loss = _direct_run_generation_epoch(
            model,
            valid_loader,
            optimizer,
            scaler,
            device,
            settings,
            False,
            seed + 1000000,
        )
        history.append({
            "epoch": epoch,
            "phase": (
                "head_only"
                if epoch <= head_only_epochs
                else "restricted_backbone_adaptation"
            ),
            "train_total": float(train_loss["total"]),
            "train_wave": float(train_loss["wave"]),
            "train_frequency": float(train_loss["frequency"]),
            "valid_total": float(valid_loss["total"]),
            "valid_wave": float(valid_loss["wave"]),
            "valid_frequency": float(valid_loss["frequency"]),
        })
        print(
            "[Fixed Generator] fixed_train_pool=12 | "
            f"epoch={epoch}/{max_epochs} | train(total={train_loss['total']:.6f}, "
            f"wave={train_loss['wave']:.6f}, freq={train_loss['frequency']:.6f}) | "
            f"valid(total={valid_loss['total']:.6f}, wave={valid_loss['wave']:.6f}, "
            f"freq={valid_loss['frequency']:.6f})"
        )
        improved = valid_loss["total"] < (
            best_loss - float(cfg.generation_direct_min_delta)
        )
        if improved:
            best_loss = float(valid_loss["total"])
            best_epoch = epoch
            best_state = _direct_generator_state(model)
            patience = 0
        else:
            patience += 1
        interval = int(cfg.generation_direct_checkpoint_interval)
        if interval > 0 and epoch % interval == 0:
            periodic_path = os.path.join(
                os.path.dirname(model_path),
                f"generation_model_epoch_{epoch:04d}.pt",
            )
            _small_sample_atomic_torch({
                "format_version": 10,
                "model_type": "04_hierarchical_event_context_generator",
                "generation_state_dict": _direct_generator_state(model),
                "settings": settings.as_dict(),
                "source_backbone": source_meta,
                "training_subjects": sorted(
                    str(x) for x in fixed_training_subjects
                ),
                "generator_scope": "one_fixed_generator_for_all_conditions",
                "checkpoint_epoch": epoch,
                "best_epoch": best_epoch,
                "best_valid_loss": best_loss,
                "history": history,
            }, periodic_path)
            print(f"[Direct Generator Checkpoint] {periodic_path}")
        if (
            epoch > head_only_epochs
            and patience >= int(cfg.generation_direct_patience)
        ):
            print(
                f"[Fixed Generator Early Stop] epoch={epoch} | best={best_epoch}"
            )
            break
    if best_state is None:
        raise RuntimeError("Direct generator training produced no checkpoint.")
    payload = {
        "format_version": 10,
        "model_type": "04_hierarchical_event_context_generator",
        "generation_state_dict": best_state,
        "settings": settings.as_dict(),
        "source_backbone": source_meta,
        "training_subjects": sorted(
            str(x) for x in fixed_training_subjects
        ),
        "generator_scope": "one_fixed_generator_for_all_conditions_and_repeats",
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_loss),
        "history": history,
        "temporal_unlabeled_split": temporal_split,
        "sleep_stage_labels_used": False,
        "event_labels_used": False,
        "downstream_validation_subjects_used": False,
        "downstream_test_subjects_used": False,
        "masking": {
            "target_epochs_per_window": 1,
            "target_patch_ratio": float(settings.mask_ratio),
            "other_context_epochs_masked": False,
            "shared_time_mask_across_eeg_loc_roc": True,
        },
        "losses": [
            "target_epoch_masked_smooth_l1",
            "frequency_head_plus_final_waveform_stft_l1",
        ],
    }
    _small_sample_atomic_torch(payload, model_path)
    _small_sample_atomic_json(
        {key: value for key, value in payload.items() if key != "generation_state_dict"},
        os.path.join(os.path.dirname(model_path), "generation_training_summary.json"),
    )
    model.load_state_dict(best_state, strict=True)
    model.eval()
    print(
        "[Fixed Generator Saved] fixed_train_pool=12 | "
        f"best_epoch={best_epoch} | path={model_path}"
    )
    return model, settings, payload, model_path


@torch.no_grad()
def _direct_generate_center_epoch(
    model,
    eeg_window,
    valid_mask,
    center_idx,
    settings,
    device,
    eog_available=None,
    seed=42,
):
    eeg_window = torch.as_tensor(eeg_window, dtype=torch.float32)
    valid_mask = torch.as_tensor(valid_mask, dtype=torch.long)
    if eeg_window.ndim == 3:
        eeg_window = eeg_window.unsqueeze(0)
    if valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)
    if eog_available is None:
        eog_available = torch.ones((eeg_window.size(0), 2))
    eog_available = torch.as_tensor(eog_available, dtype=torch.float32)
    if eog_available.ndim == 1:
        eog_available = eog_available.unsqueeze(0)
    eeg_window = eeg_window.to(device)
    valid_mask = valid_mask.to(device)
    eog_available = eog_available.to(device)
    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(int(seed))
    target_indices = torch.full(
        (eeg_window.size(0),),
        int(center_idx),
        dtype=torch.long,
        device=device,
    )
    patch_mask = _direct_patch_mask(
        valid_mask,
        eog_available,
        eeg_window.size(-1),
        settings.patch_size_samples,
        settings.mask_ratio,
        generator=mask_generator,
        target_indices=target_indices,
    )
    predictions = model(
        eeg_window,
        patch_mask,
        eog_available,
        valid_mask,
        target_indices,
    )
    center = int(center_idx)
    target_patches = _direct_patchify(
        eeg_window[:, center], settings.patch_size_samples
    )
    center_mask = patch_mask[:, center]
    reconstructed_patches = torch.where(
        center_mask.unsqueeze(-1),
        predictions["time_patches"],
        target_patches,
    )
    synthetic = _direct_unpatchify(
        reconstructed_patches, eeg_window.size(-1)
    )
    target_frequency = _direct_aligned_log_spectral_patches(
        eeg_window[:, center:center + 1], settings
    )[:, 0]
    reconstructed_frequency = torch.where(
        center_mask.unsqueeze(-1),
        predictions["frequency_patches"],
        target_frequency,
    )
    center_waveform = synthetic.unsqueeze(1)
    waveform_frequency = _direct_aligned_log_spectral_patches(
        center_waveform, settings
    )[0, 0]
    center_mask = center_mask[0]
    consistency = F.l1_loss(
        reconstructed_frequency[0][center_mask],
        waveform_frequency[center_mask],
    ).item()
    return {
        "waveform": center_waveform[0, 0].cpu().numpy(),
        "spectrum": reconstructed_frequency[0].cpu().numpy(),
        "mask": center_mask.cpu().numpy(),
        "spectrum_consistency_l1": float(consistency),
    }


def _small_sample_subject_filename(subject_id):
    readable = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(subject_id)
    ).strip("_")
    suffix = hashlib.sha256(
        str(subject_id).encode("utf-8")
    ).hexdigest()[:8]
    return f"{readable or 'subject'}_{suffix}.npz"


def _small_sample_subject_histogram(record):
    histogram = np.zeros(5, dtype=np.int64)
    for value in record.get("true_sleep_stages", []):
        stage = int(value)
        if 0 <= stage < 5:
            histogram[stage] += 1
    return histogram


def _small_sample_set_statistics(subject_ids, record_map):
    histograms = [
        _small_sample_subject_histogram(record_map[str(subject_id)])
        for subject_id in subject_ids
    ]
    total_histogram = (
        np.sum(np.stack(histograms), axis=0)
        if histograms
        else np.zeros(5, dtype=np.int64)
    )
    total_epochs = int(total_histogram.sum())
    proportions = total_histogram.astype(np.float64) / max(total_epochs, 1)
    return {
        "subjects": [str(value) for value in subject_ids],
        "num_subjects": int(len(subject_ids)),
        "stage_histogram": total_histogram.tolist(),
        "stage_proportions": proportions.tolist(),
        "labeled_epochs": total_epochs,
        "mean_labeled_epochs_per_subject": float(
            total_epochs / max(len(subject_ids), 1)
        ),
    }


def _small_sample_split_score(train_ids, val_ids, test_ids, record_map):
    all_ids = list(train_ids) + list(val_ids) + list(test_ids)
    global_stats = _small_sample_set_statistics(all_ids, record_map)
    global_proportions = np.asarray(
        global_stats["stage_proportions"], dtype=np.float64
    )
    global_mean_epochs = float(
        global_stats["mean_labeled_epochs_per_subject"]
    )
    score = 0.0
    for subject_ids in (train_ids, val_ids, test_ids):
        stats = _small_sample_set_statistics(subject_ids, record_map)
        proportions = np.asarray(
            stats["stage_proportions"], dtype=np.float64
        )
        score += float(np.abs(proportions - global_proportions).sum())
        score += 0.10 * abs(
            float(stats["mean_labeled_epochs_per_subject"])
            - global_mean_epochs
        ) / max(global_mean_epochs, 1.0)
        score += 10.0 * float(np.count_nonzero(
            np.asarray(stats["stage_histogram"]) == 0
        ))
    return score


def _small_sample_load_or_create_split(records, cfg, output_dir):
    record_map = {str(record["subject_id"]): record for record in records}
    all_subjects = sorted(record_map)
    n_train = int(cfg.generation_small_sample_train_pool_subjects)
    n_val = int(cfg.generation_small_sample_validation_subjects)
    n_test = int(cfg.generation_small_sample_test_subjects)
    path = os.path.join(
        output_dir, f"fixed_split_{n_train}_{n_val}_{n_test}.json"
    )
    required_subjects = n_train + n_val + n_test
    dataset_name = str(getattr(
        cfg, "generation_small_sample_dataset_name", "dataset"
    ))
    if len(all_subjects) < required_subjects:
        raise RuntimeError(
            f"{dataset_name} small-sample protocol requires at least "
            f"{required_subjects} subjects, found "
            f"{len(all_subjects)}: {all_subjects}"
        )
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        loaded = (
            list(payload["train_pool_subjects"])
            + list(payload["validation_subjects"])
            + list(payload["test_subjects"])
        )
        if (
            len(loaded) != required_subjects
            or len(set(loaded)) != len(loaded)
            or not set(loaded).issubset(set(all_subjects))
        ):
            raise RuntimeError(
                f"The locked {n_train}/{n_val}/{n_test} split does not "
                "match current "
                f"{dataset_name} data."
            )
        return payload

    rng = np.random.default_rng(
        int(cfg.generation_small_sample_split_seed)
    )
    trials = max(
        int(cfg.generation_small_sample_split_search_trials), 1
    )
    subject_array = np.asarray(all_subjects, dtype=object)
    best_score = float("inf")
    best = None
    for _ in range(trials):
        permutation = rng.permutation(subject_array).tolist()
        train_ids = permutation[:n_train]
        val_ids = permutation[n_train:n_train + n_val]
        test_ids = permutation[
            n_train + n_val:n_train + n_val + n_test
        ]
        score = _small_sample_split_score(
            train_ids, val_ids, test_ids, record_map
        )
        if score < best_score:
            best_score = score
            best = (
                sorted(str(value) for value in train_ids),
                sorted(str(value) for value in val_ids),
                sorted(str(value) for value in test_ids),
            )
    train_ids, val_ids, test_ids = best
    selected_subjects = sorted(train_ids + val_ids + test_ids)
    excluded_subjects = sorted(
        set(all_subjects) - set(selected_subjects)
    )
    payload = {
        "protocol": (
            f"fixed_subject_{n_train}_train_pool_"
            f"{n_val}_validation_{n_test}_test"
        ),
        "dataset_name": dataset_name,
        "available_subjects": all_subjects,
        "selected_protocol_subjects": selected_subjects,
        "excluded_subjects": excluded_subjects,
        "split_seed": int(cfg.generation_small_sample_split_seed),
        "split_search_trials": trials,
        "balance_score": float(best_score),
        "train_pool_subjects": train_ids,
        "validation_subjects": val_ids,
        "test_subjects": test_ids,
        "statistics": {
            "all_available": _small_sample_set_statistics(
                all_subjects, record_map
            ),
            "selected_protocol_subjects": _small_sample_set_statistics(
                selected_subjects, record_map
            ),
            "train_pool": _small_sample_set_statistics(
                train_ids, record_map
            ),
            "validation": _small_sample_set_statistics(val_ids, record_map),
            "test": _small_sample_set_statistics(test_ids, record_map),
        },
        "locked_before_model_training": True,
        "model_results_used_to_choose_split": False,
    }
    _small_sample_atomic_json(payload, path)
    return payload


def _small_sample_load_or_create_repeats(split_payload, cfg, output_dir):
    path = os.path.join(output_dir, "repeat_subject_manifest.json")
    train_pool = sorted(
        str(value) for value in split_payload["train_pool_subjects"]
    )
    train_sizes = [
        int(value) for value in cfg.generation_small_sample_train_sizes
    ]
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if sorted(payload["train_pool_subjects"]) != train_pool:
            raise RuntimeError(
                "Repeat manifest does not match the locked train pool."
            )
        # Preserve every previously locked permutation and subset. When a new
        # train-size button is added (for example n=6), derive that nested
        # subset from the existing permutation instead of resampling the
        # experiment or changing the established 1/2/5/12 conditions.
        manifest_changed = False
        for repeat_payload in payload.get("repeats", []):
            permutation = [
                str(value)
                for value in repeat_payload.get("subject_permutation", [])
            ]
            if (
                len(permutation) != len(train_pool)
                or sorted(permutation) != train_pool
            ):
                raise RuntimeError(
                    "Repeat manifest has an invalid subject permutation."
                )
            subjects_by_size = repeat_payload.setdefault(
                "subjects_by_train_size", {}
            )
            for size in train_sizes:
                key = str(size)
                expected_subjects = permutation[:size]
                if key not in subjects_by_size:
                    subjects_by_size[key] = expected_subjects
                    manifest_changed = True
                elif [str(value) for value in subjects_by_size[key]] != (
                    expected_subjects
                ):
                    raise RuntimeError(
                        "Repeat manifest nested subset mismatch for "
                        f"train_size={size}."
                    )
        if [int(value) for value in payload.get("train_sizes", [])] != (
            train_sizes
        ):
            payload["train_sizes"] = train_sizes
            manifest_changed = True
        if manifest_changed:
            _small_sample_atomic_json(payload, path)
            print(
                "[Repeat Manifest Updated] added configured train sizes "
                f"{train_sizes} without changing locked permutations."
            )
        return payload

    repeats = []
    used_subsets = {
        size: set() for size in train_sizes if size < len(train_pool)
    }
    for repeat_index in range(
        1, int(cfg.generation_small_sample_repeats) + 1
    ):
        accepted = None
        for attempt in range(10000):
            seed = int(
                cfg.generation_small_sample_subset_seed
                + repeat_index * 1009
                + attempt
            )
            permutation = np.random.default_rng(seed).permutation(
                np.asarray(train_pool, dtype=object)
            ).tolist()
            keys = {
                size: tuple(sorted(permutation[:size]))
                for size in train_sizes if size < len(train_pool)
            }
            if all(
                key not in used_subsets[size]
                for size, key in keys.items()
            ):
                accepted = (seed, permutation, keys)
                break
        if accepted is None:
            raise RuntimeError("Could not construct five unique subsets.")
        seed, permutation, keys = accepted
        for size, key in keys.items():
            used_subsets[size].add(key)
        repeats.append({
            "repeat": repeat_index,
            "subset_seed": seed,
            "subject_permutation": [
                str(value) for value in permutation
            ],
            "subjects_by_train_size": {
                str(size): [
                    str(value) for value in permutation[:size]
                ]
                for size in train_sizes
            },
            "generation_seed": int(
                cfg.generation_mask_seed + repeat_index * 1000000
            ),
            "classifier_seed": int(
                cfg.generation_small_sample_classifier_seed
                + repeat_index * 1000000
            ),
        })
    payload = {
        "protocol": "five_repeated_nested_random_subsets",
        "train_pool_subjects": train_pool,
        "validation_subjects": split_payload["validation_subjects"],
        "test_subjects": split_payload["test_subjects"],
        "train_sizes": train_sizes,
        "repeats": repeats,
        "subsets_nested_within_repeat": True,
        "validation_and_test_fixed_across_all_runs": True,
    }
    _small_sample_atomic_json(payload, path)
    return payload


def _small_sample_cache_valid(
    path, subject_id, repeat_seed, generator_hash
):
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(np.asarray(data["subject_id"]).item())
                == str(subject_id)
                and int(np.asarray(
                    data["repeat_generation_seed"]
                ).item()) == int(repeat_seed)
                and str(np.asarray(data["generator_sha256"]).item())
                == str(generator_hash)
                and data["waveforms"].ndim == 3
                and data["waveforms"].shape[1] == 3
            )
    except Exception:
        return False


def _small_sample_prepare_repeat_cache(
    fixed_training_records,
    repeat_payload,
    generator,
    generator_settings,
    generator_checkpoint,
    generator_hash,
    cfg,
    device,
    cache_root,
):
    repeat_index = int(repeat_payload["repeat"])
    repeat_seed = int(repeat_payload["generation_seed"])
    repeat_dir = os.path.join(
        cache_root,
        f"repeat_{repeat_index:02d}",
    )
    ensure_dir(repeat_dir)
    generated_subjects = 0
    reused_subjects = 0
    generated_epochs = 0
    consistency_values = []
    for record in fixed_training_records:
        subject_id = str(record["subject_id"])
        cache_path = os.path.join(
            repeat_dir, _small_sample_subject_filename(subject_id)
        )
        if _small_sample_cache_valid(
            cache_path, subject_id, repeat_seed, generator_hash
        ):
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_epochs = int(len(cached["epoch_indices"]))
                cached_consistency = np.asarray(
                    cached["spectrum_consistency_l1"],
                    dtype=np.float32,
                )
            reused_subjects += 1
            generated_epochs += cached_epochs
            consistency_values.extend(cached_consistency.tolist())
            print(
                f"[Generation Cache Reuse] repeat={repeat_index} | "
                f"subject={subject_id} | epochs={cached_epochs} | "
                f"path={cache_path}"
            )
            continue
        subject_seed_offset = int(hashlib.sha256(
            subject_id.encode("utf-8")
        ).hexdigest()[:8], 16)
        candidate_seed = repeat_seed + subject_seed_offset
        candidates = collect_stage_generation_candidates(
            [record],
            target_stages=cfg.generation_target_stages,
            max_per_stage=cfg.generation_samples_per_stage,
            seed=candidate_seed,
        )
        if not candidates:
            raise RuntimeError(
                f"No generation candidates for {subject_id}."
            )
        epoch_indices = []
        stages = []
        waveforms = []
        spectra = []
        masks = []
        consistency = []
        for candidate_index, candidate in enumerate(tqdm(
            candidates,
            desc=(
                f"Generate repeat {repeat_index} subject {subject_id}"
            ),
            leave=True,
        )):
            window_eeg, valid_mask, center_idx, eog_available = (
                extract_centered_generation_window(
                    record,
                    candidate["epoch_index"],
                    cfg.generation_context_seq_len,
                )
            )
            generated = _direct_generate_center_epoch(
                generator,
                window_eeg,
                valid_mask,
                center_idx,
                generator_settings,
                device,
                eog_available=eog_available,
                seed=candidate_seed + candidate_index * 1009,
            )
            epoch_indices.append(int(candidate["epoch_index"]))
            stages.append(int(candidate["stage"]))
            waveforms.append(np.asarray(
                generated["waveform"], dtype=np.float32
            ))
            spectra.append(np.asarray(
                generated["spectrum"], dtype=np.float16
            ))
            masks.append(np.asarray(
                generated["mask"], dtype=np.bool_
            ))
            consistency.append(float(
                generated["spectrum_consistency_l1"]
            ))
        print(
            f"[Generation Cache Compressing] repeat={repeat_index} | "
            f"subject={subject_id} | epochs={len(epoch_indices)} | "
            "writing compressed cache..."
        )
        temporary = f"{cache_path}.tmp.npz"
        np.savez_compressed(
            temporary,
            subject_id=np.asarray(subject_id),
            repeat=np.asarray(repeat_index, dtype=np.int64),
            repeat_generation_seed=np.asarray(
                repeat_seed, dtype=np.int64
            ),
            candidate_seed=np.asarray(candidate_seed, dtype=np.int64),
            generator_sha256=np.asarray(generator_hash),
            generator_format_version=np.asarray(
                int(generator_checkpoint["format_version"]),
                dtype=np.int64,
            ),
            generator_model_type=np.asarray(
                str(generator_checkpoint["model_type"])
            ),
            epoch_indices=np.asarray(epoch_indices, dtype=np.int32),
            stages=np.asarray(stages, dtype=np.int8),
            waveforms=np.asarray(waveforms, dtype=np.float32),
            spectra=np.asarray(spectra, dtype=np.float16),
            masks=np.asarray(masks, dtype=np.bool_),
            spectrum_consistency_l1=np.asarray(
                consistency, dtype=np.float32
            ),
        )
        os.replace(temporary, cache_path)
        generated_subjects += 1
        generated_epochs += len(epoch_indices)
        consistency_values.extend(consistency)
        print(
            f"[Generation Cache Saved] repeat={repeat_index} | "
            f"subject={subject_id} | epochs={len(epoch_indices)} | "
            f"spectrum_consistency_l1={np.mean(consistency):.6f} | "
            f"path={cache_path}"
        )
    _small_sample_atomic_json({
        "repeat": repeat_index,
        "repeat_generation_seed": repeat_seed,
        "generator_sha256": generator_hash,
        "generator_format_version": int(
            generator_checkpoint["format_version"]
        ),
        "generator_model_type": str(
            generator_checkpoint["model_type"]
        ),
        "generation_training_subjects": [
            str(record["subject_id"])
            for record in fixed_training_records
        ],
        "generator_scope": "fixed_12_subject_unlabeled_adaptation",
        "cache_scope": "all_fixed_training_pool_subjects",
        "one_synthetic_epoch_per_real_epoch": True,
        "generator_frozen": True,
        "validation_generated": False,
        "test_generated": False,
        "selected_training_waveforms_used_to_train_generator": True,
        "sleep_stage_labels_used_to_train_generator": False,
    }, os.path.join(repeat_dir, "cache_manifest.json"))
    summary = {
        "repeat": repeat_index,
        "generated_subjects": int(generated_subjects),
        "reused_subjects": int(reused_subjects),
        "total_subjects": int(len(fixed_training_records)),
        "total_cached_epochs": int(generated_epochs),
        "spectrum_consistency_l1_mean": (
            float(np.mean(consistency_values))
            if consistency_values
            else None
        ),
        "cache_dir": repeat_dir,
    }
    _small_sample_atomic_json(
        summary,
        os.path.join(repeat_dir, "generation_summary.json"),
    )
    return summary


def _small_sample_load_generated_items(
    subject_ids,
    repeat_payload,
    generator_hash,
    cfg,
    cache_root,
):
    repeat_index = int(repeat_payload["repeat"])
    repeat_seed = int(repeat_payload["generation_seed"])
    repeat_dir = os.path.join(
        cache_root,
        f"repeat_{repeat_index:02d}",
    )
    generated_items = []
    consistency = []
    for subject_id in subject_ids:
        path = os.path.join(
            repeat_dir, _small_sample_subject_filename(subject_id)
        )
        if not _small_sample_cache_valid(
            path, subject_id, repeat_seed, generator_hash
        ):
            raise RuntimeError(
                f"Missing or stale generation cache: {path}"
            )
        with np.load(path, allow_pickle=False) as data:
            epoch_indices = np.asarray(data["epoch_indices"])
            stages = np.asarray(data["stages"])
            waveforms = np.asarray(data["waveforms"], dtype=np.float32)
            consistency.extend(np.asarray(
                data["spectrum_consistency_l1"], dtype=np.float32
            ).tolist())
            for index in range(len(epoch_indices)):
                generated_items.append({
                    "subject_id": str(subject_id),
                    "epoch_index": int(epoch_indices[index]),
                    "stage": int(stages[index]),
                    "generated_eeg": waveforms[index],
                })
    return generated_items, consistency


def _small_sample_build_stage_model(cfg, device):
    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size,
    )
    maybe_load_eeg_pretrained_weights(backbone, cfg)
    return MultiTaskSleepModel(backbone, cfg=cfg).to(device)


def _small_sample_selection_values(metrics, cfg):
    names = [
        str(cfg.generation_small_sample_selection_metric),
        *list(cfg.generation_small_sample_selection_tiebreakers),
    ]
    return tuple(float(metrics.get(name, 0.0)) for name in names)


def _small_sample_is_better(candidate, best, cfg):
    if best is None:
        return True
    candidate_values = _small_sample_selection_values(candidate, cfg)
    best_values = _small_sample_selection_values(best, cfg)
    min_delta = float(cfg.early_stopping_min_delta)
    if candidate_values[0] > best_values[0] + min_delta:
        return True
    if abs(candidate_values[0] - best_values[0]) <= min_delta:
        return candidate_values[1:] > best_values[1:]
    return False


def _small_sample_train_arm(
    train_samples,
    val_samples,
    test_samples,
    selected_subjects,
    split_payload,
    repeat_payload,
    train_size,
    arm,
    real_windows,
    synthetic_windows,
    consistency,
    cfg,
    device,
    output_dir,
):
    repeat_index = int(repeat_payload["repeat"])
    run_dir = os.path.join(
        output_dir,
        "runs",
        f"train_{int(train_size):02d}",
        f"repeat_{repeat_index:02d}",
        str(arm),
    )
    ensure_dir(run_dir)
    result_path = os.path.join(run_dir, "result.json")
    if os.path.isfile(result_path):
        with open(result_path, "r", encoding="utf-8") as file:
            cached_result = json.load(file)
        cached_test = cached_result["test_metrics"]
        print(
            f"[Small Sample Test Reuse] n={train_size} "
            f"repeat={repeat_index} arm={arm} | "
            f"best_epoch={cached_result['best_epoch']} | "
            f"test acc={cached_test['acc']:.4f}, "
            f"kappa={cached_test['kappa']:.4f}, "
            f"macro_f1={cached_test['macro_f1']:.4f}, "
            f"weighted_f1={cached_test['weighted_f1']:.4f} | "
            f"path={result_path}"
        )
        return cached_result

    classifier_seed = int(
        repeat_payload["classifier_seed"] + int(train_size) * 10000
    )
    set_seed(classifier_seed)
    train_loader = DataLoader(
        SleepTextWindowDataset(
            train_samples, None, None, "sleep_stage_classification"
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        SleepTextWindowDataset(
            val_samples, None, None, "sleep_stage_classification"
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        SleepTextWindowDataset(
            test_samples, None, None, "sleep_stage_classification"
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    model = _small_sample_build_stage_model(cfg, device)
    initial_state_sha256 = _small_sample_state_sha256(
        model.state_dict()
    )
    freeze_epochs = max(int(cfg.freeze_backbone_epochs), 0)
    if freeze_epochs > 0:
        set_backbone_trainable(model, False)
        phase = "head_only"
    else:
        configure_stage_backbone_finetuning(model, cfg)
        phase = "last_transformer"
    optimizer = build_optimizer(
        model, "sleep_stage_classification", cfg
    )
    class_weights = None
    if cfg.use_stage_class_weights:
        class_weights = compute_stage_class_weights(
            train_samples,
            cfg.tasks["sleep_stage_classification"]["num_classes"],
        )

    class_names = cfg.tasks["sleep_stage_classification"]["class_names"]
    best_metrics = None
    best_epoch = -1
    patience_counter = 0
    second_stage_started = False
    history = []
    best_path = os.path.join(run_dir, "best_model.pt")
    for epoch in range(max(int(cfg.stage_loso_epochs), 1)):
        if phase == "head_only" and epoch == freeze_epochs:
            configure_stage_backbone_finetuning(model, cfg)
            optimizer = build_optimizer(
                model, "sleep_stage_classification", cfg
            )
            phase = "last_transformer"
            patience_counter = 0

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            "sleep_stage_classification",
            cfg,
            class_weights=class_weights,
        )
        val_metrics = evaluate_stage(
            model,
            val_loader,
            device,
            class_names=class_names,
            save_dir=None,
            seed=classifier_seed,
        )
        test_monitoring_metrics = None
        if bool(getattr(
            cfg, "generation_small_sample_test_each_epoch", False
        )):
            test_monitoring_metrics = evaluate_stage(
                model,
                test_loader,
                device,
                class_names=class_names,
                save_dir=None,
                seed=classifier_seed,
            )
        improved = _small_sample_is_better(
            val_metrics, best_metrics, cfg
        )
        if improved:
            best_metrics = copy.deepcopy(val_metrics)
            best_epoch = epoch + 1
            patience_counter = 0
            _small_sample_atomic_torch({
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "best_epoch": best_epoch,
                "validation_metrics": best_metrics,
                "initial_state_sha256": initial_state_sha256,
            }, best_path)
        else:
            patience_counter += 1
        history.append({
            "epoch": epoch + 1,
            "phase": phase,
            "train": train_metrics,
            "validation": val_metrics,
            "test_monitoring": test_monitoring_metrics,
            "improved": bool(improved),
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
        })
        _small_sample_atomic_json({
            "train_size": int(train_size),
            "repeat": repeat_index,
            "arm": str(arm),
            "selected_subjects": list(selected_subjects),
            "classifier_seed": classifier_seed,
            "initial_state_sha256": initial_state_sha256,
            "selection_source": "fixed_validation_only",
            "test_monitoring_only_not_used_for_selection": bool(
                test_monitoring_metrics is not None
            ),
            "history": history,
        }, os.path.join(run_dir, "progress.json"))
        test_monitoring_text = ""
        if test_monitoring_metrics is not None:
            test_monitoring_text = (
                f" | test acc={test_monitoring_metrics['acc']:.4f}, "
                f"kappa={test_monitoring_metrics['kappa']:.4f}, "
                f"macro_f1="
                f"{test_monitoring_metrics['macro_f1']:.4f}, "
                f"weighted_f1="
                f"{test_monitoring_metrics['weighted_f1']:.4f}"
            )
        print(
            f"[Small Sample] n={train_size} repeat={repeat_index} "
            f"arm={arm} | epoch={epoch + 1}/{cfg.stage_loso_epochs} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val acc={val_metrics['acc']:.4f}, "
            f"kappa={val_metrics['kappa']:.4f}, "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
            f"{test_monitoring_text} | "
            f"best_epoch={best_epoch}"
        )

        if patience_counter >= int(cfg.early_stopping_patience):
            if (
                bool(cfg.enable_plateau_second_transformer_stage)
                and not second_stage_started
                and phase == "last_transformer"
            ):
                enter_second_transformer_finetuning_stage(
                    model, optimizer, cfg
                )
                phase = "last_two_transformers"
                second_stage_started = True
                patience_counter = 0
            else:
                break

    if not os.path.isfile(best_path):
        raise RuntimeError(f"No validation-selected model: {run_dir}")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test_metrics = evaluate_stage(
        model,
        test_loader,
        device,
        class_names=class_names,
        save_dir=None,
        seed=classifier_seed,
    )
    result = {
        "protocol": "fixed_12_4_4_1_2_4_5_6_12_five_paired_repeats",
        "train_size": int(train_size),
        "repeat": repeat_index,
        "arm": str(arm),
        "selected_train_subjects": list(selected_subjects),
        "validation_subjects": split_payload["validation_subjects"],
        "test_subjects": split_payload["test_subjects"],
        "classifier_seed": classifier_seed,
        "initial_state_sha256": initial_state_sha256,
        "selection_source": "fixed_validation_subjects_only",
        "selection_metric": str(
            cfg.generation_small_sample_selection_metric
        ),
        "selection_tiebreakers": list(
            cfg.generation_small_sample_selection_tiebreakers
        ),
        "best_epoch": int(best["best_epoch"]),
        "validation_metrics": best["validation_metrics"],
        "test_evaluated_once_after_selection": not bool(getattr(
            cfg, "generation_small_sample_test_each_epoch", False
        )),
        "final_test_evaluated_after_validation_selection": True,
        "test_evaluated_each_epoch_for_monitoring": bool(getattr(
            cfg, "generation_small_sample_test_each_epoch", False
        )),
        "test_used_for_checkpoint_selection": False,
        "test_metrics": test_metrics,
        "real_train_windows": int(real_windows),
        "synthetic_train_windows": int(synthetic_windows),
        "combined_train_windows": int(len(train_samples)),
        "real_synthetic_ratio": (
            "1:1" if arm == "augmented" else "real_only"
        ),
        "classifier_spectrum_source": (
            "fixed_stft_recomputed_from_final_waveform"
            if cfg.use_stage_relative_spectral_branch
            else "disabled"
        ),
        "independent_generated_spectrum_used_by_classifier": False,
        "generation_spectrum_consistency_l1_mean": (
            float(np.mean(consistency)) if consistency else None
        ),
        "history": history,
    }
    _small_sample_atomic_json(result, result_path)
    print(
        f"[Small Sample Test] n={train_size} "
        f"repeat={repeat_index} arm={arm} | "
        f"best_epoch={result['best_epoch']} | "
        f"test acc={test_metrics['acc']:.4f}, "
        f"kappa={test_metrics['kappa']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}, "
        f"weighted_f1={test_metrics['weighted_f1']:.4f} | "
        f"saved={result_path}"
    )
    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _small_sample_summarize(output_dir, expected_runs):
    rows = []
    runs_root = os.path.join(output_dir, "runs")
    if os.path.isdir(runs_root):
        for root, _, files in os.walk(runs_root):
            if "result.json" not in files:
                continue
            with open(
                os.path.join(root, "result.json"),
                "r",
                encoding="utf-8",
            ) as file:
                result = json.load(file)
            metrics = result["test_metrics"]
            row = {
                "train_size": int(result["train_size"]),
                "repeat": int(result["repeat"]),
                "arm": str(result["arm"]),
                "acc": float(metrics["acc"]),
                "kappa": float(metrics["kappa"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "best_epoch": int(result["best_epoch"]),
            }
            for name, value in metrics.get(
                "per_class_f1", {}
            ).items():
                row[f"f1_{name}"] = float(value)
            rows.append(row)
    rows.sort(key=lambda row: (
        row["train_size"], row["repeat"], row["arm"]
    ))
    if not rows:
        return {
            "completed_runs": 0,
            "expected_runs": int(expected_runs),
        }

    fields = sorted({key for row in rows for key in row})
    with open(
        os.path.join(output_dir, "all_completed_runs.csv"),
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metric_names = [
        "acc", "kappa", "macro_f1", "weighted_f1",
        "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM",
    ]
    summary = {
        "completed_runs": len(rows),
        "expected_runs": int(expected_runs),
        "standard_deviation": "sample_sd_ddof_1",
        "groups": {},
        "paired_deltas_augmented_minus_real": {},
    }
    for train_size in sorted({row["train_size"] for row in rows}):
        size_key = str(train_size)
        summary["groups"][size_key] = {}
        for arm in ("real_only", "augmented"):
            subset = [
                row for row in rows
                if row["train_size"] == train_size
                and row["arm"] == arm
            ]
            if not subset:
                continue
            arm_summary = {"n": len(subset)}
            for metric in metric_names:
                values = [
                    row[metric] for row in subset if metric in row
                ]
                if values:
                    arm_summary[metric] = {
                        "mean": float(np.mean(values)),
                        "sd": (
                            float(np.std(values, ddof=1))
                            if len(values) > 1 else None
                        ),
                        "values": [float(value) for value in values],
                    }
            summary["groups"][size_key][arm] = arm_summary
        by_key = {
            (row["repeat"], row["arm"]): row
            for row in rows if row["train_size"] == train_size
        }
        delta_summary = {}
        for metric in metric_names:
            values = []
            repeat_ids = sorted({
                int(row["repeat"])
                for row in rows if row["train_size"] == train_size
            })
            for repeat in repeat_ids:
                real = by_key.get((repeat, "real_only"))
                augmented = by_key.get((repeat, "augmented"))
                if (
                    real is not None and augmented is not None
                    and metric in real and metric in augmented
                ):
                    values.append(augmented[metric] - real[metric])
            if values:
                delta_summary[metric] = {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sd": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1 else None
                    ),
                    "values": [float(value) for value in values],
                }
        summary[
            "paired_deltas_augmented_minus_real"
        ][size_key] = delta_summary
    _small_sample_atomic_json(
        summary, os.path.join(output_dir, "summary_mean_sd.json")
    )
    return summary


def stage_generation_augmentation_task_run(records, cfg, device):
    """
    Dataset-selectable SleepGPT-style small-sample experiment.

    DREAMS uses a fixed 12-subject train pool / 4 validation / 4 test split.
    Conditions: nested 1/2/4/7/12-subject labeled training subsets.
    Repeats: five independent paired real-only vs real+synthetic runs.
    The validation set selects checkpoints; the test set is evaluated once.
    """
    print(f"{'=' * 80}")
    selected_size_display = getattr(
        cfg, "generation_small_sample_selected_train_size", 1
    )
    run_mode_display = str(
        getattr(
            cfg,
            "generation_small_sample_run_mode",
            "prepare_generation_cache",
        )
    )
    repeat_display = int(
        getattr(cfg, "generation_small_sample_repeats", 5)
    )
    dataset_name = str(
        getattr(cfg, "generation_small_sample_dataset_name", "dataset")
    )
    split_display = (
        f"{int(cfg.generation_small_sample_train_pool_subjects)}/"
        f"{int(cfg.generation_small_sample_validation_subjects)}/"
        f"{int(cfg.generation_small_sample_test_subjects)}"
    )
    configured_size_display = [
        int(value)
        for value in cfg.generation_small_sample_train_sizes
    ]
    if (
        isinstance(selected_size_display, str)
        and selected_size_display.strip().lower() == "all"
    ):
        selected_size_display = (
            "all (" + ",".join(
                str(value) for value in configured_size_display
            ) + ")"
        )
        selected_run_count = (
            repeat_display * len(configured_size_display) * 2
        )
    else:
        selected_run_count = repeat_display * 2
    if run_mode_display.strip().lower() == "prepare_generation_cache":
        selected_run_count = 0
    print(
        "Task: sleep_stage_generation_augmentation | "
        f"{dataset_name} {split_display} | "
        f"mode={run_mode_display} | "
        f"selected train size={selected_size_display} | "
        f"repeats={repeat_display} | "
        f"selected classifier runs={selected_run_count}"
    )
    if run_mode_display.strip().lower() == "prepare_generation_cache":
        print(
            "[Generation Cache] One fixed generator will be fitted on all "
            "12 DREAMS training-pool subjects without stage labels, then "
            "five repeat-specific Mask caches will be prepared for the full "
            "training pool. The selected downstream size remains "
            f"{selected_size_display}."
        )
    if not records:
        raise RuntimeError(f"No {dataset_name} records were loaded.")

    # Task-local copy guarantees that ordinary sleep staging, slow wave,
    # disorder and spindle configurations remain unchanged.
    cfg = copy.deepcopy(cfg)
    cfg.use_stage_relative_spectral_branch = bool(
        getattr(cfg, "generation_stage_use_fixed_stft_branch", True)
    )
    if cfg.use_stage_relative_spectral_branch:
        cfg.stage_spectral_mode = "fixed_stft_log"
    cfg.use_fold_self_supervised_adaptation = False
    cfg.use_saved_stage_mc_gated_postprocess = False
    cfg.use_viterbi_postprocess = False
    cfg.save_best_stage_visualizations = False
    cfg.early_stopping_metric = str(
        cfg.generation_small_sample_selection_metric
    )

    # The button selects an approved nested condition. It never changes which
    # 12 subjects are used for the single unlabeled generator adaptation.
    selected_size_request = getattr(
        cfg, "generation_small_sample_selected_train_size", 1
    )
    if not (
        isinstance(selected_size_request, str)
        and selected_size_request.strip().lower() == "all"
    ):
        try:
            selected_size_request = int(selected_size_request)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "generation_small_sample_selected_train_size must be an "
                "integer or 'all'."
            ) from error
        configured_sizes = sorted({
            int(value)
            for value in cfg.generation_small_sample_train_sizes
        })
        if selected_size_request not in configured_sizes:
            raise ValueError(
                "generation_small_sample_selected_train_size must be "
                f"one of {configured_sizes}, or 'all'; got "
                f"{selected_size_request}."
            )

    output_dir = os.path.abspath(cfg.generation_small_sample_output_dir)
    ensure_dir(output_dir)
    split_payload = _small_sample_load_or_create_split(
        records, cfg, output_dir
    )
    repeat_manifest = _small_sample_load_or_create_repeats(
        split_payload, cfg, output_dir
    )
    record_map = {
        str(record["subject_id"]): record for record in records
    }
    train_pool_records = [
        record_map[str(value)]
        for value in split_payload["train_pool_subjects"]
    ]
    val_records = [
        record_map[str(value)]
        for value in split_payload["validation_subjects"]
    ]
    test_records = [
        record_map[str(value)]
        for value in split_payload["test_subjects"]
    ]
    val_samples = build_stage_samples(
        val_records, seq_len=cfg.seq_len, stride=cfg.stride
    )
    test_samples = build_stage_samples(
        test_records, seq_len=cfg.seq_len, stride=cfg.stride
    )
    if not val_samples or not test_samples:
        raise RuntimeError("Fixed validation or test set is empty.")

    backbone_path = os.path.abspath(cfg.generation_direct_backbone_path)
    if not os.path.isfile(backbone_path):
        raise FileNotFoundError(
            f"Direct generation Backbone not found: {backbone_path}"
        )
    backbone_hash = _small_sample_file_sha256(backbone_path)
    run_mode = str(
        cfg.generation_small_sample_run_mode
    ).strip().lower()
    valid_run_modes = {
        "prepare_generation_cache",
        "downstream_training",
    }
    if run_mode not in valid_run_modes:
        raise ValueError(
            "generation_small_sample_run_mode must be one of "
            f"{sorted(valid_run_modes)}, got {run_mode!r}."
        )
    cache_root = os.path.join(output_dir, "generation_cache")
    ensure_dir(cache_root)
    max_repeats = getattr(
        cfg, "generation_small_sample_max_repeats", None
    )
    repeats = list(repeat_manifest["repeats"])
    if max_repeats is not None:
        repeats = repeats[:max(1, int(max_repeats))]

    configured_sizes = [
        int(value) for value in cfg.generation_small_sample_train_sizes
    ]
    selected_size = getattr(
        cfg, "generation_small_sample_selected_train_size", 1
    )
    if isinstance(selected_size, str) and selected_size.strip().lower() == "all":
        train_sizes = configured_sizes
    else:
        try:
            selected_size = int(selected_size)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "generation_small_sample_selected_train_size must be "
                f"one of {configured_sizes} or 'all'."
            ) from error
        if selected_size not in configured_sizes:
            raise ValueError(
                "generation_small_sample_selected_train_size must be "
                f"one of {configured_sizes} or 'all', got {selected_size}."
            )
        train_sizes = [selected_size]

    _small_sample_atomic_json({
        "protocol": (
            f"{dataset_name} fixed {split_display}; "
            f"train sizes={cfg.generation_small_sample_train_sizes}; "
            "five paired repeats"
        ),
        "dataset_name": dataset_name,
        "dataset_dirs": list(cfg.generation_stage_train_dirs),
        "generator_source": "04_internal_hierarchical_generation_head",
        "imports_03generation_decoder": False,
        "source_backbone_path": backbone_path,
        "source_backbone_sha256": backbone_hash,
        "generator_format_version": 10,
        "generator_model_type": "04_hierarchical_event_context_generator",
        "generator_scope": "one_fixed_generator_for_all_train_sizes_and_repeats",
        "generator_training_subjects": "fixed_12_subject_train_pool_unlabeled_waveforms",
        "generator_internal_split": "per_subject_temporal_90_percent_fit_10_percent_reconstruction_validation",
        "generator_context": "full_20_epochs_with_one_random_target_epoch",
        "target_epoch_mask_ratio": 0.75,
        "local_decoder_tokens": 90,
        "cache_scope": "five_mask_seeds_times_all_12_training_pool_subjects",
        "sleep_stage_labels_used_for_generator_training": False,
        "downstream_validation_or_test_used_for_generator_training": False,
        "generation_and_downstream_separate": True,
        "current_run_mode": run_mode,
        "selected_train_size_button": (
            cfg.generation_small_sample_selected_train_size
        ),
        "validation_selects_checkpoint": True,
        "test_evaluated_during_training": bool(getattr(
            cfg, "generation_small_sample_test_each_epoch", False
        )),
        "test_monitoring_only_not_used_for_selection": bool(getattr(
            cfg, "generation_small_sample_test_each_epoch", False
        )),
        "other_04_tasks_changed": False,
    }, os.path.join(output_dir, "experiment_protocol.json"))

    if run_mode == "prepare_generation_cache":
        fixed_training_subjects = list(
            split_payload["train_pool_subjects"]
        )
        if len(fixed_training_subjects) != 12:
            raise RuntimeError(
                "The fixed DREAMS generator protocol requires exactly 12 "
                f"training-pool subjects, found {len(fixed_training_subjects)}."
            )
        generator, settings, generator_checkpoint, model_path = (
            _direct_train_or_load_generator(
                train_pool_records,
                fixed_training_subjects,
                cfg,
                device,
                output_dir,
            )
        )
        generator_hash = _small_sample_file_sha256(model_path)
        for repeat_position, repeat_payload in enumerate(repeats, 1):
            repeat_index = int(repeat_payload["repeat"])
            print(
                f"[Generation Repeat Start] repeat={repeat_index} "
                f"({repeat_position}/{len(repeats)}) | fixed_generator=True | "
                "cache_subjects=12"
            )
            repeat_summary = _small_sample_prepare_repeat_cache(
                train_pool_records,
                repeat_payload,
                generator,
                settings,
                generator_checkpoint,
                generator_hash,
                cfg,
                device,
                cache_root,
            )
            consistency_mean = repeat_summary[
                "spectrum_consistency_l1_mean"
            ]
            consistency_text = (
                f"{consistency_mean:.6f}"
                if consistency_mean is not None
                else "n/a (all subjects reused)"
            )
            print(
                f"[Generation Repeat Complete] repeat={repeat_index} | "
                f"generated subjects={repeat_summary['generated_subjects']} | "
                f"reused subjects={repeat_summary['reused_subjects']} | "
                f"cached epochs={repeat_summary['total_cached_epochs']} | "
                f"spectrum_consistency_l1={consistency_text} | "
                f"cache_dir={repeat_summary['cache_dir']}"
            )
        del generator
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            "[Complete] The fixed 12-subject generator and all five "
            "full-pool generation caches are ready. "
            "No sleep-stage classifier was constructed. Set "
            "generation_small_sample_run_mode='downstream_training' "
            "and run 04 again."
        )
        return {
            "output_dir": output_dir,
            "generation_cache_only": True,
            "run_mode": run_mode,
        }

    # Downstream mode never invokes the generator. Validate the complete cache
    # set before starting even the first real-only classifier, so a stale cache
    # cannot fail after hours of training.
    fixed_generator_path = _direct_generator_model_path(output_dir)
    if not os.path.isfile(fixed_generator_path):
        raise RuntimeError(
            "Missing fixed 12-subject generator. First run "
            "generation_small_sample_run_mode='prepare_generation_cache': "
            f"{fixed_generator_path}"
        )
    generator_hash = _small_sample_file_sha256(fixed_generator_path)
    for train_size in train_sizes:
        for repeat_payload in repeats:
            repeat_index = int(repeat_payload["repeat"])
            repeat_seed = int(repeat_payload["generation_seed"])
            selected_subjects = list(
                repeat_payload["subjects_by_train_size"][str(train_size)]
            )
            repeat_dir = os.path.join(
                cache_root,
                f"repeat_{repeat_index:02d}",
            )
            for subject_id in selected_subjects:
                cache_path = os.path.join(
                    repeat_dir, _small_sample_subject_filename(subject_id)
                )
                if not _small_sample_cache_valid(
                    cache_path,
                    subject_id,
                    repeat_seed,
                    generator_hash,
                ):
                    raise RuntimeError(
                        "Missing or stale generation cache. First run with "
                        "generation_small_sample_run_mode="
                        f"'prepare_generation_cache': {cache_path}"
                    )
    total_expected_runs = len(repeats) * len(configured_sizes) * 2
    selected_expected_runs = len(repeats) * len(train_sizes) * 2
    print(
        "[Small Sample Selection] "
        f"selected_train_sizes={train_sizes} | repeats={len(repeats)} | "
        f"paired_classifier_runs={selected_expected_runs}"
    )
    results = []
    for repeat_payload in repeats:
        repeat_index = int(repeat_payload["repeat"])
        for train_size in train_sizes:
            selected_subjects = list(
                repeat_payload["subjects_by_train_size"][
                    str(train_size)
                ]
            )
            selected_records = [
                record_map[str(value)] for value in selected_subjects
            ]
            real_samples = build_stage_samples(
                selected_records,
                seq_len=cfg.seq_len,
                stride=cfg.stride,
            )
            if not real_samples:
                raise RuntimeError(
                    f"No real samples: repeat={repeat_index}, n={train_size}"
                )
            real_result = _small_sample_train_arm(
                real_samples,
                val_samples,
                test_samples,
                selected_subjects,
                split_payload,
                repeat_payload,
                train_size,
                "real_only",
                len(real_samples),
                0,
                [],
                cfg,
                device,
                output_dir,
            )
            generated_items, consistency = (
                _small_sample_load_generated_items(
                    selected_subjects,
                    repeat_payload,
                    generator_hash,
                    cfg,
                    cache_root,
                )
            )
            synthetic_records = build_generated_subject_copy_records(
                selected_records, generated_items
            )
            synthetic_samples = build_stage_samples(
                synthetic_records,
                seq_len=cfg.seq_len,
                stride=cfg.stride,
            )
            if len(synthetic_samples) != len(real_samples):
                raise RuntimeError(
                    "Fixed 1:1 protocol violated: "
                    f"real={len(real_samples)}, "
                    f"synthetic={len(synthetic_samples)}"
                )
            augmented_samples = real_samples + synthetic_samples
            augmented_result = _small_sample_train_arm(
                augmented_samples,
                val_samples,
                test_samples,
                selected_subjects,
                split_payload,
                repeat_payload,
                train_size,
                "augmented",
                len(real_samples),
                len(synthetic_samples),
                consistency,
                cfg,
                device,
                output_dir,
            )
            if (
                real_result["initial_state_sha256"]
                != augmented_result["initial_state_sha256"]
            ):
                raise RuntimeError(
                    "Paired arms did not start from identical weights."
                )
            real_test = real_result["test_metrics"]
            augmented_test = augmented_result["test_metrics"]
            delta_acc = (
                float(augmented_test["acc"])
                - float(real_test["acc"])
            )
            delta_macro_f1 = (
                float(augmented_test["macro_f1"])
                - float(real_test["macro_f1"])
            )
            delta_kappa = (
                float(augmented_test["kappa"])
                - float(real_test["kappa"])
            )
            print(
                f"[Paired Delta] n={train_size} "
                f"repeat={repeat_index} | "
                f"delta_ACC(aug-real)={delta_acc:+.4f} | "
                f"delta_MF1(aug-real)={delta_macro_f1:+.4f} | "
                f"delta_Kappa(aug-real)={delta_kappa:+.4f}"
            )
            results.extend([real_result, augmented_result])
            # Keep a valid partial summary after every completed pair so a
            # Keep the long multi-condition run inspectable while unfinished.
            _small_sample_summarize(
                output_dir, total_expected_runs
            )
            del (
                generated_items,
                synthetic_records,
                synthetic_samples,
                augmented_samples,
                real_samples,
            )
            gc.collect()

    summary = _small_sample_summarize(
        output_dir, total_expected_runs
    )
    paired_delta_groups = summary.get(
        "paired_deltas_augmented_minus_real", {}
    )
    for train_size in train_sizes:
        delta_group = paired_delta_groups.get(str(train_size), {})
        required_metrics = ("acc", "macro_f1", "kappa")
        if not all(
            metric in delta_group
            and int(delta_group[metric].get("n", 0)) == 5
            for metric in required_metrics
        ):
            continue
        acc_delta = delta_group["acc"]
        macro_f1_delta = delta_group["macro_f1"]
        kappa_delta = delta_group["kappa"]
        print(
            f"[Five-Repeat Delta Summary] n={train_size} | "
            f"delta_ACC={acc_delta['mean']:+.4f}"
            f"+/-{acc_delta['sd']:.4f} | "
            f"delta_MF1={macro_f1_delta['mean']:+.4f}"
            f"+/-{macro_f1_delta['sd']:.4f} | "
            f"delta_Kappa={kappa_delta['mean']:+.4f}"
            f"+/-{kappa_delta['sd']:.4f} | "
            "definition=augmented-real, sample_sd(ddof=1)"
        )
    print(
        f"[Complete] selected_run_target={selected_expected_runs} | "
        "all_completed_runs_in_output="
        f"{summary['completed_runs']}/{total_expected_runs} | "
        f"summary={os.path.join(output_dir, 'summary_mean_sd.json')}"
    )
    return {
        "output_dir": output_dir,
        "split": split_payload,
        "repeat_manifest": repeat_manifest,
        "summary": summary,
        "results": results,
    }


def _split_slow_wave_train_val_records(trainval_records, cfg, fold_idx):
    if len(trainval_records) < 2:
        return trainval_records, trainval_records

    inner_idx = np.arange(len(trainval_records))
    val_size = max(1, int(round(len(trainval_records) * 0.2)))
    val_size = min(val_size, len(trainval_records) - 1)

    tr_idx, val_idx = train_test_split(
        inner_idx,
        test_size=val_size,
        random_state=cfg.seed + fold_idx,
        shuffle=True,
    )

    train_records = [trainval_records[i] for i in tr_idx]
    val_records = [trainval_records[i] for i in val_idx]
    return train_records, val_records


def slow_wave_subject_loso_cv_run(records, cfg, device):
    if len(records) == 0:
        print("slow_wave_detection: no valid subject records, skip.")
        return None

    task_name = "slow_wave_detection"
    subject_ids = np.array([str(r["subject_id"]) for r in records])
    dummy_x = np.arange(len(records))
    dummy_y = np.zeros(len(records), dtype=np.int64)

    logo = LeaveOneGroupOut()
    total_folds = len(records)
    task_out_dir = task_output_dir(cfg, task_name)
    ensure_dir(task_out_dir)

    fold_results = []
    class_names = cfg.tasks[task_name]["class_names"]

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        logo.split(dummy_x, dummy_y, groups=subject_ids), 1
    ):
        trainval_records = [records[i] for i in trainval_idx]
        test_records = [records[i] for i in test_idx]

        train_records, val_records = _split_slow_wave_train_val_records(
            trainval_records, cfg, fold_idx
        )

        train_samples = build_slow_wave_samples(train_records, seq_len=cfg.seq_len, stride=cfg.stride)
        val_samples = build_slow_wave_samples(val_records, seq_len=cfg.seq_len, stride=cfg.stride)
        test_samples = build_slow_wave_samples(test_records, seq_len=cfg.seq_len, stride=cfg.stride)

        test_subjects = sorted({str(r["subject_id"]) for r in test_records})
        train_subjects = sorted({str(r["subject_id"]) for r in train_records})
        val_subjects = sorted({str(r["subject_id"]) for r in val_records})

        print(f"\n{'=' * 80}")
        print(f"Task: {task_name} | Strict LOSO Fold {fold_idx}/{total_folds}")
        print(f"Train subjects: {train_subjects}")
        print(f"Val subjects: {val_subjects}")
        print(f"Test subjects: {test_subjects}")
        print(
            f"Train windows: {len(train_samples)} | "
            f"Val windows: {len(val_samples)} | "
            f"Test windows: {len(test_samples)}"
        )
        print(f"{'=' * 80}")

        fold_name = f"subject_{test_subjects[0]}" if len(test_subjects) == 1 else f"fold_{fold_idx}"
        fold_dir = os.path.join(task_out_dir, fold_name)
        val_dir = os.path.join(fold_dir, "inner_val")
        test_dir = os.path.join(fold_dir, "outer_test")
        ensure_dir(fold_dir)
        ensure_dir(val_dir)
        ensure_dir(test_dir)

        best_state = run_one_fold(
            task_name,
            fold_idx - 1,
            train_samples,
            val_samples,
            cfg,
            device,
            val_dir,
        )

        model_path = os.path.join(fold_dir, "best_model.pth")
        torch.save(best_state["model"], model_path)

        test_ds = SleepTextWindowDataset(
            test_samples, None, None, "slow_wave_detection"
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        )

        backbone = SleepTextBackbone(
            pretrained_model_dir=None,
            seq_len=cfg.seq_len,
            dropout=cfg.dropout,
            seq_num_layers=cfg.seq_num_layers,
            seq_num_heads=cfg.seq_num_heads,
            bert_chunk_size=0,
            eeg_hidden_dim=cfg.eeg_hidden_dim,
            text_proj_dim=0,
            fusion_dim=0,
            eeg_kernel_size=cfg.eeg_kernel_size
        )
        maybe_load_eeg_pretrained_weights(backbone, cfg)
        model = MultiTaskSleepModel(backbone, cfg=cfg).to(device)
        model.load_state_dict(best_state["model"], strict=True)

        test_metrics = evaluate_slow_wave(
            model,
            test_loader,
            device,
            class_names=class_names,
            save_dir=test_dir,
            seed=cfg.seed + fold_idx,
        )

        print(
            f"[Test] Fold {fold_idx} | "
            f"acc={test_metrics['acc']:.4f}, "
            f"kappa={test_metrics['kappa']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"weighted_f1={test_metrics.get('weighted_f1', 0.0):.4f}"
        )
        print_test_metrics_detail(test_metrics)

        with open(os.path.join(fold_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        result_json = {
            "task_name": task_name,
            **experiment_metadata(cfg),
            "fold_idx": fold_idx,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
            "val_metrics": best_state["metrics"],
            "test_metrics": test_metrics,
            "history": best_state["history"],
            "best_epoch": best_state.get("best_epoch", None),
        }
        with open(os.path.join(fold_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        fold_results.append(result_json)

    summary = {
        "task_name": task_name,
        **experiment_metadata(cfg),
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }

    if len(fold_results) > 0:
        metric_names = ["acc", "kappa", "macro_f1", "weighted_f1"]
        for metric_name in metric_names:
            metric_values = [
                fr["test_metrics"].get(metric_name, 0.0) for fr in fold_results
            ]
            summary[f"mean_{metric_name}"] = float(np.mean(metric_values))
            summary[f"std_{metric_name}"] = float(np.std(metric_values))

    with open(os.path.join(task_out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{task_name} completed.")
    if "mean_acc" in summary:
        print(
            f"Mean acc={summary['mean_acc']:.4f}+-{summary['std_acc']:.4f}, "
            f"mean kappa={summary['mean_kappa']:.4f}+-{summary['std_kappa']:.4f}, "
            f"mean macro_f1={summary['mean_macro_f1']:.4f}+-{summary['std_macro_f1']:.4f}, "
            f"mean weighted_f1={summary['mean_weighted_f1']:.4f}+-{summary['std_weighted_f1']:.4f}"
        )

    return summary


def evaluate_disorder_from_state_dict(state_dict, test_samples, cfg, device, save_dir=None, seed=None):
    test_ds = SleepTextWindowDataset(
        test_samples, None, None, "sleep_disorder_classification"
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size
    )
    maybe_load_eeg_pretrained_weights(backbone, cfg)

    model = MultiTaskSleepModel(backbone, cfg=cfg).to(device)
    model.load_state_dict(state_dict, strict=True)

    return evaluate_disorder(
        model,
        test_loader,
        device,
        class_names=cfg.tasks["sleep_disorder_classification"]["class_names"],
        save_dir=save_dir,
        seed=cfg.seed if seed is None else seed,
    )


def _split_disorder_train_val_records(
    trainval_records, trainval_y, cfg, fold_idx, split_seed=None
):
    if len(trainval_records) < 2:
        return trainval_records, trainval_records

    trainval_y = np.asarray(trainval_y, dtype=np.int64)
    inner_idx = np.arange(len(trainval_records))
    unique_labels, counts = np.unique(trainval_y, return_counts=True)

    val_size = max(1, int(round(len(trainval_records) * 0.2)))
    if len(unique_labels) > 1:
        val_size = max(val_size, len(unique_labels))
    val_size = min(val_size, len(trainval_records) - 1)

    can_stratify = (
        len(unique_labels) > 1
        and counts.min() >= 2
        and val_size >= len(unique_labels)
        and (len(trainval_records) - val_size) >= len(unique_labels)
    )

    stratify = trainval_y if can_stratify else None
    if (not can_stratify) and len(unique_labels) > 1:
        print(f"[Disorder CV] Fold {fold_idx}: inner split fallback to unstratified validation.")

    base_seed = int(cfg.seed if split_seed is None else split_seed)
    tr_idx, val_idx = train_test_split(
        inner_idx,
        test_size=val_size,
        random_state=base_seed + fold_idx,
        stratify=stratify,
        shuffle=True,
    )

    train_records = [trainval_records[i] for i in tr_idx]
    val_records = [trainval_records[i] for i in val_idx]
    return train_records, val_records


class DisorderSubjectFeatureDataset(Dataset):
    """One sample is one complete CAP recording, never an overlapping window."""

    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "subject_id": str(record["subject_id"]),
            "features": torch.as_tensor(record["features"], dtype=torch.float32),
            "stages": torch.as_tensor(record["stages"], dtype=torch.long),
            "label": int(record["sleep_disorder_label"]),
        }


def collate_disorder_subject_features(batch):
    lengths = torch.as_tensor(
        [item["features"].shape[0] for item in batch], dtype=torch.long
    )
    max_len = int(lengths.max().item())
    hidden_dim = int(batch[0]["features"].shape[-1])
    features = torch.zeros(len(batch), max_len, hidden_dim, dtype=torch.float32)
    stages = torch.full((len(batch), max_len), -100, dtype=torch.long)
    valid_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for index, item in enumerate(batch):
        length = int(item["features"].shape[0])
        features[index, :length] = item["features"]
        stages[index, :length] = item["stages"]
        valid_mask[index, :length] = True
    return {
        "subject_id": [item["subject_id"] for item in batch],
        "features": features,
        "stages": stages,
        "valid_mask": valid_mask,
        "lengths": lengths,
        "labels": torch.as_tensor(
            [item["label"] for item in batch], dtype=torch.long
        ),
    }


class StageAwareDisorderHead(nn.Module):
    """Whole-night stage-mean head for sleep-disorder classification.

    The frozen event backbone already supplies contextual epoch embeddings.
    For each subject, valid epochs are averaged separately within
    W/N1/N2/N3/REM. Stage identity, stage fraction, and a learned token for a
    missing stage are retained before the five stage representations are
    combined by stage attention.
    """

    def __init__(
        self,
        input_dim=256,
        attention_hidden_dim=64,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()
        self.num_stages = 5
        self.input_norm = nn.LayerNorm(input_dim)
        self.stage_embedding = nn.Parameter(
            torch.empty(self.num_stages, input_dim)
        )
        self.missing_stage_token = nn.Parameter(
            torch.empty(self.num_stages, input_dim)
        )
        self.stage_fraction_proj = nn.Sequential(
            nn.Linear(1, input_dim),
            nn.Tanh(),
        )
        self.stage_attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )
        nn.init.normal_(self.stage_embedding, std=0.02)
        nn.init.normal_(self.missing_stage_token, std=0.02)

    def forward(
        self, features, stages, valid_mask, lengths, return_embedding=False
    ):
        # ``lengths`` remains in the interface because the collate/evaluation
        # code also uses it. Mean pooling itself is governed by valid_mask.
        _ = lengths
        normalized = self.input_norm(features)

        stage_tokens = []
        valid_count = valid_mask.sum(dim=1).clamp(min=1).to(features.dtype)
        for stage_index in range(self.num_stages):
            stage_mask = valid_mask & stages.eq(stage_index)
            has_stage = stage_mask.any(dim=1)
            weights = stage_mask.to(normalized.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = torch.sum(normalized * weights.unsqueeze(-1), dim=1)
            missing = self.missing_stage_token[stage_index].unsqueeze(0).expand_as(pooled)
            pooled = torch.where(has_stage.unsqueeze(-1), pooled, missing)
            fraction = stage_mask.sum(dim=1).to(features.dtype) / valid_count
            pooled = (
                pooled
                + self.stage_embedding[stage_index].unsqueeze(0)
                + self.stage_fraction_proj(fraction.unsqueeze(-1))
            )
            stage_tokens.append(pooled)

        stage_tokens = torch.stack(stage_tokens, dim=1)
        stage_weights = torch.softmax(
            self.stage_attention(stage_tokens).squeeze(-1), dim=1
        )
        subject_embedding = torch.sum(
            stage_tokens * stage_weights.unsqueeze(-1), dim=1
        )
        logits = self.classifier(subject_embedding)
        if return_embedding:
            return logits, subject_embedding, stage_weights
        return logits


def _build_frozen_disorder_backbone(cfg, device):
    backbone = SleepTextBackbone(
        pretrained_model_dir=None,
        seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=0,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=0,
        fusion_dim=0,
        eeg_kernel_size=cfg.eeg_kernel_size,
    ).to(device)
    loaded = load_eeg_pretrained_weights(
        backbone,
        cfg.eeg_pretrained_model_path,
        cfg.load_full_backbone_pretrain,
        interpolate_pos_embed=bool(
            getattr(cfg, "interpolate_pretrained_pos_embed", False)
        ),
    )
    if not loaded:
        raise RuntimeError(
            "CAP disorder classification requires the event-detached pretrained backbone."
        )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()
    return backbone


def extract_full_night_disorder_features(records, cfg, device):
    """Create one contextual embedding per original epoch.

    The backbone still receives its native 20-epoch windows. Overlapping
    outputs are averaged back onto their original epoch index, so a subject is
    represented by one ordered full-night sequence rather than repeated
    window-level labels.
    """
    backbone = _build_frozen_disorder_backbone(cfg, device)
    feature_records = []
    feature_batch_size = max(1, int(cfg.disorder_feature_batch_size))

    with torch.inference_mode():
        for record in tqdm(records, desc="Disorder frozen full-night features"):
            label = int(record["sleep_disorder_label"])
            if label not in {0, 1, 2}:
                continue
            signal_epochs, eog_available = build_signal_epochs(record)
            raw_stages = np.asarray(
                record.get("true_sleep_stages", []), dtype=np.int64
            ).reshape(-1)
            n_epochs = min(len(signal_epochs), len(raw_stages))
            if n_epochs <= 0:
                continue
            signal_epochs = signal_epochs[:n_epochs]
            raw_stages = raw_stages[:n_epochs]
            if not np.isin(raw_stages, [0, 1, 2, 3, 4]).any():
                raise ValueError(
                    f"{record['subject_id']} has no valid expert CAP sleep stages. "
                    "Re-run 01sleeppreprocessing_morphological_yasa_cap_two_multimodal.py."
                )

            starts = build_window_starts(n_epochs, cfg.seq_len, cfg.stride)
            feature_sum = torch.zeros(n_epochs, cfg.eeg_hidden_dim, dtype=torch.float32)
            feature_count = torch.zeros(n_epochs, 1, dtype=torch.float32)

            for offset in range(0, len(starts), feature_batch_size):
                chunk_starts = starts[offset:offset + feature_batch_size]
                windows = []
                valid_masks = []
                valid_lengths = []
                for start in chunk_starts:
                    end = min(start + cfg.seq_len, n_epochs)
                    valid_len = int(end - start)
                    valid_mask = pad_valid_mask(valid_len, cfg.seq_len)
                    window = pad_eeg_epochs(
                        signal_epochs[start:end], cfg.seq_len
                    )
                    windows.append(normalize_eeg_window(window, valid_mask))
                    valid_masks.append(valid_mask)
                    valid_lengths.append(valid_len)

                eeg = torch.as_tensor(
                    np.stack(windows), dtype=torch.float32, device=device
                )
                batch_valid_mask = torch.as_tensor(
                    np.asarray(valid_masks), dtype=torch.long, device=device
                )
                availability = torch.as_tensor(
                    np.repeat(eog_available[None, :], len(chunk_starts), axis=0),
                    dtype=torch.float32,
                    device=device,
                )
                contextual = backbone(
                    eeg,
                    eog_available=availability,
                    valid_mask=batch_valid_mask,
                ).float().cpu()
                for batch_index, (start, valid_len) in enumerate(
                    zip(chunk_starts, valid_lengths)
                ):
                    end = start + valid_len
                    feature_sum[start:end] += contextual[batch_index, :valid_len]
                    feature_count[start:end] += 1.0

            if torch.any(feature_count.eq(0)):
                raise RuntimeError(
                    f"Full-night stitching left uncovered epochs for {record['subject_id']}."
                )
            unique_features = feature_sum / feature_count
            feature_records.append({
                "subject_id": str(record["subject_id"]),
                "features": unique_features.numpy().astype(np.float32),
                "stages": raw_stages.astype(np.int64),
                "sleep_disorder_label": label,
                "n_windows": int(len(starts)),
                "n_epochs": int(n_epochs),
            })

    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feature_records


def build_disorder_stage_rf_tensors(
    records, num_stages=5, include_stage_std=True
):
    """Build one feature vector per subject and sleep stage.

    Epoch embeddings are normalized exactly like ``nn.LayerNorm`` without
    trainable affine parameters. Each W/N1/N2/N3/REM vector contains that
    stage's mean, optionally its population standard deviation, and all five
    stage fractions. ``stage_available`` marks whether the subject truly has
    the stage; missing-stage zero vectors are never passed to a classifier.
    No disease label is used during feature creation.
    """
    subject_stage_vectors = []
    subject_stage_available = []
    labels = []
    subject_ids = []
    expected_dim = None
    for record in records:
        features = np.asarray(record["features"], dtype=np.float32)
        stages = np.asarray(record["stages"], dtype=np.int64).reshape(-1)
        if features.ndim != 2:
            raise ValueError(
                f"{record['subject_id']} disorder features must be 2-D, "
                f"got {features.shape}."
            )
        n_epochs = min(len(features), len(stages))
        if n_epochs <= 0:
            raise ValueError(
                f"{record['subject_id']} has no disorder epochs."
            )
        features = features[:n_epochs]
        stages = stages[:n_epochs]
        if expected_dim is None:
            expected_dim = int(features.shape[1])
        elif int(features.shape[1]) != expected_dim:
            raise ValueError("Disorder backbone feature dimensions differ.")

        epoch_mean = features.mean(axis=1, keepdims=True)
        epoch_variance = np.mean(
            np.square(features - epoch_mean), axis=1, keepdims=True
        )
        normalized = (features - epoch_mean) / np.sqrt(
            epoch_variance + 1e-5
        )
        valid_stage = (stages >= 0) & (stages < int(num_stages))
        valid_count = int(valid_stage.sum())
        if valid_count <= 0:
            raise ValueError(
                f"{record['subject_id']} has no valid W/N1/N2/N3/REM stages."
            )

        stage_means = []
        stage_stds = []
        stage_available = []
        stage_fractions = []
        for stage_index in range(int(num_stages)):
            stage_mask = valid_stage & (stages == stage_index)
            has_stage = bool(np.any(stage_mask))
            if has_stage:
                stage_features = normalized[stage_mask]
                stage_mean = stage_features.mean(axis=0)
                stage_std = stage_features.std(axis=0, ddof=0)
            else:
                stage_mean = np.zeros(expected_dim, dtype=np.float32)
                stage_std = np.zeros(expected_dim, dtype=np.float32)
            stage_means.append(np.asarray(stage_mean, dtype=np.float32))
            stage_stds.append(np.asarray(stage_std, dtype=np.float32))
            stage_available.append(has_stage)
            stage_fractions.append(float(stage_mask.sum()) / valid_count)

        fractions = np.asarray(stage_fractions, dtype=np.float32)
        stage_vectors = []
        for stage_index in range(int(num_stages)):
            statistics = [stage_means[stage_index]]
            if bool(include_stage_std):
                statistics.append(stage_stds[stage_index])
            statistics.append(fractions)
            stage_vector = np.concatenate(statistics, axis=0)
            stage_vectors.append(np.nan_to_num(stage_vector, copy=False))
        subject_stage_vectors.append(np.stack(stage_vectors))
        subject_stage_available.append(
            np.asarray(stage_available, dtype=bool)
        )
        labels.append(int(record["sleep_disorder_label"]))
        subject_ids.append(str(record["subject_id"]))

    if not subject_stage_vectors:
        return (
            np.empty((0, int(num_stages), 0), dtype=np.float32),
            np.empty((0, int(num_stages)), dtype=bool),
            np.empty(0, dtype=np.int64),
            [],
        )
    return (
        np.stack(subject_stage_vectors).astype(np.float32),
        np.stack(subject_stage_available).astype(bool),
        np.asarray(labels, dtype=np.int64),
        subject_ids,
    )


def build_disorder_sleep_architecture_features(
    records, num_stages=5, epoch_seconds=30.0
):
    """Build compact, ordered whole-night sleep-architecture features.

    Invalid/unknown epochs break bouts and are never bridged when transitions
    are counted. The feature construction is label-independent and retains all
    W/N1/N2/N3/REM stages, including N1.
    """
    stage_names = ["W", "N1", "N2", "N3", "REM"][:int(num_stages)]
    feature_names = []
    for source_name in stage_names:
        for target_name in stage_names:
            feature_names.append(
                f"transition_probability_{source_name}_to_{target_name}"
            )
    feature_names.extend(f"stage_fraction_{name}" for name in stage_names)
    feature_names.extend(f"bout_count_{name}" for name in stage_names)
    feature_names.extend(f"mean_bout_minutes_{name}" for name in stage_names)
    feature_names.extend(f"max_bout_minutes_{name}" for name in stage_names)
    feature_names.extend([
        "total_transition_count",
        "awakening_count_after_sleep_onset",
        "wake_after_sleep_onset_fraction",
        "sleep_onset_fraction",
        "nrem_to_rem_transition_count",
    ])

    all_features = []
    labels = []
    subject_ids = []
    epoch_minutes = float(epoch_seconds) / 60.0
    for record in records:
        stages = np.asarray(record["stages"], dtype=np.int64).reshape(-1)
        valid = (stages >= 0) & (stages < int(num_stages))
        valid_count = int(valid.sum())
        if valid_count <= 0:
            raise ValueError(
                f"{record['subject_id']} has no valid stages for sleep architecture."
            )

        stage_counts = np.bincount(
            stages[valid], minlength=int(num_stages)
        ).astype(np.float64)
        stage_fractions = stage_counts / float(valid_count)

        transition_counts = np.zeros(
            (int(num_stages), int(num_stages)), dtype=np.float64
        )
        if len(stages) > 1:
            adjacent_valid = valid[:-1] & valid[1:]
            for source, target in zip(
                stages[:-1][adjacent_valid], stages[1:][adjacent_valid]
            ):
                transition_counts[int(source), int(target)] += 1.0
        outgoing = transition_counts.sum(axis=1, keepdims=True)
        transition_probabilities = transition_counts / np.clip(
            outgoing, 1.0, None
        )

        bout_lengths = [[] for _ in range(int(num_stages))]
        current_stage = None
        current_length = 0
        for stage in stages:
            stage = int(stage)
            if not (0 <= stage < int(num_stages)):
                if current_stage is not None:
                    bout_lengths[current_stage].append(current_length)
                current_stage = None
                current_length = 0
                continue
            if current_stage == stage:
                current_length += 1
            else:
                if current_stage is not None:
                    bout_lengths[current_stage].append(current_length)
                current_stage = stage
                current_length = 1
        if current_stage is not None:
            bout_lengths[current_stage].append(current_length)

        bout_counts = np.asarray(
            [len(lengths) for lengths in bout_lengths], dtype=np.float64
        )
        mean_bout_minutes = np.asarray([
            (float(np.mean(lengths)) * epoch_minutes if lengths else 0.0)
            for lengths in bout_lengths
        ], dtype=np.float64)
        max_bout_minutes = np.asarray([
            (float(np.max(lengths)) * epoch_minutes if lengths else 0.0)
            for lengths in bout_lengths
        ], dtype=np.float64)

        valid_sequence = stages[valid]
        sleep_rows = np.flatnonzero(valid_sequence != 0)
        if len(sleep_rows) > 0:
            first_sleep = int(sleep_rows[0])
            sleep_onset_fraction = first_sleep / max(len(valid_sequence), 1)
            post_onset = valid_sequence[first_sleep:]
            waso_fraction = float(np.mean(post_onset == 0))
        else:
            sleep_onset_fraction = 1.0
            waso_fraction = 0.0

        awakening_count = float(transition_counts[1:, 0].sum())
        nrem_to_rem_count = (
            float(transition_counts[1:4, 4].sum())
            if int(num_stages) >= 5 else 0.0
        )
        summary_features = np.asarray([
            float(transition_counts.sum()),
            awakening_count,
            waso_fraction,
            float(sleep_onset_fraction),
            nrem_to_rem_count,
        ], dtype=np.float64)
        subject_features = np.concatenate([
            transition_probabilities.reshape(-1),
            stage_fractions,
            bout_counts,
            mean_bout_minutes,
            max_bout_minutes,
            summary_features,
        ]).astype(np.float32)
        all_features.append(np.nan_to_num(subject_features, copy=False))
        labels.append(int(record["sleep_disorder_label"]))
        subject_ids.append(str(record["subject_id"]))

    if not all_features:
        return (
            np.empty((0, len(feature_names)), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            [],
            feature_names,
        )
    features = np.stack(all_features).astype(np.float32)
    if features.shape[1] != len(feature_names):
        raise RuntimeError(
            "Sleep-architecture feature names do not match feature width: "
            f"{len(feature_names)} vs {features.shape[1]}."
        )
    return (
        features,
        np.asarray(labels, dtype=np.int64),
        subject_ids,
        feature_names,
    )


def _build_disorder_architecture_logistic_pipeline(cfg, c_value, random_state):
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=int(random_state),
            ),
        ),
    ])


def _aligned_disorder_architecture_probabilities(
    model, features, num_classes=3
):
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(
        model.named_steps["logistic"].classes_, dtype=np.int64
    )
    aligned = np.zeros((len(features), int(num_classes)), dtype=np.float64)
    for source_index, class_index in enumerate(classes):
        if 0 <= int(class_index) < int(num_classes):
            aligned[:, int(class_index)] = raw[:, source_index]
    return aligned / np.clip(aligned.sum(axis=1, keepdims=True), 1e-12, None)


def _fuse_stage_and_architecture_probabilities(
    stage_probabilities, architecture_probabilities, architecture_weight
):
    architecture_weight = float(architecture_weight)
    if not 0.0 <= architecture_weight <= 1.0:
        raise ValueError(
            f"Architecture fusion weight must be in [0, 1], got "
            f"{architecture_weight}."
        )
    stage_probabilities = np.asarray(stage_probabilities, dtype=np.float64)
    architecture_probabilities = np.asarray(
        architecture_probabilities, dtype=np.float64
    )
    if stage_probabilities.shape != architecture_probabilities.shape:
        raise ValueError(
            "Stage and sleep-architecture probability shapes differ: "
            f"{stage_probabilities.shape} vs {architecture_probabilities.shape}."
        )
    fused = (
        (1.0 - architecture_weight) * stage_probabilities
        + architecture_weight * architecture_probabilities
    )
    return fused / np.clip(fused.sum(axis=1, keepdims=True), 1e-12, None)


def _apply_disorder_decision_multipliers(probabilities, multipliers):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    multipliers = np.asarray(multipliers, dtype=np.float64).reshape(1, -1)
    if probabilities.ndim != 2 or probabilities.shape[1] != multipliers.shape[1]:
        raise ValueError(
            "Decision multiplier width does not match disorder probabilities."
        )
    if np.any(multipliers <= 0.0):
        raise ValueError("Decision multipliers must all be positive.")
    adjusted = probabilities * multipliers
    return adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), 1e-12, None)


def _select_disorder_decision_multipliers(
    labels, probabilities, cfg, class_names
):
    """Choose HC/RBD argmax calibration using inner OOF predictions only."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if list(class_names) != ["HC", "NFLE", "RBD"]:
        raise ValueError(
            "Decision calibration expects class order ['HC', 'NFLE', 'RBD']."
        )
    history = []
    best_entry = None
    best_key = None
    for hc_multiplier in cfg.disorder_decision_hc_multipliers:
        for rbd_multiplier in cfg.disorder_decision_rbd_multipliers:
            multipliers = [float(hc_multiplier), 1.0, float(rbd_multiplier)]
            adjusted = _apply_disorder_decision_multipliers(
                probabilities, multipliers
            )
            predictions = np.argmax(adjusted, axis=1)
            acc = float(accuracy_score(labels, predictions))
            balanced_acc = float(
                balanced_accuracy_score(labels, predictions)
            )
            macro_f1 = float(
                f1_score(
                    labels, predictions, average="macro", zero_division=0
                )
            )
            selection_score = 0.5 * acc + 0.5 * balanced_acc
            entry = {
                "multipliers": multipliers,
                "hc_multiplier": float(hc_multiplier),
                "nfle_multiplier": 1.0,
                "rbd_multiplier": float(rbd_multiplier),
                "inner_oof_selection_score": float(selection_score),
                "inner_oof_acc": acc,
                "inner_oof_balanced_acc": balanced_acc,
                "inner_oof_macro_f1": macro_f1,
            }
            history.append(entry)
            distance_from_identity = (
                abs(float(hc_multiplier) - 1.0)
                + abs(float(rbd_multiplier) - 1.0)
            )
            selection_key = (
                selection_score,
                macro_f1,
                balanced_acc,
                acc,
                -distance_from_identity,
            )
            if best_key is None or selection_key > best_key:
                best_key = selection_key
                best_entry = entry
    if best_entry is None:
        raise RuntimeError("Disorder decision calibration produced no candidate.")
    return best_entry, history


def _build_disorder_rf_pipeline(
    cfg, pca_components, max_depth, min_samples_leaf, random_state, n_samples
):
    effective_components = max(
        1,
        min(
            int(pca_components),
            int(n_samples) - 1,
        ),
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(
                n_components=effective_components,
                svd_solver="randomized",
                random_state=int(random_state),
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=int(cfg.disorder_rf_n_estimators),
                max_depth=max_depth,
                min_samples_leaf=int(min_samples_leaf),
                max_features=cfg.disorder_rf_max_features,
                class_weight=cfg.disorder_rf_class_weight,
                random_state=int(random_state),
                n_jobs=int(cfg.disorder_rf_n_jobs),
            ),
        ),
    ])


def _aligned_disorder_rf_probabilities(model, features, num_classes=3):
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.named_steps["rf"].classes_, dtype=np.int64)
    aligned = np.zeros((len(features), int(num_classes)), dtype=np.float64)
    for source_index, class_index in enumerate(classes):
        if 0 <= int(class_index) < int(num_classes):
            aligned[:, int(class_index)] = raw[:, source_index]
    aligned = aligned / np.clip(aligned.sum(axis=1, keepdims=True), 1e-12, None)
    return aligned


def _fit_disorder_stage_rf_models(
    stage_features,
    stage_available,
    labels,
    cfg,
    pca_components,
    max_depth,
    min_samples_leaf,
    base_random_state,
    num_classes=3,
):
    """Fit one multiclass RF per available stage with shared settings."""
    stage_features = np.asarray(stage_features, dtype=np.float32)
    stage_available = np.asarray(stage_available, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    if stage_features.ndim != 3:
        raise ValueError(
            f"Stage RF features must be [subjects, stages, features], got "
            f"{stage_features.shape}."
        )
    if stage_available.shape != stage_features.shape[:2]:
        raise ValueError("Stage RF availability mask shape does not match features.")

    models = []
    training_counts = []
    for stage_index in range(stage_features.shape[1]):
        stage_rows = stage_available[:, stage_index]
        stage_labels = labels[stage_rows]
        class_counts = np.bincount(
            stage_labels, minlength=int(num_classes)
        ).astype(np.int64)
        training_counts.append(class_counts.tolist())
        # A stage model must remain a genuine three-class classifier. If an
        # inner-training split has no example from one class, that stage is
        # omitted from that split's probability average instead of fitting a
        # degenerate model or injecting a zero-vector subject.
        if np.any(class_counts == 0):
            models.append(None)
            continue
        stage_seed = int(base_random_state + (stage_index + 1) * 10000000)
        model = _build_disorder_rf_pipeline(
            cfg,
            pca_components,
            max_depth,
            min_samples_leaf,
            stage_seed,
            int(stage_rows.sum()),
        )
        model.fit(stage_features[stage_rows, stage_index], stage_labels)
        models.append(model)
    if not any(model is not None for model in models):
        raise RuntimeError("No complete three-class sleep-stage RF could be fitted.")
    return models, training_counts


def _fuse_disorder_stage_rf_probabilities(
    models, stage_features, stage_available, num_classes=3
):
    """Uniformly average probabilities from models for truly present stages."""
    stage_features = np.asarray(stage_features, dtype=np.float32)
    stage_available = np.asarray(stage_available, dtype=bool)
    n_subjects, n_stages = stage_features.shape[:2]
    if len(models) != n_stages:
        raise ValueError("Number of stage RF models does not match stage features.")

    probability_sum = np.zeros(
        (n_subjects, int(num_classes)), dtype=np.float64
    )
    contributing_stage_count = np.zeros(n_subjects, dtype=np.int64)
    per_stage_probabilities = np.full(
        (n_subjects, n_stages, int(num_classes)), np.nan, dtype=np.float64
    )
    for stage_index, model in enumerate(models):
        if model is None:
            continue
        rows = stage_available[:, stage_index]
        if not np.any(rows):
            continue
        probabilities = _aligned_disorder_rf_probabilities(
            model,
            stage_features[rows, stage_index],
            num_classes=num_classes,
        )
        probability_sum[rows] += probabilities
        contributing_stage_count[rows] += 1
        per_stage_probabilities[rows, stage_index] = probabilities

    if np.any(contributing_stage_count == 0):
        missing_rows = np.flatnonzero(contributing_stage_count == 0).tolist()
        raise RuntimeError(
            "No fitted stage RF contributed to subjects at rows "
            f"{missing_rows}."
        )
    fused = probability_sum / contributing_stage_count[:, None]
    fused = fused / np.clip(fused.sum(axis=1, keepdims=True), 1e-12, None)
    return fused, per_stage_probabilities, contributing_stage_count


def _transform_disorder_stage_rf_embeddings(
    models, stage_features, stage_available
):
    """Concatenate per-stage PCA embeddings for optional fold visualization."""
    stage_features = np.asarray(stage_features, dtype=np.float32)
    stage_available = np.asarray(stage_available, dtype=bool)
    transformed_stages = []
    for stage_index, model in enumerate(models):
        if model is None:
            continue
        pca = model.named_steps["pca"]
        width = int(pca.n_components_)
        transformed = np.zeros((len(stage_features), width), dtype=np.float32)
        rows = stage_available[:, stage_index]
        if np.any(rows):
            scaled = model.named_steps["scaler"].transform(
                stage_features[rows, stage_index]
            )
            transformed[rows] = pca.transform(scaled).astype(np.float32)
        transformed_stages.append(transformed)
    if not transformed_stages:
        return np.empty((len(stage_features), 0), dtype=np.float32)
    return np.concatenate(
        transformed_stages + [stage_available.astype(np.float32)], axis=1
    )


def select_disorder_stage_fusion_rf_by_inner_cv(
    stage_features,
    stage_available,
    labels,
    cfg,
    run_seed,
    outer_fold_idx,
    class_names,
):
    """Select one shared setting from fused inner-OOF stage probabilities."""
    labels = np.asarray(labels, dtype=np.int64)
    min_class_count = int(np.bincount(labels, minlength=3).min())
    inner_splits = min(int(cfg.disorder_rf_inner_splits), min_class_count)
    if inner_splits < 2:
        raise ValueError("Not enough trainval subjects for nested RF CV.")
    inner_seed = int(run_seed * 100 + outer_fold_idx)
    inner_cv = list(StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=inner_seed,
    ).split(np.arange(len(labels)), labels))

    search_history = []
    best_entry = None
    best_key = None
    candidate_index = 0
    for pca_components in cfg.disorder_rf_pca_components:
        for max_depth in cfg.disorder_rf_max_depths:
            for min_samples_leaf in cfg.disorder_rf_min_samples_leaf:
                candidate_index += 1
                oof_probabilities = np.zeros(
                    (len(labels), len(class_names)), dtype=np.float64
                )
                covered = np.zeros(len(labels), dtype=bool)
                for inner_fold_idx, (inner_train, inner_validation) in enumerate(
                    inner_cv, 1
                ):
                    model_seed = int(
                        run_seed * 100000
                        + outer_fold_idx * 1000
                        + inner_fold_idx
                    )
                    models, _ = _fit_disorder_stage_rf_models(
                        stage_features[inner_train],
                        stage_available[inner_train],
                        labels[inner_train],
                        cfg,
                        pca_components,
                        max_depth,
                        min_samples_leaf,
                        model_seed,
                        num_classes=len(class_names),
                    )
                    fused_probabilities, _, _ = (
                        _fuse_disorder_stage_rf_probabilities(
                            models,
                            stage_features[inner_validation],
                            stage_available[inner_validation],
                            num_classes=len(class_names),
                        )
                    )
                    oof_probabilities[inner_validation] = fused_probabilities
                    covered[inner_validation] = True
                if not np.all(covered):
                    raise RuntimeError(
                        "Nested stage-fusion RF CV left uncovered trainval rows."
                    )

                metrics = _disorder_metrics_from_probabilities(
                    labels,
                    oof_probabilities,
                    class_names,
                )
                entry = {
                    "candidate_index": int(candidate_index),
                    "pca_components": int(pca_components),
                    "max_depth": (
                        None if max_depth is None else int(max_depth)
                    ),
                    "min_samples_leaf": int(min_samples_leaf),
                    "inner_splits": int(inner_splits),
                    "inner_pooled_macro_ovr_auroc": float(
                        metrics["macro_ovr_auroc"]
                    ),
                    "inner_pooled_macro_f1": float(metrics["macro_f1"]),
                    "inner_pooled_balanced_acc": float(
                        metrics["balanced_acc"]
                    ),
                    "inner_pooled_acc": float(metrics["acc"]),
                }
                search_history.append(entry)
                depth_complexity = 100000 if max_depth is None else int(max_depth)
                selection_key = (
                    entry["inner_pooled_macro_ovr_auroc"],
                    entry["inner_pooled_macro_f1"],
                    entry["inner_pooled_balanced_acc"],
                    -int(pca_components),
                    -depth_complexity,
                    int(min_samples_leaf),
                )
                if best_key is None or selection_key > best_key:
                    best_key = selection_key
                    best_entry = entry
                print(
                    "[Disorder Nested Stage-Fusion RF Candidate] "
                    f"outer_fold={outer_fold_idx} | "
                    f"PCA={pca_components} | depth={max_depth} | "
                    f"min_leaf={min_samples_leaf} | "
                    f"fused inner pooled macro_OVR_AUC="
                    f"{metrics['macro_ovr_auroc']:.4f} | "
                    f"macro_F1={metrics['macro_f1']:.4f}"
                )

    if best_entry is None:
        raise RuntimeError("Nested stage-fusion RF search produced no candidate.")
    return best_entry, search_history


def select_disorder_stage_architecture_fusion_by_inner_cv(
    stage_features,
    stage_available,
    architecture_features,
    labels,
    cfg,
    run_seed,
    outer_fold_idx,
    class_names,
):
    """Select stage RF, architecture regularization, and fusion in trainval."""
    labels = np.asarray(labels, dtype=np.int64)
    architecture_features = np.asarray(
        architecture_features, dtype=np.float32
    )
    if len(architecture_features) != len(labels):
        raise ValueError(
            "Sleep-architecture and label row counts differ in inner CV."
        )
    min_class_count = int(np.bincount(labels, minlength=3).min())
    inner_splits = min(int(cfg.disorder_rf_inner_splits), min_class_count)
    if inner_splits < 2:
        raise ValueError("Not enough trainval subjects for nested fusion CV.")
    inner_seed = int(run_seed * 100 + outer_fold_idx)
    inner_cv = list(StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=inner_seed,
    ).split(np.arange(len(labels)), labels))

    architecture_oof_by_c = {}
    architecture_metrics_by_c = {}
    architecture_search_history = []
    for c_index, c_value in enumerate(
        cfg.disorder_architecture_logistic_cs, 1
    ):
        oof_probabilities = np.zeros(
            (len(labels), len(class_names)), dtype=np.float64
        )
        covered = np.zeros(len(labels), dtype=bool)
        for inner_fold_idx, (inner_train, inner_validation) in enumerate(
            inner_cv, 1
        ):
            architecture_seed = int(
                run_seed * 1000000
                + outer_fold_idx * 10000
                + c_index * 100
                + inner_fold_idx
            )
            model = _build_disorder_architecture_logistic_pipeline(
                cfg, c_value, architecture_seed
            )
            model.fit(
                architecture_features[inner_train], labels[inner_train]
            )
            oof_probabilities[inner_validation] = (
                _aligned_disorder_architecture_probabilities(
                    model,
                    architecture_features[inner_validation],
                    num_classes=len(class_names),
                )
            )
            covered[inner_validation] = True
        if not np.all(covered):
            raise RuntimeError(
                "Nested architecture CV left uncovered trainval rows."
            )
        c_key = float(c_value)
        architecture_oof_by_c[c_key] = oof_probabilities
        architecture_metrics_by_c[c_key] = (
            _disorder_metrics_from_probabilities(
                labels, oof_probabilities, class_names
            )
        )
        architecture_search_history.append({
            "architecture_logistic_c": c_key,
            "inner_pooled_macro_ovr_auroc": float(
                architecture_metrics_by_c[c_key]["macro_ovr_auroc"]
            ),
            "inner_pooled_macro_f1": float(
                architecture_metrics_by_c[c_key]["macro_f1"]
            ),
            "inner_pooled_balanced_acc": float(
                architecture_metrics_by_c[c_key]["balanced_acc"]
            ),
        })

    selected_architecture = max(
        architecture_search_history,
        key=lambda entry: (
            entry["inner_pooled_macro_ovr_auroc"],
            entry["inner_pooled_macro_f1"],
            entry["inner_pooled_balanced_acc"],
            -entry["architecture_logistic_c"],
        ),
    )
    selected_architecture_c = float(
        selected_architecture["architecture_logistic_c"]
    )
    selected_architecture_oof = architecture_oof_by_c[
        selected_architecture_c
    ]
    selected_architecture_metrics = architecture_metrics_by_c[
        selected_architecture_c
    ]
    print(
        "[Disorder Nested Architecture Selected] "
        f"outer_fold={outer_fold_idx} | C={selected_architecture_c:.2g} | "
        f"architecture-only inner macro_OVR_AUC="
        f"{selected_architecture_metrics['macro_ovr_auroc']:.4f}"
    )

    search_history = []
    best_entry = None
    best_key = None
    candidate_index = 0
    candidate_oof_probabilities = {}
    for pca_components in cfg.disorder_rf_pca_components:
        for max_depth in cfg.disorder_rf_max_depths:
            for min_samples_leaf in cfg.disorder_rf_min_samples_leaf:
                stage_oof_probabilities = np.zeros(
                    (len(labels), len(class_names)), dtype=np.float64
                )
                covered = np.zeros(len(labels), dtype=bool)
                for inner_fold_idx, (inner_train, inner_validation) in enumerate(
                    inner_cv, 1
                ):
                    model_seed = int(
                        run_seed * 100000
                        + outer_fold_idx * 1000
                        + inner_fold_idx
                    )
                    models, _ = _fit_disorder_stage_rf_models(
                        stage_features[inner_train],
                        stage_available[inner_train],
                        labels[inner_train],
                        cfg,
                        pca_components,
                        max_depth,
                        min_samples_leaf,
                        model_seed,
                        num_classes=len(class_names),
                    )
                    stage_oof_probabilities[inner_validation], _, _ = (
                        _fuse_disorder_stage_rf_probabilities(
                            models,
                            stage_features[inner_validation],
                            stage_available[inner_validation],
                            num_classes=len(class_names),
                        )
                    )
                    covered[inner_validation] = True
                if not np.all(covered):
                    raise RuntimeError(
                        "Nested stage RF CV left uncovered trainval rows."
                    )
                stage_metrics = _disorder_metrics_from_probabilities(
                    labels, stage_oof_probabilities, class_names
                )

                local_best_entry = None
                local_best_key = None
                c_value = selected_architecture_c
                architecture_oof_probabilities = selected_architecture_oof
                architecture_metrics = selected_architecture_metrics
                for architecture_weight in (
                    cfg.disorder_architecture_fusion_weights
                ):
                    candidate_index += 1
                    fused_probabilities = (
                        _fuse_stage_and_architecture_probabilities(
                            stage_oof_probabilities,
                            architecture_oof_probabilities,
                            architecture_weight,
                        )
                    )
                    metrics = _disorder_metrics_from_probabilities(
                        labels, fused_probabilities, class_names
                    )
                    entry = {
                        "candidate_index": int(candidate_index),
                        "pca_components": int(pca_components),
                        "max_depth": (
                            None if max_depth is None else int(max_depth)
                        ),
                        "min_samples_leaf": int(min_samples_leaf),
                        "architecture_logistic_c": float(c_value),
                        "architecture_fusion_weight": float(
                            architecture_weight
                        ),
                        "stage_fusion_weight": float(
                            1.0 - float(architecture_weight)
                        ),
                        "inner_splits": int(inner_splits),
                        "inner_stage_only_macro_ovr_auroc": float(
                            stage_metrics["macro_ovr_auroc"]
                        ),
                        "inner_architecture_only_macro_ovr_auroc": float(
                            architecture_metrics["macro_ovr_auroc"]
                        ),
                        "inner_pooled_macro_ovr_auroc": float(
                            metrics["macro_ovr_auroc"]
                        ),
                        "inner_pooled_macro_f1": float(metrics["macro_f1"]),
                        "inner_pooled_balanced_acc": float(
                            metrics["balanced_acc"]
                        ),
                        "inner_pooled_acc": float(metrics["acc"]),
                    }
                    search_history.append(entry)
                    candidate_oof_probabilities[int(candidate_index)] = (
                        fused_probabilities.copy()
                    )
                    depth_complexity = (
                        100000 if max_depth is None else int(max_depth)
                    )
                    selection_key = (
                        entry["inner_pooled_macro_ovr_auroc"],
                        entry["inner_pooled_macro_f1"],
                        entry["inner_pooled_balanced_acc"],
                        -float(architecture_weight),
                        -int(pca_components),
                        -depth_complexity,
                        int(min_samples_leaf),
                        -float(c_value),
                    )
                    if local_best_key is None or selection_key > local_best_key:
                        local_best_key = selection_key
                        local_best_entry = entry
                    if best_key is None or selection_key > best_key:
                        best_key = selection_key
                        best_entry = entry

                print(
                    "[Disorder Nested Stage+Architecture Candidate] "
                    f"outer_fold={outer_fold_idx} | PCA={pca_components} | "
                    f"depth={max_depth} | min_leaf={min_samples_leaf} | "
                    f"best_C={local_best_entry['architecture_logistic_c']:.2g} | "
                    f"architecture_weight="
                    f"{local_best_entry['architecture_fusion_weight']:.2f} | "
                    f"inner macro_OVR_AUC="
                    f"{local_best_entry['inner_pooled_macro_ovr_auroc']:.4f}"
                )

    if best_entry is None:
        raise RuntimeError(
            "Nested stage-plus-architecture search produced no candidate."
        )
    maximum_inner_auc = max(
        entry["inner_pooled_macro_ovr_auroc"] for entry in search_history
    )
    auc_tolerance = float(
        cfg.disorder_architecture_fusion_auc_tolerance
    )
    eligible_entries = [
        entry for entry in search_history
        if entry["inner_pooled_macro_ovr_auroc"]
        >= maximum_inner_auc - auc_tolerance
    ]
    conservative_entry = max(
        eligible_entries,
        key=lambda entry: (
            -entry["architecture_fusion_weight"],
            entry["inner_pooled_macro_ovr_auroc"],
            entry["inner_pooled_macro_f1"],
            entry["inner_pooled_balanced_acc"],
            -entry["pca_components"],
            -(100000 if entry["max_depth"] is None else entry["max_depth"]),
            entry["min_samples_leaf"],
        ),
    )
    best_entry = dict(conservative_entry)
    best_entry["maximum_inner_pooled_macro_ovr_auroc"] = float(
        maximum_inner_auc
    )
    best_entry["architecture_fusion_auc_tolerance"] = auc_tolerance
    best_entry["conservative_smallest_weight_selection"] = True
    selected_oof_probabilities = candidate_oof_probabilities[
        int(best_entry["candidate_index"])
    ]
    decision_calibration, decision_calibration_history = (
        _select_disorder_decision_multipliers(
            labels, selected_oof_probabilities, cfg, class_names
        )
    )
    best_entry["decision_calibration"] = decision_calibration
    best_entry["decision_calibration_history"] = (
        decision_calibration_history
    )
    best_entry["architecture_logistic_search"] = architecture_search_history
    return best_entry, search_history


def _disorder_metrics_from_arrays(
    labels, logits, class_names, subject_ids=None
):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    predictions = np.argmax(logits, axis=1)
    probabilities = np.stack([_softmax_np(row) for row in logits], axis=0)
    num_classes = len(class_names)
    try:
        auroc = roc_auc_score(
            labels,
            probabilities,
            labels=list(range(num_classes)),
            multi_class="ovr",
            average="macro",
        )
        if np.isnan(auroc):
            auroc = 0.0
    except Exception:
        auroc = 0.0
    report = classification_report(
        labels,
        predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        labels, predictions, labels=list(range(num_classes))
    )
    cm_normalized = cm.astype(np.float64) / np.clip(
        cm.sum(axis=1, keepdims=True), 1, None
    )
    per_class_metrics = {}
    for class_index, name in enumerate(class_names):
        binary_true = labels == class_index
        tp = int(cm[class_index, class_index])
        fn = int(cm[class_index, :].sum() - tp)
        fp = int(cm[:, class_index].sum() - tp)
        tn = int(len(labels) - tp - fn - fp)
        specificity = tn / max(tn + fp, 1)
        per_class_metrics[name] = {
            # Per-class ACC is explicitly one-vs-rest, matching the clinical
            # interpretation used alongside class-specific ROC-AUC.
            "ovr_acc": float((tp + tn) / max(len(labels), 1)),
            "auroc": _safe_binary_auroc(
                binary_true.astype(np.int64).tolist(),
                probabilities[:, class_index].tolist(),
            ),
            "precision": float(report.get(name, {}).get("precision", 0.0)),
            "recall": float(report.get(name, {}).get("recall", 0.0)),
            "sensitivity": float(report.get(name, {}).get("recall", 0.0)),
            "specificity": float(specificity),
            "f1": float(report.get(name, {}).get("f1-score", 0.0)),
            "support": int(report.get(name, {}).get("support", 0)),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }
    macro_report = report.get("macro avg", {})
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "kappa": _safe_kappa(labels.tolist(), predictions.tolist()),
        "balanced_acc": _safe_balanced_accuracy(
            labels.tolist(), predictions.tolist()
        ),
        "auroc": float(auroc),
        "macro_ovr_auroc": float(auroc),
        "macro_precision": float(macro_report.get("precision", 0.0)),
        "macro_recall": float(macro_report.get("recall", 0.0)),
        "macro_f1": float(f1_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            labels, predictions, average="weighted", zero_division=0
        )),
        "per_class_metrics": per_class_metrics,
        "per_class_f1": {
            name: values["f1"] for name, values in per_class_metrics.items()
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": cm_normalized.tolist(),
        "report": report,
        "n_subjects": int(len(labels)),
        "subject_ids": list(subject_ids or []),
        "y_true": labels.tolist(),
        "y_pred": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }


def _disorder_metrics_from_probabilities(
    labels,
    probabilities,
    class_names,
    subject_ids=None,
    decision_multipliers=None,
    hard_predictions=None,
):
    """Evaluate probabilities with optional externally supplied predictions."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError(
            "Disorder probabilities must have shape "
            f"[subjects, {len(class_names)}], got {probabilities.shape}."
        )
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / np.clip(row_sums, 1e-12, None)
    # _disorder_metrics_from_arrays applies softmax. log(probability) maps back
    # to the same normalized probabilities and keeps one metric implementation.
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    ranking_metrics = _disorder_metrics_from_arrays(
        labels, logits, class_names, subject_ids=subject_ids
    )
    if decision_multipliers is not None and hard_predictions is not None:
        raise ValueError(
            "Provide either decision_multipliers or hard_predictions, not both."
        )
    if decision_multipliers is None and hard_predictions is None:
        return ranking_metrics

    if hard_predictions is None:
        decision_probabilities = _apply_disorder_decision_multipliers(
            probabilities, decision_multipliers
        )
    else:
        hard_predictions = np.asarray(hard_predictions, dtype=np.int64)
        if hard_predictions.shape != (len(probabilities),):
            raise ValueError(
                "hard_predictions must contain one class index per subject; "
                f"got {hard_predictions.shape} for {len(probabilities)} subjects."
            )
        if np.any(hard_predictions < 0) or np.any(
            hard_predictions >= len(class_names)
        ):
            raise ValueError("hard_predictions contains an invalid class index.")
        decision_probabilities = np.eye(
            len(class_names), dtype=np.float64
        )[hard_predictions]
    decision_logits = np.log(
        np.clip(decision_probabilities, 1e-12, 1.0)
    )
    decision_metrics = _disorder_metrics_from_arrays(
        labels, decision_logits, class_names, subject_ids=subject_ids
    )
    # Classification metrics and the confusion matrix use the supplied fold
    # predictions. ROC-AUC continues to use the original model probabilities.
    decision_metrics["auroc"] = ranking_metrics["auroc"]
    decision_metrics["macro_ovr_auroc"] = ranking_metrics[
        "macro_ovr_auroc"
    ]
    for class_name in class_names:
        decision_metrics["per_class_metrics"][class_name]["auroc"] = (
            ranking_metrics["per_class_metrics"][class_name]["auroc"]
        )
    decision_metrics["probabilities"] = probabilities.tolist()
    decision_metrics["probability_argmax_y_pred"] = ranking_metrics["y_pred"]
    if hard_predictions is None:
        decision_metrics["decision_probabilities"] = (
            decision_probabilities.tolist()
        )
        decision_metrics["decision_multipliers"] = [
            float(value) for value in decision_multipliers
        ]
    else:
        decision_metrics["hard_predictions_supplied"] = True
    return decision_metrics


def evaluate_stage_aware_disorder_head(model, records, cfg, device):
    loader = DataLoader(
        DisorderSubjectFeatureDataset(records),
        batch_size=max(1, int(cfg.disorder_subject_batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_disorder_subject_features,
    )
    model.eval()
    all_logits = []
    all_labels = []
    all_subject_ids = []
    all_embeddings = []
    all_stage_weights = []
    with torch.inference_mode():
        for batch in loader:
            logits, embeddings, stage_weights = model(
                batch["features"].to(device),
                batch["stages"].to(device),
                batch["valid_mask"].to(device),
                batch["lengths"],
                return_embedding=True,
            )
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(batch["labels"].numpy().tolist())
            all_subject_ids.extend(batch["subject_id"])
            all_embeddings.append(embeddings.cpu().numpy())
            all_stage_weights.append(stage_weights.cpu().numpy())
    metrics = _disorder_metrics_from_arrays(
        all_labels,
        np.concatenate(all_logits, axis=0),
        cfg.tasks["sleep_disorder_classification"]["class_names"],
        subject_ids=all_subject_ids,
    )
    metrics["subject_embeddings"] = np.concatenate(
        all_embeddings, axis=0
    ).tolist()
    metrics["stage_attention_weights"] = np.concatenate(
        all_stage_weights, axis=0
    ).tolist()
    return metrics


def _disorder_checkpoint_is_better(current, best, cfg):
    if best is None:
        return True
    names = [str(cfg.disorder_selection_metric)] + list(
        cfg.disorder_selection_tiebreakers
    )
    for index, name in enumerate(names):
        current_value = float(current.get(name, 0.0))
        best_value = float(best.get(name, 0.0))
        delta = float(cfg.disorder_min_delta) if index == 0 else 1e-12
        if current_value > best_value + delta:
            return True
        if current_value < best_value - delta:
            return False
    return False


def _disorder_class_weights(records, num_classes, device):
    counts = np.bincount(
        [int(record["sleep_disorder_label"]) for record in records],
        minlength=num_classes,
    ).astype(np.float64)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = weights / np.mean(weights)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def disorder_subject_cv_run(records, cfg, device):
    records = [r for r in records if int(r["sleep_disorder_label"]) in [0, 1]]
    if len(records) == 0:
        print("sleep_disorder_classification: no valid subject-level labels, skip.")
        return None

    subject_ids = np.array([str(r["subject_id"]) for r in records])
    y = np.array([int(r["sleep_disorder_label"]) for r in records], dtype=np.int64)

    unique_labels, counts = np.unique(y, return_counts=True)
    if len(unique_labels) < 2:
        print("sleep_disorder_classification: need both normal and disorder subjects.")
        return None

    max_splits = int(counts.min())
    n_splits = min(cfg.n_splits, max_splits)
    if n_splits < 2:
        print("sleep_disorder_classification: not enough subjects per class for StratifiedKFold.")
        return None

    task_name = "sleep_disorder_classification"
    task_out_dir = task_output_dir(cfg, task_name)
    ensure_dir(task_out_dir)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    fold_results = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(subject_ids, y), 1):
        trainval_records = [records[i] for i in trainval_idx]
        test_records = [records[i] for i in test_idx]
        trainval_y = y[trainval_idx]

        train_records, val_records = _split_disorder_train_val_records(
            trainval_records, trainval_y, cfg, fold_idx
        )

        train_samples = build_disorder_samples(train_records, seq_len=cfg.seq_len, stride=cfg.stride)
        val_samples = build_disorder_samples(val_records, seq_len=cfg.seq_len, stride=cfg.stride)
        test_samples = build_disorder_samples(test_records, seq_len=cfg.seq_len, stride=cfg.stride)

        fold_dir = os.path.join(task_out_dir, f"fold_{fold_idx}")
        val_dir = os.path.join(fold_dir, "inner_val")
        test_dir = os.path.join(fold_dir, "outer_test")
        ensure_dir(fold_dir)
        ensure_dir(val_dir)
        ensure_dir(test_dir)

        train_subjects = sorted({str(r["subject_id"]) for r in train_records})
        val_subjects = sorted({str(r["subject_id"]) for r in val_records})
        test_subjects = sorted({str(r["subject_id"]) for r in test_records})

        print(f"\n{'=' * 80}")
        print(f"Task: {task_name} | Subject-level Stratified Fold {fold_idx}/{n_splits}")
        print(f"Train subjects: {train_subjects}")
        print(f"Val subjects: {val_subjects}")
        print(f"Test subjects: {test_subjects}")
        print(
            f"Train windows: {len(train_samples)} | "
            f"Val windows: {len(val_samples)} | "
            f"Test windows: {len(test_samples)}"
        )
        print(f"{'=' * 80}")

        best_state = run_one_fold(
            task_name,
            fold_idx - 1,
            train_samples,
            val_samples,
            cfg,
            device,
            val_dir,
        )

        model_path = os.path.join(fold_dir, "best_model.pth")
        torch.save(best_state["model"], model_path)

        test_metrics = evaluate_disorder_from_state_dict(
            best_state["model"],
            test_samples,
            cfg,
            device,
            save_dir=test_dir,
            seed=cfg.seed + fold_idx,
        )

        print(
            f"[Test] Fold {fold_idx} | "
            f"acc={test_metrics['acc']:.4f}, "
            f"kappa={test_metrics['kappa']:.4f}, "
            f"balanced_acc={test_metrics.get('balanced_acc', 0.0):.4f}, "
            f"auroc={test_metrics.get('auroc', 0.0):.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )
        print_test_metrics_detail(test_metrics)

        with open(os.path.join(fold_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        result_json = {
            "task_name": task_name,
            **experiment_metadata(cfg),
            "fold_idx": fold_idx,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
            "val_metrics": best_state["metrics"],
            "test_metrics": test_metrics,
            "history": best_state["history"],
            "best_epoch": best_state.get("best_epoch", None),
        }
        with open(os.path.join(fold_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        fold_results.append(result_json)

    summary = {
        "task_name": task_name,
        **experiment_metadata(cfg),
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }

    if len(fold_results) > 0:
        metric_names = [
            "acc", "kappa", "balanced_acc", "auroc", "macro_f1", "weighted_f1"
        ]
        for metric_name in metric_names:
            metric_values = [
                fr["test_metrics"].get(metric_name, 0.0) for fr in fold_results
            ]
            summary[f"mean_{metric_name}"] = float(np.mean(metric_values))
            summary[f"std_{metric_name}"] = float(np.std(metric_values))

    with open(os.path.join(task_out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{task_name} completed.")
    if "mean_acc" in summary:
        print(
            f"Mean acc={summary['mean_acc']:.4f}+-{summary['std_acc']:.4f}, "
            f"mean kappa={summary['mean_kappa']:.4f}+-{summary['std_kappa']:.4f}, "
            f"mean balanced_acc={summary['mean_balanced_acc']:.4f}+-{summary['std_balanced_acc']:.4f}, "
            f"mean auroc={summary['mean_auroc']:.4f}+-{summary['std_auroc']:.4f}, "
            f"mean macro_f1={summary['mean_macro_f1']:.4f}+-{summary['std_macro_f1']:.4f}"
        )

    return summary


def disorder_subject_cv_run_model_b(
    records,
    cfg,
    device,
    _feature_records=None,
    _repeat_seed=None,
    _repeat_out_dir=None,
):
    """CAP nested-CV stage-RF plus sleep-architecture probability fusion."""
    records = [
        record for record in records
        if int(record["sleep_disorder_label"]) in {0, 1, 2}
    ]
    if not records:
        print(
            "sleep_disorder_classification: no CAP HC/NFLE/RBD labels; "
            "run the CAP preprocessing script first."
        )
        return None

    class_names = cfg.tasks["sleep_disorder_classification"]["class_names"]
    raw_labels = np.asarray(
        [int(record["sleep_disorder_label"]) for record in records],
        dtype=np.int64,
    )
    unique_labels, counts = np.unique(raw_labels, return_counts=True)
    if set(unique_labels.tolist()) != {0, 1, 2}:
        raise ValueError(
            "Disorder Stage+Architecture Fusion requires all three classes "
            "HC(0), NFLE(1), RBD(2); "
            f"found labels={unique_labels.tolist()}."
        )
    if _feature_records is None:
        print(
            "[Disorder Nested Stage+Architecture Fusion] raw subjects="
            + ", ".join(
                f"{class_names[int(label)]}:{int(count)}"
                for label, count in zip(unique_labels, counts)
            )
        )
        print(
            "[Disorder Nested Stage+Architecture Fusion] frozen event-detached backbone "
            "-> native 20-epoch context -> unique full-night epoch sequence -> "
            "W/N1/N2/N3/REM mean/std/fractions -> five multiclass RFs; "
            "ordered expert-stage sequence -> compact architecture logistic model"
        )
        print(
            "[Disorder Nested Stage+Architecture Fusion] N1 retained | fixed STFT "
            "disabled | nested inner CV selects RF settings, logistic C, and "
            "fusion weight | complete trainval refit | outer test once"
        )
        feature_records = extract_full_night_disorder_features(
            records, cfg, device
        )
        if len(feature_records) != len(records):
            raise RuntimeError(
                f"Expected {len(records)} CAP feature records, "
                f"got {len(feature_records)}."
            )

        repeat_seeds = [
            int(seed) for seed in getattr(
                cfg, "disorder_repeat_seeds", [cfg.seed]
            )
        ]
        if not repeat_seeds:
            raise ValueError("disorder_repeat_seeds must contain at least one seed.")
        if len(set(repeat_seeds)) != len(repeat_seeds):
            raise ValueError(
                f"disorder_repeat_seeds contains duplicates: {repeat_seeds}"
            )

        task_name = "sleep_disorder_classification"
        task_out_root = os.path.join(
            task_output_dir(cfg, task_name),
            f"{int(cfg.disorder_n_splits)}fold",
        )
        ensure_dir(task_out_root)
        print(
            f"[Disorder Nested Stage+Architecture Fusion Repeated CV] "
            f"repeats={len(repeat_seeds)} | "
            f"seeds={repeat_seeds} | frozen features extracted once"
        )
        repeat_summaries = []
        for repeat_index, repeat_seed in enumerate(repeat_seeds, 1):
            repeat_out_dir = os.path.join(
                task_out_root, f"seed_{repeat_seed}"
            )
            ensure_dir(repeat_out_dir)
            print(
                f"\n{'#' * 80}\n"
                f"Disorder Nested Stage+Architecture Fusion Repeat "
                f"{repeat_index}/{len(repeat_seeds)} | "
                f"seed={repeat_seed}\n"
                f"output={repeat_out_dir}\n"
                f"{'#' * 80}"
            )
            repeat_summary = disorder_subject_cv_run_model_b(
                records,
                cfg,
                device,
                _feature_records=feature_records,
                _repeat_seed=repeat_seed,
                _repeat_out_dir=repeat_out_dir,
            )
            repeat_summary["repeat_index"] = int(repeat_index)
            repeat_summary["output_dir"] = os.path.abspath(repeat_out_dir)
            repeat_summaries.append(repeat_summary)

        return _summarize_disorder_repeats(
            repeat_summaries,
            repeat_seeds,
            class_names,
            task_out_root,
        )

    feature_records = list(_feature_records)
    run_seed = int(cfg.seed if _repeat_seed is None else _repeat_seed)

    y = np.asarray(
        [int(record["sleep_disorder_label"]) for record in feature_records],
        dtype=np.int64,
    )
    use_stage_std = bool(cfg.disorder_rf_use_stage_std)
    stage_rf_features, stage_available, rf_labels, rf_subject_ids = (
        build_disorder_stage_rf_tensors(
        feature_records,
        include_stage_std=use_stage_std,
        )
    )
    (
        architecture_features,
        architecture_labels,
        architecture_subject_ids,
        architecture_feature_names,
    ) = build_disorder_sleep_architecture_features(
        feature_records,
        epoch_seconds=float(cfg.disorder_architecture_epoch_seconds),
    )
    if not np.array_equal(rf_labels, y):
        raise RuntimeError(
            "Nested stage-fusion RF feature labels changed record order."
        )
    if not np.array_equal(architecture_labels, y):
        raise RuntimeError(
            "Sleep-architecture feature labels changed record order."
        )
    if architecture_subject_ids != rf_subject_ids:
        raise RuntimeError(
            "Sleep-architecture features changed the subject order."
        )
    min_class_count = int(np.bincount(y, minlength=3).min())
    n_splits = min(int(cfg.disorder_n_splits), min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough subjects per CAP class for stratified CV.")

    task_name = "sleep_disorder_classification"
    task_out_dir = str(
        _repeat_out_dir or task_output_dir(cfg, task_name)
    )
    ensure_dir(task_out_dir)
    split_manifest = {
        "task_name": task_name,
        "repeat_seed": int(run_seed),
        "model": "nested_cv_stage_rf_plus_sleep_architecture_logistic_fusion",
        "class_names": class_names,
        "class_counts": {
            class_names[index]: int(np.sum(y == index)) for index in range(3)
        },
        "n_splits": int(n_splits),
        "backbone_path": os.path.abspath(cfg.eeg_pretrained_model_path),
        "backbone_frozen": True,
        "fixed_stft_used": False,
        "feature_protocol": (
            "20_epoch_context_stride10_overlap_average_to_unique_epochs"
        ),
        "stage_protocol": "expert_RK_S3_S4_merged_to_N3",
        "subject_feature_protocol": (
            "five_separate_layernorm_stage_mean_population_std_vectors_"
            "each_appended_with_five_stage_fractions"
            if use_stage_std
            else "five_separate_layernorm_stage_mean_vectors_"
            "each_appended_with_five_stage_fractions"
        ),
        "stage_names": ["W", "N1", "N2", "N3", "REM"],
        "per_stage_feature_dim": int(stage_rf_features.shape[2]),
        "stage_available_counts": {
            name: int(stage_available[:, index].sum())
            for index, name in enumerate(["W", "N1", "N2", "N3", "REM"])
        },
        "sleep_architecture_protocol": (
            "ordered_expert_stage_transition_probabilities_stage_fractions_"
            "bout_counts_mean_and_max_bout_durations_plus_global_sleep_"
            "continuity_features"
        ),
        "sleep_architecture_feature_dim": int(architecture_features.shape[1]),
        "sleep_architecture_feature_names": architecture_feature_names,
        "classifier_protocol": (
            "five_multiclass_stage_rf_models_with_one_shared_setting_selected_"
            "jointly_with_class_balanced_multinomial_logistic_sleep_"
            "architecture_model_and_probability_weight_by_inner_oof_macro_auc_"
            "then_full_trainval_refit"
        ),
        "rf_search_space": {
            "inner_splits": int(cfg.disorder_rf_inner_splits),
            "pca_components": [
                int(value) for value in cfg.disorder_rf_pca_components
            ],
            "max_depth": [
                None if value is None else int(value)
                for value in cfg.disorder_rf_max_depths
            ],
            "min_samples_leaf": [
                int(value) for value in cfg.disorder_rf_min_samples_leaf
            ],
            "n_estimators": int(cfg.disorder_rf_n_estimators),
            "max_features": cfg.disorder_rf_max_features,
            "class_weight": cfg.disorder_rf_class_weight,
            "selection_metric": "fused_inner_pooled_macro_ovr_auroc",
            "fusion": "uniform_mean_over_present_stages",
        },
        "sleep_architecture_search_space": {
            "classifier": "standard_scaler_plus_class_balanced_logistic",
            "logistic_c": [
                float(value)
                for value in cfg.disorder_architecture_logistic_cs
            ],
            "architecture_fusion_weight": [
                float(value)
                for value in cfg.disorder_architecture_fusion_weights
            ],
            "maximum_auc_tolerance_for_smallest_weight": float(
                cfg.disorder_architecture_fusion_auc_tolerance
            ),
            "selection_metric": "joint_inner_pooled_macro_ovr_auroc",
        },
        "decision_calibration_search_space": {
            "hc_multiplier": [
                float(value)
                for value in cfg.disorder_decision_hc_multipliers
            ],
            "nfle_multiplier": [1.0],
            "rbd_multiplier": [
                float(value)
                for value in cfg.disorder_decision_rbd_multipliers
            ],
            "selection_metric": "0.5_inner_oof_acc_plus_0.5_balanced_acc",
            "affects_roc_auc": False,
        },
        "sleepfm_influence": (
            "frozen_pretrained_embeddings_plus_separate_stage_multiclass_rf_"
            "probability_fusion_with_"
            + ("mean_and_variability_pooling" if use_stage_std else "mean_pooling")
        ),
        "folds": [],
    }

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=run_seed
    )
    fold_results = []
    indices = np.arange(len(feature_records))

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        skf.split(indices, y), 1
    ):
        fold_dir = os.path.join(task_out_dir, f"fold_{fold_idx}")
        ensure_dir(fold_dir)
        fold_seed = int(run_seed + fold_idx)
        set_seed(fold_seed)

        trainval_subjects = [rf_subject_ids[index] for index in trainval_idx]
        test_subjects = [rf_subject_ids[index] for index in test_idx]
        print(f"\n{'=' * 80}")
        print(
            f"Disorder Nested Stage+Architecture Fusion | seed={run_seed} | "
            f"Fold {fold_idx}/{n_splits}"
        )
        print(
            f"Trainval subjects ({len(trainval_subjects)}): "
            f"{trainval_subjects}"
        )
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
        print(
            "Features: contextual epochs -> five separate stage mean"
            + ("/std" if use_stage_std else "")
            + "+fraction vectors -> five multiclass StandardScaler/PCA/RFs "
            "-> uniform stage probability; ordered stage sequence -> "
            "sleep-architecture logistic -> probability fusion"
        )
        print(f"{'=' * 80}")

        best_rf_settings, search_history = (
            select_disorder_stage_architecture_fusion_by_inner_cv(
                stage_rf_features[trainval_idx],
                stage_available[trainval_idx],
                architecture_features[trainval_idx],
                y[trainval_idx],
                cfg,
                run_seed,
                fold_idx,
                class_names,
            )
        )
        print(
            "[Disorder Nested Stage+Architecture Selected] "
            f"fold={fold_idx} | "
            f"PCA={best_rf_settings['pca_components']} | "
            f"depth={best_rf_settings['max_depth']} | "
            f"min_leaf={best_rf_settings['min_samples_leaf']} | "
            f"architecture_C="
            f"{best_rf_settings['architecture_logistic_c']:.2g} | "
            f"architecture_weight="
            f"{best_rf_settings['architecture_fusion_weight']:.2f} | "
            f"max_inner_AUC="
            f"{best_rf_settings['maximum_inner_pooled_macro_ovr_auroc']:.4f} | "
            f"inner pooled macro_OVR_AUC="
            f"{best_rf_settings['inner_pooled_macro_ovr_auroc']:.4f}"
        )
        decision_calibration = best_rf_settings["decision_calibration"]
        print(
            "[Disorder Nested Decision Calibration Selected] "
            f"fold={fold_idx} | HC/NFLE/RBD multipliers="
            f"{decision_calibration['multipliers']} | "
            f"inner score="
            f"{decision_calibration['inner_oof_selection_score']:.4f} | "
            f"inner ACC={decision_calibration['inner_oof_acc']:.4f} | "
            f"inner balanced_ACC="
            f"{decision_calibration['inner_oof_balanced_acc']:.4f}"
        )
        final_model_seed = int(run_seed * 100000 + fold_idx * 1000 + 999)
        stage_models, stage_training_counts = _fit_disorder_stage_rf_models(
            stage_rf_features[trainval_idx],
            stage_available[trainval_idx],
            y[trainval_idx],
            cfg,
            best_rf_settings["pca_components"],
            best_rf_settings["max_depth"],
            best_rf_settings["min_samples_leaf"],
            final_model_seed,
            num_classes=len(class_names),
        )
        (
            test_stage_fused_probabilities,
            test_stage_probabilities,
            test_contributing_stage_counts,
        ) = _fuse_disorder_stage_rf_probabilities(
            stage_models,
            stage_rf_features[test_idx],
            stage_available[test_idx],
            num_classes=len(class_names),
        )
        architecture_model_seed = int(final_model_seed + 70000000)
        architecture_model = _build_disorder_architecture_logistic_pipeline(
            cfg,
            best_rf_settings["architecture_logistic_c"],
            architecture_model_seed,
        )
        architecture_model.fit(
            architecture_features[trainval_idx], y[trainval_idx]
        )
        test_architecture_probabilities = (
            _aligned_disorder_architecture_probabilities(
                architecture_model,
                architecture_features[test_idx],
                num_classes=len(class_names),
            )
        )
        test_probabilities = _fuse_stage_and_architecture_probabilities(
            test_stage_fused_probabilities,
            test_architecture_probabilities,
            best_rf_settings["architecture_fusion_weight"],
        )
        test_uncalibrated_metrics = _disorder_metrics_from_probabilities(
            y[test_idx],
            test_probabilities,
            class_names,
            subject_ids=test_subjects,
        )
        test_metrics = _disorder_metrics_from_probabilities(
            y[test_idx],
            test_probabilities,
            class_names,
            subject_ids=test_subjects,
            decision_multipliers=decision_calibration["multipliers"],
        )
        test_stage_only_metrics = _disorder_metrics_from_probabilities(
            y[test_idx],
            test_stage_fused_probabilities,
            class_names,
            subject_ids=test_subjects,
        )
        test_architecture_only_metrics = _disorder_metrics_from_probabilities(
            y[test_idx],
            test_architecture_probabilities,
            class_names,
            subject_ids=test_subjects,
        )
        test_embeddings = _transform_disorder_stage_rf_embeddings(
            stage_models,
            stage_rf_features[test_idx],
            stage_available[test_idx],
        )
        stage_names = ["W", "N1", "N2", "N3", "REM"]
        stage_test_metrics = {}
        for stage_index, stage_name in enumerate(stage_names):
            rows = np.isfinite(test_stage_probabilities[:, stage_index, 0])
            if not np.any(rows):
                stage_test_metrics[stage_name] = None
                continue
            stage_subjects = [
                test_subjects[index] for index in np.flatnonzero(rows)
            ]
            stage_test_metrics[stage_name] = (
                _disorder_metrics_from_probabilities(
                    y[test_idx][rows],
                    test_stage_probabilities[rows, stage_index],
                    class_names,
                    subject_ids=stage_subjects,
                )
            )
        print(
            f"[Disorder Nested Stage+Architecture Test] Fold {fold_idx} | "
            f"acc={test_metrics['acc']:.4f}, "
            f"balanced_acc={test_metrics['balanced_acc']:.4f}, "
            f"macro_precision={test_metrics['macro_precision']:.4f}, "
            f"macro_recall={test_metrics['macro_recall']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"kappa={test_metrics['kappa']:.4f}, "
            f"macro_ovr_auroc={test_metrics['macro_ovr_auroc']:.4f}"
        )
        print(
            "  component macro_OVR_AUC: "
            f"stage_only={test_stage_only_metrics['macro_ovr_auroc']:.4f} | "
            f"architecture_only="
            f"{test_architecture_only_metrics['macro_ovr_auroc']:.4f} | "
            f"selected architecture_weight="
            f"{best_rf_settings['architecture_fusion_weight']:.2f}"
        )
        print(
            "  decision calibration: "
            f"uncalibrated_ACC={test_uncalibrated_metrics['acc']:.4f} -> "
            f"calibrated_ACC={test_metrics['acc']:.4f} | "
            f"uncalibrated_balanced_ACC="
            f"{test_uncalibrated_metrics['balanced_acc']:.4f} -> "
            f"calibrated_balanced_ACC={test_metrics['balanced_acc']:.4f} | "
            "AUC uses uncalibrated ranking probabilities"
        )
        print(
            "  fused present-stage counts="
            + str(test_contributing_stage_counts.tolist())
        )
        for name, values in test_metrics["per_class_metrics"].items():
            print(
                f"  {name}: ovr_acc={values['ovr_acc']:.4f}, "
                f"auroc={values['auroc']:.4f}, "
                f"precision={values['precision']:.4f}, "
                f"recall/sensitivity={values['recall']:.4f}, "
                f"specificity={values['specificity']:.4f}, "
                f"f1={values['f1']:.4f}, "
                f"support={values['support']}"
            )

        model_bundle = {
            "stage_models": {
                stage_name: stage_models[index]
                for index, stage_name in enumerate(stage_names)
            },
            "architecture_model": architecture_model,
            "model_name": (
                "FiveStageProbabilityFusionPCAClassBalancedRFPlus"
                "SleepArchitectureLogistic"
            ),
            "repeat_seed": int(run_seed),
            "fold_seed": int(fold_seed),
            "final_model_seed": int(final_model_seed),
            "selected_settings": best_rf_settings,
            "decision_multipliers": decision_calibration["multipliers"],
            "decision_calibration_affects_auc": False,
            "architecture_model_seed": int(architecture_model_seed),
            "architecture_feature_names": architecture_feature_names,
            "stage_training_class_counts": {
                stage_name: stage_training_counts[index]
                for index, stage_name in enumerate(stage_names)
            },
            "fusion": {
                "within_stage_models": "uniform_mean_over_present_stages",
                "stage_probability_weight": float(
                    best_rf_settings["stage_fusion_weight"]
                ),
                "sleep_architecture_probability_weight": float(
                    best_rf_settings["architecture_fusion_weight"]
                ),
            },
            "class_names": class_names,
            "backbone_path": os.path.abspath(cfg.eeg_pretrained_model_path),
            "backbone_frozen": True,
            "fixed_stft_used": False,
            "feature_protocol": (
                "separate_stage_mean_population_std_plus_all_stage_fractions"
                if use_stage_std
                else "separate_stage_mean_plus_all_stage_fractions"
            ),
            "per_stage_feature_dim": int(stage_rf_features.shape[2]),
            "sleep_architecture_feature_dim": int(
                architecture_features.shape[1]
            ),
            "train_subjects": trainval_subjects,
        }
        with open(
            os.path.join(
                fold_dir,
                "best_nested_stage_architecture_fusion.pkl",
            ), "wb"
        ) as handle:
            pickle.dump(model_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        save_disorder_confusion_matrix_plot(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            class_names,
            os.path.join(fold_dir, "confusion_matrix_test.png"),
            f"Disorder Nested Stage+Architecture Fusion Fold {fold_idx}",
        )
        if len(test_embeddings) > 1:
            save_tsne_plot(
                test_embeddings,
                np.asarray(test_metrics["y_true"]),
                class_names,
                os.path.join(fold_dir, "tsne_test_subjects.png"),
                f"Disorder Nested Stage+Architecture Fusion Fold {fold_idx}",
                seed=fold_seed,
                max_points=1000,
            )

        result = {
            "task_name": task_name,
            "model": (
                "nested_cv_stage_rf_plus_sleep_architecture_logistic_fusion"
            ),
            "repeat_seed": int(run_seed),
            "fold_seed": int(fold_seed),
            "fold_idx": int(fold_idx),
            "train_subjects": trainval_subjects,
            "val_subjects": [],
            "test_subjects": test_subjects,
            "best_epoch": None,
            "inner_cv_best": best_rf_settings,
            "test_metrics": test_metrics,
            "probability_fusion": {
                "stage_probability_weight": float(
                    best_rf_settings["stage_fusion_weight"]
                ),
                "sleep_architecture_probability_weight": float(
                    best_rf_settings["architecture_fusion_weight"]
                ),
                "stage_only_probabilities": (
                    test_stage_fused_probabilities.tolist()
                ),
                "sleep_architecture_only_probabilities": (
                    test_architecture_probabilities.tolist()
                ),
                "stage_only_test_metrics": test_stage_only_metrics,
                "sleep_architecture_only_test_metrics": (
                    test_architecture_only_metrics
                ),
                "uncalibrated_fused_test_metrics": (
                    test_uncalibrated_metrics
                ),
                "decision_calibration": decision_calibration,
                "decision_calibration_note": (
                    "multipliers_affect_argmax_and_confusion_metrics_only;_"
                    "ROC_AUC_uses_uncalibrated_fused_probabilities"
                ),
            },
            "stage_fusion": {
                "stage_names": stage_names,
                "available_mask": stage_available[test_idx].astype(int).tolist(),
                "contributing_stage_counts": (
                    test_contributing_stage_counts.tolist()
                ),
                "per_stage_test_metrics": stage_test_metrics,
            },
            "history": search_history,
        }
        with open(
            os.path.join(fold_dir, "result.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        fold_results.append(result)
        split_manifest["folds"].append({
            "fold_idx": int(fold_idx),
            "fold_seed": int(fold_seed),
            "train_subjects": trainval_subjects,
            "val_subjects": [],
            "test_subjects": test_subjects,
            "inner_cv_best": best_rf_settings,
        })
        del stage_models, architecture_model

    summary = {
        key: value for key, value in split_manifest.items() if key != "folds"
    }
    summary["fold_results"] = fold_results
    for metric_name in [
        "acc", "balanced_acc", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "kappa", "auroc"
    ]:
        values = [
            float(result["test_metrics"][metric_name])
            for result in fold_results
        ]
        summary[f"mean_{metric_name}"] = float(np.mean(values))
        summary[f"std_{metric_name}"] = float(np.std(values))

    per_class_summary = {}
    class_metric_names = [
        "ovr_acc", "auroc", "precision", "recall", "specificity", "f1"
    ]
    for class_name in class_names:
        per_class_summary[class_name] = {}
        for metric_name in class_metric_names:
            values = [
                float(
                    result["test_metrics"]["per_class_metrics"]
                    [class_name][metric_name]
                )
                for result in fold_results
            ]
            per_class_summary[class_name][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "fold_values": values,
            }
        per_class_summary[class_name]["total_support"] = int(sum(
            int(
                result["test_metrics"]["per_class_metrics"]
                [class_name]["support"]
            )
            for result in fold_results
        ))
    summary["per_class_metrics_mean_std"] = per_class_summary

    pooled_cm = np.sum([
        np.asarray(
            result["test_metrics"]["confusion_matrix"], dtype=np.int64
        )
        for result in fold_results
    ], axis=0)
    pooled_cm_normalized = pooled_cm.astype(np.float64) / np.clip(
        pooled_cm.sum(axis=1, keepdims=True), 1, None
    )
    summary["pooled_confusion_matrix"] = pooled_cm.tolist()
    summary["pooled_confusion_matrix_row_normalized"] = (
        pooled_cm_normalized.tolist()
    )
    pooled_y_true = [
        label
        for result in fold_results
        for label in result["test_metrics"]["y_true"]
    ]
    pooled_y_pred = [
        label
        for result in fold_results
        for label in result["test_metrics"]["y_pred"]
    ]
    pooled_subject_ids = [
        subject_id
        for result in fold_results
        for subject_id in result["test_metrics"]["subject_ids"]
    ]
    pooled_probabilities = [
        probability
        for result in fold_results
        for probability in result["test_metrics"]["probabilities"]
    ]
    expected_subject_ids = {
        str(record["subject_id"]) for record in feature_records
    }
    if len(pooled_subject_ids) != len(feature_records):
        raise RuntimeError(
            f"seed={run_seed}: pooled OOF predictions contain "
            f"{len(pooled_subject_ids)} subjects; expected "
            f"{len(feature_records)}."
        )
    if len(set(pooled_subject_ids)) != len(pooled_subject_ids):
        raise RuntimeError(
            f"seed={run_seed}: duplicate subjects found in pooled OOF predictions."
        )
    if set(pooled_subject_ids) != expected_subject_ids:
        raise RuntimeError(
            f"seed={run_seed}: pooled OOF subject IDs do not match the input cohort."
        )
    pooled_oof_metrics = _disorder_metrics_from_probabilities(
        pooled_y_true,
        pooled_probabilities,
        class_names,
        subject_ids=pooled_subject_ids,
        hard_predictions=pooled_y_pred,
    )
    summary["pooled_oof_metrics"] = pooled_oof_metrics
    save_disorder_confusion_matrix_plot(
        pooled_y_true,
        pooled_y_pred,
        class_names,
        os.path.join(task_out_dir, "confusion_matrix_all_folds.png"),
        "Disorder Stage+Architecture Fusion pooled outer-test confusion matrix",
    )
    with open(
        os.path.join(task_out_dir, "fold_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(split_manifest, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(task_out_dir, "summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        f"\n[Disorder Stage+Architecture Fusion Fold-Mean Summary] seed={run_seed} | "
        f"ACC={summary['mean_acc']:.4f}+-{summary['std_acc']:.4f} | "
        f"balanced_ACC={summary['mean_balanced_acc']:.4f}+-"
        f"{summary['std_balanced_acc']:.4f} | "
        f"macro_precision={summary['mean_macro_precision']:.4f}+-"
        f"{summary['std_macro_precision']:.4f} | "
        f"macro_recall={summary['mean_macro_recall']:.4f}+-"
        f"{summary['std_macro_recall']:.4f} | "
        f"macro_F1={summary['mean_macro_f1']:.4f}+-"
        f"{summary['std_macro_f1']:.4f} | "
        f"macro_OVR_AUC={summary['mean_auroc']:.4f}+-"
        f"{summary['std_auroc']:.4f} | "
        f"Kappa={summary['mean_kappa']:.4f}+-{summary['std_kappa']:.4f}"
    )
    print("[Disorder Stage+Architecture Fusion Per-Class Summary]")
    for class_name in class_names:
        values = per_class_summary[class_name]
        print(
            f"  {class_name}: "
            f"OVR_ACC={values['ovr_acc']['mean']:.4f}+-"
            f"{values['ovr_acc']['std']:.4f} | "
            f"AUC={values['auroc']['mean']:.4f}+-"
            f"{values['auroc']['std']:.4f} | "
            f"Precision={values['precision']['mean']:.4f}+-"
            f"{values['precision']['std']:.4f} | "
            f"Recall={values['recall']['mean']:.4f}+-"
            f"{values['recall']['std']:.4f} | "
            f"F1={values['f1']['mean']:.4f}+-"
            f"{values['f1']['std']:.4f} | "
            f"support={values['total_support']}"
        )
    print("[Disorder Stage+Architecture Fusion Pooled Row-Normalized Confusion Matrix]")
    print("  Ground Truth \\ Prediction -> " + " | ".join(class_names))
    for row_index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: "
            + " | ".join(
                f"{value:.4f}" for value in pooled_cm_normalized[row_index]
            )
        )
    print(
        f"[Disorder Stage+Architecture Fusion Pooled OOF] seed={run_seed} | "
        f"subjects={pooled_oof_metrics['n_subjects']} | "
        f"ACC={pooled_oof_metrics['acc']:.4f} | "
        f"balanced_ACC={pooled_oof_metrics['balanced_acc']:.4f} | "
        f"macro_F1={pooled_oof_metrics['macro_f1']:.4f} | "
        f"macro_OVR_AUC={pooled_oof_metrics['macro_ovr_auroc']:.4f}"
    )
    print(
        "[Disorder Stage+Architecture Fusion Pooled OOF Per-Class AUC] "
        + " | ".join(
            f"{class_name}="
            f"{pooled_oof_metrics['per_class_metrics'][class_name]['auroc']:.4f}"
            for class_name in class_names
        )
    )
    return summary


def _summarize_disorder_repeats(
    repeat_summaries, repeat_seeds, class_names, task_out_root
):
    """Aggregate one pooled OOF result per complete repeated CV run."""
    if len(repeat_summaries) != len(repeat_seeds):
        raise RuntimeError(
            "Repeated disorder CV summaries and seeds have different lengths."
        )

    compact_results = []
    for repeat_index, (repeat_seed, summary) in enumerate(
        zip(repeat_seeds, repeat_summaries), 1
    ):
        pooled_metrics = summary.get("pooled_oof_metrics")
        if pooled_metrics is None:
            raise RuntimeError(
                f"seed={repeat_seed} has no pooled_oof_metrics."
            )
        compact_results.append({
            "repeat_index": int(repeat_index),
            "seed": int(repeat_seed),
            "output_dir": str(summary.get("output_dir", "")),
            "pooled_oof_metrics": pooled_metrics,
        })

    aggregate = {
        "task_name": "sleep_disorder_classification",
        "model": "nested_cv_sleep_stage_probability_fusion_random_forest",
        "protocol": "repeated_subject_level_stratified_cv_pooled_oof",
        "n_repeats": int(len(repeat_seeds)),
        "repeat_seeds": [int(seed) for seed in repeat_seeds],
        "std_definition": (
            "numpy population std (ddof=0) across complete repeat-level "
            "pooled OOF metrics"
        ),
        "repeat_results": compact_results,
    }

    metric_names = [
        "acc",
        "balanced_acc",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "kappa",
        "macro_ovr_auroc",
    ]
    aggregate_metrics = {}
    for metric_name in metric_names:
        values = [
            float(result["pooled_oof_metrics"][metric_name])
            for result in compact_results
        ]
        aggregate_metrics[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "repeat_values": values,
        }
    aggregate["pooled_oof_metrics_mean_std"] = aggregate_metrics

    class_metric_names = [
        "ovr_acc", "auroc", "precision", "recall", "specificity", "f1"
    ]
    per_class = {}
    for class_name in class_names:
        per_class[class_name] = {}
        for metric_name in class_metric_names:
            values = [
                float(
                    result["pooled_oof_metrics"]["per_class_metrics"]
                    [class_name][metric_name]
                )
                for result in compact_results
            ]
            per_class[class_name][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "repeat_values": values,
            }
    aggregate["per_class_pooled_oof_metrics_mean_std"] = per_class

    repeat_normalized_matrices = np.stack([
        np.asarray(
            result["pooled_oof_metrics"][
                "confusion_matrix_row_normalized"
            ],
            dtype=np.float64,
        )
        for result in compact_results
    ], axis=0)
    mean_normalized_matrix = np.mean(repeat_normalized_matrices, axis=0)
    std_normalized_matrix = np.std(repeat_normalized_matrices, axis=0)
    aggregate["mean_normalized_confusion_matrix"] = (
        mean_normalized_matrix.tolist()
    )
    aggregate["std_normalized_confusion_matrix"] = (
        std_normalized_matrix.tolist()
    )
    aggregate["confusion_matrix_class_order"] = list(class_names)
    aggregate["repeat_normalized_confusion_matrices"] = (
        repeat_normalized_matrices.tolist()
    )

    comparison_order = [
        class_name for class_name in ["NFLE", "RBD", "HC"]
        if class_name in class_names
    ]
    if len(comparison_order) == len(class_names):
        comparison_indices = [
            class_names.index(class_name) for class_name in comparison_order
        ]
        comparison_mean = mean_normalized_matrix[
            np.ix_(comparison_indices, comparison_indices)
        ]
        comparison_std = std_normalized_matrix[
            np.ix_(comparison_indices, comparison_indices)
        ]
        aggregate["sleepgpt_comparison_confusion_matrix"] = {
            "class_order": comparison_order,
            "mean_normalized": comparison_mean.tolist(),
            "std_normalized": comparison_std.tolist(),
            "aggregation": (
                "cell-wise mean/std of one row-normalized pooled-OOF "
                "confusion matrix per complete repeated CV experiment"
            ),
        }
        save_mean_normalized_confusion_matrix_plot(
            comparison_mean,
            comparison_std,
            comparison_order,
            os.path.join(
                task_out_root,
                "mean_normalized_confusion_matrix_sleepgpt_order.png",
            ),
            "Disorder repeated-CV mean normalized confusion matrix",
        )

    save_mean_normalized_confusion_matrix_plot(
        mean_normalized_matrix,
        std_normalized_matrix,
        class_names,
        os.path.join(
            task_out_root, "mean_normalized_confusion_matrix.png"
        ),
        "Disorder repeated-CV mean normalized confusion matrix",
    )

    summary_path = os.path.join(
        task_out_root, "repeated_cv_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    if len(repeat_seeds) == 10:
        with open(
            os.path.join(task_out_root, "ten_repeat_summary.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    print(
        f"\n[Disorder Stage+Architecture Fusion Repeated-CV Complete] "
        f"repeats={len(repeat_seeds)} | "
        f"seeds={list(repeat_seeds)}"
    )
    print(
        "[Disorder Stage+Architecture Fusion Repeat-Level Pooled-OOF Summary] "
        + " | ".join(
            f"{name}={aggregate_metrics[name]['mean']:.4f}+-"
            f"{aggregate_metrics[name]['std']:.4f}"
            for name in [
                "acc", "balanced_acc", "macro_f1", "macro_ovr_auroc", "kappa"
            ]
        )
    )
    print("[Disorder Stage+Architecture Fusion Repeat-Level Per-Class Pooled-OOF AUC]")
    for class_name in class_names:
        values = per_class[class_name]["auroc"]
        print(
            f"  {class_name}: AUC={values['mean']:.4f}+-"
            f"{values['std']:.4f} | values="
            + ", ".join(f"{value:.4f}" for value in values["repeat_values"])
        )
    matrix_order = comparison_order if len(comparison_order) == len(class_names) else list(class_names)
    if len(comparison_order) == len(class_names):
        matrix_to_print = comparison_mean
    else:
        matrix_to_print = mean_normalized_matrix
    print(
        "[Disorder Stage+Architecture Fusion Mean Normalized Confusion Matrix "
        "Across Repeats]"
    )
    print("  Ground Truth \\ Prediction -> " + " | ".join(matrix_order))
    for row_index, class_name in enumerate(matrix_order):
        print(
            f"  {class_name}: "
            + " | ".join(
                f"{value:.4f}" for value in matrix_to_print[row_index]
            )
        )
    print(
        f"[Disorder Stage+Architecture Fusion Repeated-CV Summary Saved] "
        f"{summary_path}"
    )
    return aggregate


def load_clas_spindle_records(cfg):
    """Read CLAS C3-A1 morphology signals and spindle onset/duration annotations."""
    paths = {}
    for directory in cfg.spindle_train_dirs:
        if not os.path.isdir(directory):
            raise FileNotFoundError(directory)
        for name in sorted(os.listdir(directory)):
            if re.fullmatch(r"Base_SLP\d+_morphology\.npz", name):
                subject = name.removesuffix("_morphology.npz")
                if subject in paths:
                    raise ValueError(f"Duplicate CLAS subject: {subject}")
                paths[subject] = os.path.join(directory, name)
    label_dir = cfg.spindle_annotation_dir
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(label_dir)
    annotations = {name.removesuffix("_Spindle.txt"): os.path.join(label_dir, name)
                   for name in os.listdir(label_dir)
                   if re.fullmatch(r"Base_SLP\d+_Spindle\.txt", name)}
    if set(paths) != set(annotations) or len(paths) != 35:
        raise ValueError(f"Expected 35 matched CLAS subjects; signals={len(paths)}, labels={len(annotations)}, "
                         f"missing_labels={sorted(set(paths)-set(annotations))}, "
                         f"missing_signals={sorted(set(annotations)-set(paths))}")
    records = []
    for subject, path in sorted(paths.items()):
        with np.load(path, allow_pickle=False) as data:
            fs = float(data["sampling_rate"].item())
            epoch_seconds = float(data["epoch_duration"].item())
            if fs != cfg.spindle_sampling_rate or epoch_seconds != 30:
                raise ValueError(f"Unexpected time base for {subject}: {fs} Hz, {epoch_seconds} s")
            if str(data["channel"].item()).upper() != "C3-A1":
                raise ValueError(f"Expected C3-A1 signal for {subject}")
            if str(data["subject_id"].item()) != subject:
                raise ValueError(f"Subject ID mismatch: {path}")
            eeg = np.asarray(data["eeg_epochs"], dtype=np.float32)
            loc = np.asarray(data["eog_loc_epochs"], dtype=np.float32)
            roc = np.asarray(data["eog_roc_epochs"], dtype=np.float32)
            stages = np.asarray(data["true_sleep_stages"], dtype=np.int64)
            if (eeg.ndim != 2 or eeg.shape != loc.shape or eeg.shape != roc.shape
                    or eeg.shape != (len(stages), int(fs * epoch_seconds))):
                raise ValueError(f"Signal/stage shape mismatch for {subject}")
            if not all(np.isfinite(x).all() for x in (eeg, loc, roc)):
                raise ValueError(f"Nonfinite signals for {subject}")
            if not np.isin(stages, [0, 1, 2, 3, 4]).all():
                raise ValueError(f"Invalid CLAS stages for {subject}")
            availability = np.asarray(data["eog_available"], dtype=np.float32).reshape(2)
        duration = eeg.size / fs
        intervals = []
        with open(annotations[subject], encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if line.startswith("["):
                    if line != "[Spindle/C3-A1]":
                        raise ValueError(f"Unexpected annotation channel: {subject}: {line}")
                    continue
                if not line:
                    continue
                fields = re.split(r"[\s,;]+", line)
                if len(fields) != 2:
                    raise ValueError(f"Malformed event: {subject}:{line_number}")
                onset, length = map(float, fields)
                end = onset + length
                if not np.isfinite([onset, length]).all() or onset < 0 or length <= 0 or end > duration:
                    raise ValueError(f"Event outside recording or invalid: {subject}:{line_number}")
                intervals.append((onset, end))
        labels = np.zeros(eeg.size, dtype=np.int8)
        for start, end in intervals:
            labels[int(np.floor(start * fs)):int(np.ceil(end * fs))] = 1
        labels = labels.reshape(eeg.shape)
        valid = np.broadcast_to((stages == 2)[:, None], eeg.shape).copy()
        if not valid.any():
            raise ValueError(f"No N2 epochs for {subject}")
        labels[~valid] = -100
        records.append(dict(subject_id=subject, eeg_epochs=eeg, eog_loc_epochs=loc,
                            eog_roc_epochs=roc, eog_available=availability,
                            point_labels=labels, point_valid_mask=valid.astype(np.uint8),
                            n2_point_mask=valid.astype(np.uint8),
                            expert_intervals=np.asarray(sorted(intervals), dtype=np.float32).reshape(-1, 2),
                            sampling_rate=int(fs), source_file=path,
                            annotation_file=annotations[subject], label_source="CLAS spindle annotations"))
        print(f"[CLAS Spindle] {subject}: epochs={len(stages)}, N2={int((stages == 2).sum())}, events={len(intervals)}")
    print(f"[CLAS Spindle Data] subjects={len(records)} | source=CLAS annotations | evaluation=N2")
    return records


def load_bp_spindle_records(cfg):
    """Read T3 morphology signals and C3 YASA onset/duration annotations."""
    paths = {}
    for directory in cfg.spindle_train_dirs:
        if not os.path.isdir(directory):
            raise FileNotFoundError(directory)
        for name in sorted(os.listdir(directory)):
            if re.fullmatch(r"BP-SLP\d+_morphology\.npz", name):
                subject = name.removesuffix("_morphology.npz")
                if subject in paths:
                    raise ValueError(f"Duplicate BP subject: {subject}")
                paths[subject] = os.path.join(directory, name)
    label_dir = cfg.spindle_annotation_dir
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(label_dir)
    annotations = {name.removesuffix("_C3_YASA_Spindle.txt"): os.path.join(label_dir, name)
                   for name in os.listdir(label_dir)
                   if re.fullmatch(r"BP-SLP\d+_C3_YASA_Spindle\.txt", name)}
    if set(paths) != set(annotations) or len(paths) != 20:
        raise ValueError(f"Expected 20 matched BP subjects; signals={len(paths)}, labels={len(annotations)}, "
                         f"missing_labels={sorted(set(paths)-set(annotations))}, "
                         f"missing_signals={sorted(set(annotations)-set(paths))}")
    records = []
    for subject, path in sorted(paths.items()):
        with np.load(path, allow_pickle=False) as data:
            fs = float(data["sampling_rate"].item())
            epoch_seconds = float(data["epoch_duration"].item())
            if fs != cfg.spindle_sampling_rate or epoch_seconds != 30:
                raise ValueError(f"Unexpected time base for {subject}: {fs} Hz, {epoch_seconds} s")
            if str(data["label_technician"].item()) != "T3":
                raise ValueError(f"Expected T3 labels for {subject}")
            if str(data["eeg_channel"].item()).upper() != "C3":
                raise ValueError(f"Expected C3 signal for {subject}")
            if str(data["subject_id"].item()) != subject:
                raise ValueError(f"Subject ID mismatch: {path}")
            eeg = np.asarray(data["eeg_epochs"], dtype=np.float32)
            loc = np.asarray(data["eog_loc_epochs"], dtype=np.float32)
            roc = np.asarray(data["eog_roc_epochs"], dtype=np.float32)
            stages = np.asarray(data["true_sleep_stages"], dtype=np.int64)
            if (eeg.ndim != 2 or eeg.shape != loc.shape or eeg.shape != roc.shape
                    or eeg.shape != (len(stages), int(fs * epoch_seconds))):
                raise ValueError(f"Signal/stage shape mismatch for {subject}")
            if not all(np.isfinite(x).all() for x in (eeg, loc, roc)):
                raise ValueError(f"Nonfinite signals for {subject}")
            if not np.isin(stages, [0, 1, 2, 3, 4]).all():
                raise ValueError(f"Invalid T3 stages for {subject}")
            availability = np.asarray(data["eog_available"], dtype=np.float32).reshape(2)
        duration = eeg.size / fs
        intervals = []
        with open(annotations[subject], encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line or line.startswith("["):
                    continue
                fields = re.split(r"[\s,;]+", line)
                if len(fields) != 2:
                    raise ValueError(f"Malformed event: {subject}:{line_number}")
                onset, length = map(float, fields)
                end = onset + length
                if not np.isfinite([onset, length]).all() or onset < 0 or length <= 0 or end > duration:
                    raise ValueError(f"Event outside recording or invalid: {subject}:{line_number}")
                intervals.append((onset, end))
        labels = np.zeros(eeg.size, dtype=np.int8)
        for start, end in intervals:
            labels[int(np.floor(start * fs)):int(np.ceil(end * fs))] = 1
        labels = labels.reshape(eeg.shape)
        valid = np.broadcast_to((stages == 2)[:, None], eeg.shape).copy()
        if not valid.any():
            raise ValueError(f"No T3 N2 epochs for {subject}")
        labels[~valid] = -100
        records.append(dict(subject_id=subject, eeg_epochs=eeg, eog_loc_epochs=loc,
                            eog_roc_epochs=roc, eog_available=availability,
                            point_labels=labels, point_valid_mask=valid.astype(np.uint8),
                            n2_point_mask=valid.astype(np.uint8),
                            expert_intervals=np.asarray(sorted(intervals), dtype=np.float32).reshape(-1, 2),
                            sampling_rate=int(fs), source_file=path,
                            annotation_file=annotations[subject], label_source="YASA", label_technician="T3"))
        print(f"[BP Spindle] {subject}: epochs={len(stages)}, N2={int((stages == 2).sum())}, events={len(intervals)}")
    print(f"[BP Spindle Data] subjects={len(records)} | source=YASA | evaluation=T3 N2")
    return records


def load_spindle_records_from_dirs(processed_dirs, expert="E1"):
    """加载预处理NPZ，并只选择指定专家具有有效标注的记录。

    UNION表示两位专家标注区间取并集；E2缺失的excerpt7/8会自动
    使用E1标签。单独选择E2时，excerpt7/8会被跳过。
    """
    expert = str(expert).upper()
    if expert not in {"E1", "E2", "UNION"}:
        raise ValueError(
            f"spindle_expert must be E1, E2, or UNION, got {expert!r}."
        )
    suffix = expert.lower()
    records = []
    seen_subjects = set()
    for processed_dir in processed_dirs:
        if not os.path.isdir(processed_dir):
            raise FileNotFoundError(
                f"Spindle preprocessing directory does not exist: {processed_dir}"
            )
        file_paths = sorted(
            os.path.join(processed_dir, name)
            for name in os.listdir(processed_dir)
            if name.lower().endswith("_spindles.npz")
        )
        for file_path in file_paths:
            with np.load(file_path, allow_pickle=True) as data:
                subject_id = _safe_scalar_to_str(data["subject_id"])
                available = np.asarray(
                    data.get("spindle_expert_available", [0, 0])
                ).reshape(-1)
                if expert == "UNION":
                    label_available = bool(np.any(available > 0))
                else:
                    expert_index = 0 if expert == "E1" else 1
                    label_available = (
                        available.size > expert_index
                        and int(available[expert_index]) > 0
                    )
                if not label_available:
                    continue
                labels_key = f"spindle_point_labels_{suffix}"
                valid_key = f"spindle_valid_mask_{suffix}"
                intervals_key = f"spindle_intervals_{suffix}"
                required = [
                    "eeg_epochs", "eog_loc_epochs", "eog_roc_epochs",
                    labels_key, valid_key, "n2_point_mask",
                ]
                missing = [key for key in required if key not in data]
                if missing:
                    raise KeyError(f"{file_path} is missing keys: {missing}")
                eeg = np.asarray(data["eeg_epochs"], dtype=np.float32)
                loc = np.asarray(data["eog_loc_epochs"], dtype=np.float32)
                roc = np.asarray(data["eog_roc_epochs"], dtype=np.float32)
                labels = np.asarray(data[labels_key], dtype=np.int8)
                valid = np.asarray(data[valid_key], dtype=np.uint8)
                n2_mask = np.asarray(data["n2_point_mask"], dtype=np.uint8)
                n = min(
                    len(eeg), len(loc), len(roc), len(labels), len(valid),
                    len(n2_mask),
                )
                if n == 0:
                    continue
                if subject_id in seen_subjects:
                    raise ValueError(f"Duplicate spindle subject_id: {subject_id}")
                seen_subjects.add(subject_id)
                intervals = (
                    np.asarray(data[intervals_key], dtype=np.float32).reshape(-1, 2)
                    if intervals_key in data
                    else np.empty((0, 2), dtype=np.float32)
                )
                sampling_rate = int(np.asarray(data["sampling_rate"]).item())
                # SUMOv2只在N2中训练和评估。专家有效掩码负责排除E1的
                # 1000秒后区域；N2掩码负责排除所有非N2区间。
                effective_valid = (
                    valid[:n].astype(bool) & n2_mask[:n].astype(bool)
                )
                effective_labels = labels[:n].copy()
                effective_labels[~effective_valid] = -100
                records.append({
                    "subject_id": subject_id,
                    "eeg_epochs": eeg[:n],
                    "eog_loc_epochs": loc[:n],
                    "eog_roc_epochs": roc[:n],
                    "eog_available": np.asarray(
                        data.get("eog_available", [1.0, 1.0]),
                        dtype=np.float32,
                    ).reshape(2),
                    "point_labels": effective_labels,
                    "point_valid_mask": effective_valid.astype(np.uint8),
                    "n2_point_mask": n2_mask[:n],
                    "expert_intervals": intervals,
                    "sampling_rate": sampling_rate,
                    "source_file": file_path,
                })
    print(
        f"[Spindle Data] expert={expert} | subjects={len(records)} | "
        f"events={sum(len(rec['expert_intervals']) for rec in records)}"
    )
    return records


def build_spindle_samples(
    records,
    seq_len=20,
    stride=10,
    segment_len=None,
    non_overlapping=False,
):
    """按连续epoch构建窗口，并保留原始epoch起点用于重叠预测聚合。

    point_valid_mask控制逐采样点监督；valid_mask只表示窗口中的epoch
    是否为真实epoch（而不是padding），二者不能混用。
    """
    samples = []
    for rec in records:
        signal_epochs, eog_available = build_signal_epochs(rec)
        labels = np.asarray(rec["point_labels"], dtype=np.int8)
        point_valid = np.asarray(rec["point_valid_mask"], dtype=np.uint8)
        n = min(len(signal_epochs), len(labels), len(point_valid))
        if n == 0:
            continue
        signal_epochs = signal_epochs[:n]
        labels = labels[:n]
        point_valid = point_valid[:n]
        content_len = (
            int(seq_len) if segment_len is None else int(segment_len)
        )
        if content_len <= 0 or content_len > int(seq_len):
            raise ValueError(
                "Spindle segment_len must be in [1, seq_len], got "
                f"segment_len={content_len}, seq_len={seq_len}."
            )
        if non_overlapping:
            starts = list(range(0, n, content_len))
        else:
            starts = build_window_starts(n, content_len, stride)
        for start in starts:
            end = min(start + content_len, n)
            valid_len = end - start
            window_valid = pad_valid_mask(valid_len, seq_len)
            window_eeg = pad_eeg_epochs(signal_epochs[start:end], seq_len)
            window_eeg = normalize_eeg_window(window_eeg, window_valid)
            label_pad = np.full(
                (seq_len, labels.shape[1]), -100, dtype=np.int8
            )
            point_valid_pad = np.zeros(
                (seq_len, labels.shape[1]), dtype=np.uint8
            )
            label_pad[:valid_len] = labels[start:end]
            point_valid_pad[:valid_len] = point_valid[start:end]
            if not np.any(point_valid_pad):
                continue
            samples.append({
                "subject_id": rec["subject_id"],
                "start_idx": int(start),
                "start_point": int(start * labels.shape[1]),
                "valid_len": int(valid_len),
                "segment_len": int(content_len),
                "total_epochs": int(n),
                "eeg": window_eeg,
                "eog_available": eog_available,
                "valid_mask": window_valid,
                "point_labels": label_pad,
                "point_valid_mask": point_valid_pad,
            })
    return samples


def build_spindle_point_frames(
    records,
    frame_seconds=1.0,
    overlap=0.50,
):
    """Build short overlapping point-level frames from complete records."""
    frame_seconds = float(frame_seconds)
    overlap = float(overlap)
    if frame_seconds <= 0.0:
        raise ValueError(
            "spindle_random_frame_seconds must be > 0, got "
            f"{frame_seconds}."
        )
    if not 0.0 <= overlap < 1.0:
        raise ValueError(
            "spindle_random_frame_overlap must be in [0, 1), got "
            f"{overlap}."
        )

    frames = []
    for rec in records:
        sampling_rate = int(rec["sampling_rate"])
        frame_points = int(round(frame_seconds * sampling_rate))
        stride_points = max(
            1, int(round(frame_points * (1.0 - overlap)))
        )
        signal_epochs, eog_available = build_signal_epochs(rec)
        if len(signal_epochs) == 0:
            continue

        epoch_points = int(signal_epochs.shape[-1])
        continuous_signal = (
            signal_epochs.transpose(1, 0, 2)
            .reshape(signal_epochs.shape[1], -1)
        )
        labels = np.asarray(rec["point_labels"], dtype=np.int8).reshape(-1)
        point_valid = np.asarray(
            rec["point_valid_mask"], dtype=np.uint8
        ).reshape(-1)
        total_points = min(
            continuous_signal.shape[-1], len(labels), len(point_valid)
        )
        if total_points < frame_points:
            continue
        continuous_signal = continuous_signal[:, :total_points]
        labels = labels[:total_points]
        point_valid = point_valid[:total_points]

        for start_point in range(
            0, total_points - frame_points + 1, stride_points
        ):
            end_point = start_point + frame_points
            frame_valid = point_valid[start_point:end_point]
            if not np.any(frame_valid):
                continue
            frame_eeg = continuous_signal[
                :, start_point:end_point
            ][None, :, :].astype(np.float32, copy=True)
            frame_eeg = normalize_eeg_window(
                frame_eeg, np.ones(1, dtype=np.int64)
            )
            frames.append({
                "subject_id": rec["subject_id"],
                "start_idx": int(start_point // epoch_points),
                "start_point": int(start_point),
                "valid_len": 1,
                "segment_len": 1,
                "total_epochs": int(
                    np.ceil(total_points / float(epoch_points))
                ),
                "total_points": int(total_points),
                "frame_points": int(frame_points),
                "frame_seconds": float(
                    frame_points / float(sampling_rate)
                ),
                "stride_points": int(stride_points),
                "stride_seconds": float(
                    stride_points / float(sampling_rate)
                ),
                "eeg": frame_eeg,
                "eog_available": eog_available,
                "valid_mask": np.ones(1, dtype=np.int64),
                "point_labels": labels[
                    start_point:end_point
                ][None, :].copy(),
                "point_valid_mask": frame_valid[None, :].copy(),
            })
    return frames


class SpindleWindowDataset(Dataset):
    """把numpy窗口转换成PyTorch张量，不在这里重新修改标签。"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "eeg": torch.tensor(sample["eeg"], dtype=torch.float32),
            "eog_available": torch.tensor(
                sample["eog_available"], dtype=torch.float32
            ),
            "valid_mask": torch.tensor(sample["valid_mask"], dtype=torch.long),
            "point_labels": torch.tensor(
                sample["point_labels"], dtype=torch.float32
            ),
            "point_valid_mask": torch.tensor(
                sample["point_valid_mask"], dtype=torch.bool
            ),
            "subject_id": sample["subject_id"],
            "start_idx": torch.tensor(sample["start_idx"], dtype=torch.long),
            "start_point": torch.tensor(
                sample["start_point"], dtype=torch.long
            ),
            "valid_len": torch.tensor(sample["valid_len"], dtype=torch.long),
        }


def _make_spindle_backbone(cfg):
    return SleepTextBackbone(
        pretrained_model_dir=cfg.pretrained_model_dir,
        seq_len=int(cfg.spindle_seq_len),
        dropout=cfg.dropout,
        seq_num_layers=cfg.seq_num_layers,
        seq_num_heads=cfg.seq_num_heads,
        bert_chunk_size=cfg.bert_chunk_size,
        eeg_hidden_dim=cfg.eeg_hidden_dim,
        text_proj_dim=cfg.text_proj_dim,
        fusion_dim=cfg.fusion_dim,
        eeg_kernel_size=cfg.eeg_kernel_size,
    )


def _set_spindle_backbone_frozen(model):
    """冻结整个上游，warm-up阶段只训练新增的检测头。"""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def _unfreeze_final_spindle_transformer_layer(model):
    """只解冻Transformer最后一层，其余骨干参数保持冻结。"""
    _set_spindle_backbone_frozen(model)
    layers = model.backbone.seq_encoder.layers
    if not layers:
        raise ValueError("Spindle fine-tuning requires a Transformer layer.")
    for parameter in layers[-1].parameters():
        parameter.requires_grad = True
    trainable = sum(
        parameter.numel()
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.backbone.parameters())
    print(
        "[Spindle Unfreeze] final Transformer layer only | "
        f"trainable_backbone={trainable}/{total} "
        f"({100.0 * trainable / max(total, 1):.2f}%)"
    )


def _build_spindle_optimizer(model, cfg):
    """按睡眠分期方式为当前可训练检测头和骨干参数重建优化器。"""
    parameter_groups = []
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    if backbone_parameters:
        parameter_groups.append({
            "params": backbone_parameters,
            "lr": float(cfg.spindle_backbone_lr),
            "name": "last_transformer_stage1",
        })
    parameter_groups.append({
        "params": list(model.head_parameters()),
        "lr": float(cfg.spindle_head_lr),
        "name": "spindle_head",
    })
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(cfg.spindle_weight_decay),
    )


def _enter_second_spindle_transformer_stage(model, optimizer, cfg):
    """保持优化器状态，继续解冻倒数第二层Transformer。"""
    layers = model.backbone.seq_encoder.layers
    if len(layers) < 2:
        raise ValueError(
            "Second spindle Transformer stage requires at least two layers."
        )

    _set_spindle_backbone_frozen(model)
    for layer in layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    last_layer_parameter_ids = {
        id(parameter) for parameter in layers[-1].parameters()
    }
    last_layer_group_found = False
    for group in optimizer.param_groups:
        if any(
            id(parameter) in last_layer_parameter_ids
            for parameter in group["params"]
        ):
            group["lr"] = float(cfg.spindle_lr_last_transformer_stage2)
            group["name"] = "last_transformer_stage2"
            last_layer_group_found = True
    if not last_layer_group_found:
        raise RuntimeError(
            "The final Transformer layer is missing from the spindle optimizer."
        )

    existing_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    second_last_parameters = [
        parameter
        for parameter in layers[-2].parameters()
        if id(parameter) not in existing_parameter_ids
    ]
    if not second_last_parameters:
        raise RuntimeError(
            "The second-last Transformer layer is already in the optimizer."
        )
    optimizer.add_param_group({
        "params": second_last_parameters,
        "lr": float(cfg.spindle_lr_second_last_transformer),
        "weight_decay": float(cfg.spindle_weight_decay),
        "name": "second_last_transformer_stage2",
    })

    trainable = sum(
        parameter.numel()
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )
    total = sum(
        parameter.numel() for parameter in model.backbone.parameters()
    )
    print(
        "[Spindle Second Transformer Unfreeze] "
        f"trainable_backbone={trainable}/{total} "
        f"({100.0 * trainable / max(total, 1):.2f}%) | "
        f"last_layer_lr={float(cfg.spindle_lr_last_transformer_stage2):.2e} | "
        f"second_last_layer_lr="
        f"{float(cfg.spindle_lr_second_last_transformer):.2e} | "
        "optimizer_state=preserved"
    )


def _keep_frozen_spindle_modules_in_eval(model):
    for module in model.backbone.modules():
        direct_parameters = list(module.parameters(recurse=False))
        if direct_parameters and not any(
            parameter.requires_grad for parameter in direct_parameters
        ):
            module.eval()


def _spindle_pos_weight(records, maximum=10.0):
    """从训练被试原始标签统计类别权重，不重复计算重叠窗口中的采样点。"""
    positive = 0
    negative = 0
    for rec in records:
        labels = np.asarray(rec["point_labels"])
        valid = np.asarray(rec["point_valid_mask"]).astype(bool)
        positive += int(np.sum((labels == 1) & valid))
        negative += int(np.sum((labels == 0) & valid))
    if positive == 0:
        raise ValueError("The spindle training split contains no positive samples.")
    weight = float(np.sqrt(negative / max(positive, 1)))
    weight = float(np.clip(weight, 1.0, maximum))
    print(
        f"[Spindle Class Balance] unique points | positive={positive} | "
        f"negative={negative} | sqrt_pos_weight={weight:.4f}"
    )
    return weight


def _spindle_pos_weight_from_samples(samples, maximum=10.0):
    """Compute class weight from only the samples assigned to the train split."""
    positive = 0
    negative = 0
    for sample in samples:
        labels = np.asarray(sample["point_labels"])
        valid = np.asarray(sample["point_valid_mask"]).astype(bool)
        positive += int(np.sum((labels == 1) & valid))
        negative += int(np.sum((labels == 0) & valid))
    if positive == 0:
        raise ValueError(
            "The random-segment training split contains no positive samples."
        )
    weight = float(np.sqrt(negative / max(positive, 1)))
    weight = float(np.clip(weight, 1.0, maximum))
    print(
        f"[Spindle Class Balance] train segments only | positive={positive} | "
        f"negative={negative} | sqrt_pos_weight={weight:.4f}"
    )
    return weight


def _allocate_random_split_counts(total, fractions):
    """Allocate integer split counts while keeping non-zero requested splits."""
    total = int(total)
    fractions = np.asarray(fractions, dtype=np.float64)
    if total <= 0:
        return [0] * len(fractions)
    if np.any(fractions < 0) or not np.isclose(np.sum(fractions), 1.0):
        raise ValueError(
            f"Random-segment split fractions must sum to 1, got {fractions}."
        )
    raw = fractions * total
    counts = np.floor(raw).astype(np.int64)
    remaining = total - int(np.sum(counts))
    fractional_order = np.argsort(-(raw - counts), kind="stable")
    for idx in fractional_order[:remaining]:
        counts[idx] += 1

    requested = np.flatnonzero(fractions > 0)
    if total >= len(requested):
        for empty_idx in requested[counts[requested] == 0]:
            donors = [
                int(idx)
                for idx in requested
                if counts[idx] > 1
            ]
            if not donors:
                break
            donor = max(donors, key=lambda idx: (counts[idx], fractions[idx]))
            counts[donor] -= 1
            counts[empty_idx] += 1
    return [int(value) for value in counts]


def split_spindle_windows_randomly(windows, cfg):
    """Split continuous blocks, then keep only frames contained in each block."""
    fractions = (
        float(cfg.spindle_random_train_fraction),
        float(cfg.spindle_random_validation_fraction),
        float(cfg.spindle_random_test_fraction),
    )
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(
            "spindle_random_train/validation/test_fraction must sum to 1.0, "
            f"got {fractions}."
        )
    if not windows:
        raise ValueError("No valid spindle windows were generated.")

    block_points = int(round(
        float(cfg.spindle_random_block_seconds)
        * int(cfg.spindle_sampling_rate)
    ))
    if block_points <= 0:
        raise ValueError(
            "spindle_random_block_seconds must produce a positive block size."
        )

    windows_by_subject = defaultdict(list)
    total_points_by_subject = {}
    for frame in windows:
        subject_id = str(frame["subject_id"])
        windows_by_subject[subject_id].append(frame)
        total_points_by_subject[subject_id] = max(
            total_points_by_subject.get(subject_id, 0),
            int(frame["total_points"]),
        )

    blocks = []
    assigned_frame_keys = set()
    for subject_id in sorted(windows_by_subject):
        subject_windows = windows_by_subject[subject_id]
        total_points = int(total_points_by_subject[subject_id])
        for block_index, block_start in enumerate(
            range(0, total_points, block_points)
        ):
            block_end = min(block_start + block_points, total_points)
            contained = []
            for frame in subject_windows:
                frame_start = int(frame["start_point"])
                frame_end = frame_start + int(frame["frame_points"])
                if frame_start >= block_start and frame_end <= block_end:
                    contained.append(frame)
            if not contained:
                continue
            blocks.append({
                "subject_id": subject_id,
                "block_index": int(block_index),
                "start_point": int(block_start),
                "end_point_exclusive": int(block_end),
                "frames": contained,
            })

    if not blocks:
        raise ValueError("No continuous spindle blocks contain a complete frame.")

    rng = np.random.RandomState(int(cfg.seed))
    order = rng.permutation(len(blocks)).tolist()
    shuffled = [blocks[idx] for idx in order]
    train_count, validation_count, test_count = (
        _allocate_random_split_counts(len(shuffled), fractions)
    )
    train_end = train_count
    validation_end = train_end + validation_count
    split_blocks = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[
            validation_end:validation_end + test_count
        ],
    }
    if any(not values for values in split_blocks.values()):
        raise ValueError(
            "Random-block split produced an empty subset; adjust fractions."
        )

    split_frames = {name: [] for name in split_blocks}
    for split_name, values in split_blocks.items():
        for block in values:
            for frame in block["frames"]:
                frame["random_block_index"] = int(block["block_index"])
                frame["random_block_start_point"] = int(block["start_point"])
                frame["random_block_end_point_exclusive"] = int(
                    block["end_point_exclusive"]
                )
                split_frames[split_name].append(frame)
                assigned_frame_keys.add((
                    str(frame["subject_id"]), int(frame["start_point"])
                ))

    if any(not values for values in split_frames.values()):
        raise ValueError(
            "Random-block split produced an empty frame subset."
        )

    frame_key_sets = {
        split_name: {
            (str(frame["subject_id"]), int(frame["start_point"]))
            for frame in values
        }
        for split_name, values in split_frames.items()
    }
    if (
        frame_key_sets["train"] & frame_key_sets["validation"]
        or frame_key_sets["train"] & frame_key_sets["test"]
        or frame_key_sets["validation"] & frame_key_sets["test"]
    ):
        raise RuntimeError(
            "The same exact random window was assigned to multiple splits."
        )

    subject_ids = sorted(
        {str(frame["subject_id"]) for frame in windows}
    )
    per_subject_counts = {
        subject_id: {
            "total": 0,
            "train": 0,
            "validation": 0,
            "test": 0,
        }
        for subject_id in subject_ids
    }
    for frame in windows:
        per_subject_counts[str(frame["subject_id"])]["total"] += 1
    for split_name, values in split_frames.items():
        for frame in values:
            per_subject_counts[str(frame["subject_id"])][split_name] += 1

    coverage = {
        split_name: {
            subject_id: np.zeros(total_points, dtype=bool)
            for subject_id, total_points in total_points_by_subject.items()
        }
        for split_name in split_frames
    }
    for split_name, values in split_frames.items():
        for frame in values:
            subject_id = str(frame["subject_id"])
            start = int(frame["start_point"])
            end = start + int(frame["frame_points"])
            coverage[split_name][subject_id][start:end] = True

    overlap_summary = {}
    for left_name, right_name in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        pair_name = f"{left_name}_vs_{right_name}"
        per_subject = {
            subject_id: int(
                np.sum(
                    coverage[left_name][subject_id]
                    & coverage[right_name][subject_id]
                )
            )
            for subject_id in subject_ids
        }
        overlap_summary[pair_name] = {
            "overlapping_raw_points": int(sum(per_subject.values())),
            "overlapping_seconds": float(
                sum(per_subject.values())
                / float(cfg.spindle_sampling_rate)
            ),
            "per_subject_overlapping_raw_points": per_subject,
        }
    block_summary = {
        "block_seconds": float(cfg.spindle_random_block_seconds),
        "block_points": int(block_points),
        "block_counts": {
            name: len(values) for name, values in split_blocks.items()
        },
        "excluded_cross_boundary_frames": int(
            len(windows) - len(assigned_frame_keys)
        ),
        "blocks": {
            name: [
                {
                    "subject_id": str(block["subject_id"]),
                    "block_index": int(block["block_index"]),
                    "start_point": int(block["start_point"]),
                    "end_point_exclusive": int(
                        block["end_point_exclusive"]
                    ),
                    "start_seconds": float(
                        block["start_point"] / cfg.spindle_sampling_rate
                    ),
                    "end_seconds": float(
                        block["end_point_exclusive"]
                        / cfg.spindle_sampling_rate
                    ),
                    "frame_count": len(block["frames"]),
                }
                for block in values
            ]
            for name, values in split_blocks.items()
        },
    }
    return (
        split_frames,
        per_subject_counts,
        overlap_summary,
        block_summary,
    )


def _masked_spindle_loss(logits, labels, valid_mask, pos_weight, dice_weight):
    """只在专家确实标注的采样点上计算BCE和Dice损失。"""
    valid = valid_mask.bool() & (labels >= 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    selected_logits = logits[valid]
    selected_labels = labels[valid]
    bce = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_labels,
        pos_weight=torch.as_tensor(
            pos_weight, dtype=selected_logits.dtype, device=selected_logits.device
        ),
    )
    probabilities = torch.sigmoid(selected_logits)
    intersection = torch.sum(probabilities * selected_labels)
    dice = 1.0 - (
        (2.0 * intersection + 1.0)
        / (torch.sum(probabilities) + torch.sum(selected_labels) + 1.0)
    )
    return bce + float(dice_weight) * dice


def train_spindle_epoch(model, loader, optimizer, device, cfg, pos_weight):
    model.train()
    _keep_frozen_spindle_modules_in_eval(model)
    use_amp = device.type == "cuda" and bool(cfg.use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = 0.0
    batches = 0
    for batch in tqdm(loader, desc="Train spindle", leave=False):
        eeg = batch["eeg"].to(device)
        eog_available = batch["eog_available"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        labels = batch["point_labels"].to(device)
        point_valid = batch["point_valid_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                eeg, eog_available=eog_available, valid_mask=valid_mask
            )
            loss = _masked_spindle_loss(
                logits,
                labels,
                point_valid,
                pos_weight=pos_weight,
                dice_weight=cfg.spindle_dice_loss_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def predict_spindle_subject_probabilities(model, samples, records, cfg, device):
    """预测并按原始时间位置平均重叠窗口的概率。

    该聚合保证一个采样点即使出现在多个窗口中，最终也只保留一个概率。
    """
    loader = DataLoader(
        SpindleWindowDataset(samples),
        batch_size=cfg.spindle_batch_size,
        shuffle=False,
        num_workers=0,
    )
    sample_subject_ids = {sample["subject_id"] for sample in samples}
    record_lookup = {
        rec["subject_id"]: rec
        for rec in records
        if rec["subject_id"] in sample_subject_ids
    }
    missing_subject_ids = sample_subject_ids.difference(record_lookup)
    if missing_subject_ids:
        raise ValueError(
            "Spindle samples refer to unknown subjects: "
            f"{sorted(missing_subject_ids)}"
        )
    sums = {}
    counts = {}
    for subject_id, rec in record_lookup.items():
        total_points = int(np.prod(rec["point_labels"].shape))
        sums[subject_id] = np.zeros(total_points, dtype=np.float64)
        counts[subject_id] = np.zeros(total_points, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval spindle", leave=False):
            eeg = batch["eeg"].to(device)
            logits = model(
                eeg,
                eog_available=batch["eog_available"].to(device),
                valid_mask=batch["valid_mask"].to(device),
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for batch_idx, subject_id in enumerate(batch["subject_id"]):
                start_point = int(batch["start_point"][batch_idx].item())
                valid_len = int(batch["valid_len"][batch_idx].item())
                samples_per_epoch = probabilities.shape[-1]
                end_point = start_point + valid_len * samples_per_epoch
                values = probabilities[batch_idx, :valid_len].reshape(-1)
                sums[subject_id][start_point:end_point] += values
                counts[subject_id][start_point:end_point] += 1.0

    predictions = {}
    for subject_id, rec in record_lookup.items():
        count = counts[subject_id]
        probability = np.divide(
            sums[subject_id],
            count,
            out=np.zeros_like(sums[subject_id]),
            where=count > 0,
        )
        point_valid = (
            np.asarray(rec["point_valid_mask"]).reshape(-1).astype(bool)
            & (count > 0)
        )
        smoothed_probability = _smooth_spindle_probabilities(
            probability,
            point_valid,
            sampling_rate=int(rec["sampling_rate"]),
            smoothing_seconds=cfg.spindle_probability_smoothing_seconds,
        )
        predictions[subject_id] = {
            "probability": probability,
            "smoothed_probability": smoothed_probability,
            "labels": np.asarray(rec["point_labels"]).reshape(-1),
            "valid": point_valid,
            "original_valid": np.asarray(
                rec["point_valid_mask"]
            ).reshape(-1).astype(bool),
            "evaluation_intervals": copy.deepcopy(
                rec.get("spindle_evaluation_intervals")
            ),
            "sampling_rate": int(rec["sampling_rate"]),
        }
    return predictions


def _binary_runs(binary):
    """将逐点0/1序列转换成半开区间[start, end)。"""
    binary = np.asarray(binary, dtype=np.uint8).reshape(-1)
    padded = np.pad(binary, (1, 1), constant_values=0)
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _smooth_spindle_probabilities(
    probability,
    valid,
    sampling_rate,
    smoothing_seconds,
):
    """Apply a centered moving average without treating invalid points as zero."""
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if probability.shape != valid.shape:
        raise ValueError(
            "Spindle probability/valid shapes must match, got "
            f"{probability.shape} and {valid.shape}."
        )
    probability = np.nan_to_num(
        probability, nan=0.0, posinf=1.0, neginf=0.0
    )
    probability = np.clip(probability, 0.0, 1.0)
    window_size = max(
        1, int(round(float(smoothing_seconds) * int(sampling_rate)))
    )
    if window_size % 2 == 0:
        window_size += 1
    if window_size == 1:
        smoothed = probability.copy()
    else:
        kernel = np.ones(window_size, dtype=np.float64)
        numerator = np.convolve(
            probability * valid.astype(np.float64), kernel, mode="same"
        )
        denominator = np.convolve(
            valid.astype(np.float64), kernel, mode="same"
        )
        smoothed = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    smoothed[~valid] = 0.0
    return smoothed


def _hysteresis_spindle_runs(
    probability,
    valid,
    high_threshold,
    low_ratio,
):
    """Start on the high threshold and retain connected low-threshold support."""
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    high_threshold = float(np.clip(high_threshold, 0.0, 1.0))
    low_ratio = float(low_ratio)
    if not 0.0 <= low_ratio <= 1.0:
        raise ValueError(
            f"spindle_hysteresis_low_ratio must be in [0, 1], got {low_ratio}."
        )
    low_threshold = high_threshold * low_ratio
    high_active = (probability >= high_threshold) & valid
    low_active = (probability >= low_threshold) & valid
    return [
        (start, end)
        for start, end in _binary_runs(low_active)
        if bool(np.any(high_active[start:end]))
    ]


def _split_long_spindle_run_at_valleys(
    start,
    end,
    probability,
    minimum,
    maximum,
    valley_search_radius,
):
    """Split one long run near equally spaced probability minima."""
    start = int(start)
    end = int(end)
    length = end - start
    if maximum is None or length <= maximum:
        return [(start, end)]

    num_parts = int(np.ceil(length / float(maximum)))
    boundaries = [start]
    previous = start
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)

    for part_idx in range(1, num_parts):
        remaining_parts = num_parts - part_idx
        lower = max(
            previous + minimum,
            end - remaining_parts * maximum,
        )
        upper = min(
            previous + maximum,
            end - remaining_parts * minimum,
        )
        if lower > upper:
            raise RuntimeError(
                "Could not split a long spindle event into valid durations: "
                f"event=({start}, {end}), minimum={minimum}, maximum={maximum}."
            )

        target = int(round(start + length * part_idx / float(num_parts)))
        search_start = max(lower, target - valley_search_radius)
        search_end = min(upper, target + valley_search_radius)
        if search_start > search_end:
            search_start, search_end = lower, upper

        candidates = np.arange(search_start, search_end + 1, dtype=np.int64)
        left_values = probability[np.maximum(candidates - 1, 0)]
        right_values = probability[
            np.minimum(candidates, len(probability) - 1)
        ]
        valley_values = 0.5 * (left_values + right_values)
        minimum_value = float(np.min(valley_values))
        valley_candidates = candidates[
            np.isclose(valley_values, minimum_value, rtol=0.0, atol=1e-12)
        ]
        split_point = int(
            valley_candidates[
                np.argmin(np.abs(valley_candidates - target))
            ]
        )
        boundaries.append(split_point)
        previous = split_point

    boundaries.append(end)
    return [
        (left, right)
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]


def probabilities_to_spindle_events(
    probability,
    valid,
    threshold,
    sampling_rate,
    min_duration_seconds,
    max_duration_seconds,
    merge_gap_seconds,
    hysteresis_low_ratio=0.60,
    long_event_valley_search_seconds=0.25,
):
    """Convert smoothed probabilities into spindle event intervals.

    The high threshold starts an event, the low threshold extends it, short
    gaps are merged, short events are removed, and long events are split at
    nearby probability valleys instead of being discarded.
    """
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    runs = _hysteresis_spindle_runs(
        probability,
        valid,
        high_threshold=threshold,
        low_ratio=hysteresis_low_ratio,
    )
    max_gap = int(round(float(merge_gap_seconds) * sampling_rate))
    merged = []
    for start, end in runs:
        # SUMOv2规定“间隔小于0.1秒”才能合并，因此这里使用严格小于，
        # 恰好0.1秒的间隔不会合并。
        if merged and start - merged[-1][1] < max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    minimum = int(round(float(min_duration_seconds) * sampling_rate))
    maximum = (
        int(round(float(max_duration_seconds) * sampling_rate))
        if max_duration_seconds is not None and float(max_duration_seconds) > 0
        else None
    )
    valley_search_radius = max(
        0,
        int(
            round(
                float(long_event_valley_search_seconds) * int(sampling_rate)
            )
        ),
    )
    events = []
    for start, end in merged:
        if end - start < minimum:
            continue
        events.extend(
            _split_long_spindle_run_at_valleys(
                start,
                end,
                probability,
                minimum=minimum,
                maximum=maximum,
                valley_search_radius=valley_search_radius,
            )
        )
    return events


def _match_spindle_events(true_events, predicted_events, iou_threshold):
    """按IoU从高到低进行一对一匹配。

    一个预测事件最多匹配一个专家事件，一个专家事件也只能匹配一次。
    匹配数为TP，未匹配预测为FP，未匹配专家事件为FN。
    """
    candidates = []
    for true_idx, (true_start, true_end) in enumerate(true_events):
        for pred_idx, (pred_start, pred_end) in enumerate(predicted_events):
            intersection = max(
                0, min(true_end, pred_end) - max(true_start, pred_start)
            )
            union = max(true_end, pred_end) - min(true_start, pred_start)
            iou = intersection / union if union > 0 else 0.0
            if iou >= float(iou_threshold):
                candidates.append((iou, true_idx, pred_idx))
    used_true = set()
    used_pred = set()
    matched_ious = []
    for iou, true_idx, pred_idx in sorted(candidates, reverse=True):
        if true_idx in used_true or pred_idx in used_pred:
            continue
        used_true.add(true_idx)
        used_pred.add(pred_idx)
        matched_ious.append(float(iou))
    tp = len(matched_ious)
    return tp, len(predicted_events) - tp, len(true_events) - tp, matched_ious


def evaluate_spindle_predictions(predictions, cfg, threshold, iou_threshold):
    """在事件层面计算Precision、Recall、F1、FP/hour和匹配IoU。"""
    total_tp = total_fp = total_fn = 0
    all_ious = []
    per_subject = []
    total_valid_seconds = 0.0
    total_excluded_boundary_true_events = 0
    for subject_id, output in predictions.items():
        labels = output["labels"]
        valid = output["valid"]
        sampling_rate = output["sampling_rate"]
        evaluation_intervals = output.get("evaluation_intervals")

        if evaluation_intervals:
            original_valid = np.asarray(
                output.get("original_valid", valid)
            ).reshape(-1).astype(bool)
            full_true_events = _binary_runs(
                (labels == 1) & original_valid
            )
            true_event_count = 0
            predicted_event_count = 0
            tp = fp = fn = 0
            matched_ious = []
            valid_seconds = 0.0
            included_true_events = set()
            boundary_overlapping_true_events = set()
            for interval in evaluation_intervals:
                block_start = max(0, int(interval[0]))
                block_end = min(len(labels), int(interval[1]))
                if block_end <= block_start:
                    continue
                block_true_events = [
                    (event_start - block_start, event_end - block_start)
                    for event_start, event_end in full_true_events
                    if event_start >= block_start and event_end <= block_end
                ]
                included_true_events.update(
                    (event_start, event_end)
                    for event_start, event_end in full_true_events
                    if event_start >= block_start and event_end <= block_end
                )
                boundary_overlapping_true_events.update(
                    (event_start, event_end)
                    for event_start, event_end in full_true_events
                    if event_start < block_end and event_end > block_start
                )
                block_valid = valid[block_start:block_end]
                block_probability = output["probability"][
                    block_start:block_end
                ]
                block_smoothed = _smooth_spindle_probabilities(
                    block_probability,
                    block_valid,
                    sampling_rate=sampling_rate,
                    smoothing_seconds=(
                        cfg.spindle_probability_smoothing_seconds
                    ),
                )
                block_predicted_events = probabilities_to_spindle_events(
                    block_smoothed,
                    block_valid,
                    threshold=threshold,
                    sampling_rate=sampling_rate,
                    min_duration_seconds=cfg.spindle_min_duration_seconds,
                    max_duration_seconds=cfg.spindle_max_duration_seconds,
                    merge_gap_seconds=cfg.spindle_merge_gap_seconds,
                    hysteresis_low_ratio=cfg.spindle_hysteresis_low_ratio,
                    long_event_valley_search_seconds=(
                        cfg.spindle_long_event_valley_search_seconds
                    ),
                )
                block_tp, block_fp, block_fn, block_ious = (
                    _match_spindle_events(
                        block_true_events,
                        block_predicted_events,
                        iou_threshold,
                    )
                )
                true_event_count += len(block_true_events)
                predicted_event_count += len(block_predicted_events)
                tp += block_tp
                fp += block_fp
                fn += block_fn
                matched_ious.extend(block_ious)
                valid_seconds += float(np.sum(block_valid)) / sampling_rate
            excluded_boundary_true_events = int(
                len(
                    boundary_overlapping_true_events
                    - included_true_events
                )
            )
        else:
            true_events = _binary_runs((labels == 1) & valid)
            smoothed_probability = output.get("smoothed_probability")
            if smoothed_probability is None:
                smoothed_probability = _smooth_spindle_probabilities(
                    output["probability"],
                    valid,
                    sampling_rate=sampling_rate,
                    smoothing_seconds=(
                        cfg.spindle_probability_smoothing_seconds
                    ),
                )
            predicted_events = probabilities_to_spindle_events(
                smoothed_probability,
                valid,
                threshold=threshold,
                sampling_rate=sampling_rate,
                min_duration_seconds=cfg.spindle_min_duration_seconds,
                max_duration_seconds=cfg.spindle_max_duration_seconds,
                merge_gap_seconds=cfg.spindle_merge_gap_seconds,
                hysteresis_low_ratio=cfg.spindle_hysteresis_low_ratio,
                long_event_valley_search_seconds=(
                    cfg.spindle_long_event_valley_search_seconds
                ),
            )
            tp, fp, fn, matched_ious = _match_spindle_events(
                true_events, predicted_events, iou_threshold
            )
            true_event_count = len(true_events)
            predicted_event_count = len(predicted_events)
            valid_seconds = float(np.sum(valid)) / sampling_rate
            excluded_boundary_true_events = 0

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        total_valid_seconds += valid_seconds
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_excluded_boundary_true_events += (
            excluded_boundary_true_events
        )
        all_ious.extend(matched_ious)
        per_subject.append({
            "subject_id": subject_id,
            "true_events": true_event_count,
            "predicted_events": predicted_event_count,
            "excluded_boundary_true_events": (
                excluded_boundary_true_events
            ),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "valid_seconds": valid_seconds,
        })
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "hysteresis_low_threshold": float(
            threshold * cfg.spindle_hysteresis_low_ratio
        ),
        "iou_threshold": float(iou_threshold),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fp_per_hour": float(total_fp / max(total_valid_seconds / 3600.0, 1e-12)),
        "mean_matched_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "excluded_boundary_true_events": int(
            total_excluded_boundary_true_events
        ),
        "per_subject": per_subject,
    }





def _build_spindle_model(cfg, device):
    backbone = _make_spindle_backbone(cfg)
    maybe_load_eeg_pretrained_weights(backbone, cfg)
    model = SpindleTemporalDetectionModel(
        backbone,
        head_channels=cfg.spindle_head_channels,
        num_layers=cfg.spindle_head_layers,
        dropout=cfg.spindle_head_dropout,
    ).to(device)
    return model





class SpindleResumeConfigMismatch(ValueError):
    """Refuse to mix completed folds from different experiment settings."""


def _load_completed_spindle_fold(fold_dir, fold_idx, train_ids, test_ids, cfg):
    """Validate and reuse a finished fold, without rebuilding its sample arrays."""
    metrics_path = os.path.join(fold_dir, "metrics.json")
    checkpoint_path = os.path.join(fold_dir, "final_spindle_model.pt")
    with open(metrics_path, encoding="utf-8") as handle:
        result = json.load(handle)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for saved in (result, checkpoint):
        split = saved["split"]
        if (split.get("protocol") != cfg.spindle_split_protocol
                or split.get("threshold_selection_uses_test_labels") is not False
                or split.get("plateau_monitor") != "training_loss"
                or saved.get("fixed_probability_threshold") != float(cfg.spindle_probability_threshold)):
            raise ValueError(f"Saved fold {fold_idx} uses an old test-selected protocol; retrain.")
        if (int(split["fold"]) != fold_idx or split["train_subjects"] != train_ids
                or split["test_subjects"] != test_ids or int(split["split_seed"]) != int(cfg.seed)
                or str(split["expert"]).upper() != str(cfg.spindle_expert).upper()
                or int(saved["final_epoch"]) != int(cfg.spindle_epochs)):
            raise SpindleResumeConfigMismatch(f"Saved fold {fold_idx} split/seed/expert/epochs do not match current run")
    mapping = {
        "sampling_rate": "spindle_sampling_rate", "backbone_position_seq_len": "spindle_seq_len",
        "frame_seconds": "spindle_subject_frame_seconds", "frame_overlap_fraction": "spindle_subject_frame_overlap",
        "primary_iou_threshold": "spindle_primary_iou_threshold",
        "min_duration_seconds": "spindle_min_duration_seconds", "max_duration_seconds": "spindle_max_duration_seconds",
        "merge_gap_seconds": "spindle_merge_gap_seconds", "probability_smoothing_seconds": "spindle_probability_smoothing_seconds",
        "hysteresis_low_ratio": "spindle_hysteresis_low_ratio", "long_event_valley_search_seconds": "spindle_long_event_valley_search_seconds",
        "probability_threshold": "spindle_probability_threshold", "head_learning_rate": "spindle_head_lr",
        "freeze_backbone_epochs": "spindle_freeze_backbone_epochs",
        "plateau_transition_patience": "spindle_plateau_transition_patience",
        "plateau_second_unfreeze_enabled": "spindle_enable_plateau_second_transformer_stage",
        "last_transformer_stage1_lr": "spindle_backbone_lr",
        "last_transformer_stage2_lr": "spindle_lr_last_transformer_stage2",
        "second_last_transformer_lr": "spindle_lr_second_last_transformer",
        "n2_only_pointwise_segmentation": "spindle_n2_only_segmentation",
    }
    for saved_key, attr in mapping.items():
        if checkpoint["config"][saved_key] != getattr(cfg, attr):
            raise SpindleResumeConfigMismatch(f"Saved fold {fold_idx} setting mismatch: {saved_key}")
    if not checkpoint.get("model_state_dict") or checkpoint["test_metrics"] != result["primary_test"]:
        raise ValueError(f"Incomplete or inconsistent saved fold {fold_idx}")
    del checkpoint
    gc.collect()
    return result


def _try_resume_spindle_fold(fold_dir, fold_idx, train_ids, test_ids, cfg):
    if not bool(getattr(cfg, "spindle_auto_resume", True)):
        return None
    if os.path.exists(os.path.join(fold_dir, "training_in_progress.json")):
        print(f"[Spindle Resume] interrupted fold {fold_idx}; restart from epoch 1", flush=True)
        return None
    required = ("metrics.json", "final_spindle_model.pt")
    if not all(os.path.isfile(os.path.join(fold_dir, name)) for name in required):
        return None
    try:
        return _load_completed_spindle_fold(fold_dir, fold_idx, train_ids, test_ids, cfg)
    except SpindleResumeConfigMismatch:
        raise
    except (OSError, EOFError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError) as error:
        print(f"[Spindle Resume] incomplete/unreadable fold {fold_idx}: {error}; restart from epoch 1", flush=True)
        gc.collect()
        return None


def spindle_event_cv_run(records, cfg, device):
    """Run subject LOSO with a fixed probability threshold and final-epoch checkpoint."""
    if len(records) < 2:
        raise ValueError("Spindle event detection requires at least two subjects.")
    sampling_rates = sorted({int(rec["sampling_rate"]) for rec in records})
    if sampling_rates != [int(cfg.spindle_sampling_rate)]:
        raise ValueError(
            f"Expected spindle sampling rate {cfg.spindle_sampling_rate}, "
            f"got {sampling_rates}. Re-run the spindle preprocessor."
        )

    split_mode = str(
        getattr(cfg, "spindle_split_mode", "subject")
    ).strip().lower()
    if split_mode not in {"subject", "subject_loso", "loso"}:
        raise ValueError(
            "spindle_split_mode must be subject LOSO, "
            f"got {split_mode!r}."
        )

    output_dir = os.path.join(
        cfg.spindle_output_root,
        "spindle_detection",
        "soushizhe",
        f"{cfg.spindle_experiment_name}_{str(cfg.spindle_expert).lower()}",
    )
    if not 0.0 <= float(cfg.spindle_probability_threshold) <= 1.0:
        raise ValueError("spindle_probability_threshold must be between 0 and 1.")
    if int(cfg.spindle_epochs) < 1:
        raise ValueError("spindle_epochs must be at least 1.")
    ensure_dir(output_dir)
    split_protocol = str(
        getattr(
            cfg,
            "spindle_split_protocol",
            "leave_one_subject_out_fixed_threshold_train_loss_plateau",
        )
    ).lower()
    if (
        split_protocol
        != "leave_one_subject_out_fixed_threshold_train_loss_plateau"
    ):
        raise ValueError(
            f"Unknown spindle_split_protocol: {split_protocol!r}."
        )

    # The seed controls fold order only. Every subject is held out exactly once.
    indices = np.arange(len(records))
    subject_order = np.random.RandomState(cfg.seed).permutation(indices).tolist()
    fold_splits = []
    for heldout_subject_idx in subject_order:
        heldout_idx = [int(heldout_subject_idx)]
        train_idx = [
            int(subject_idx)
            for subject_idx in subject_order
            if int(subject_idx) != int(heldout_subject_idx)
        ]
        if len(train_idx) != len(records) - 1 or len(heldout_idx) != 1:
            raise RuntimeError(
                "Internal LOSO split error: expected all-but-one training "
                f"subjects and one held-out subject, got "
                f"{len(train_idx)}/{len(heldout_idx)}."
            )
        fold_splits.append((train_idx, heldout_idx))

    n_splits = len(fold_splits)
    fold_results = []

    for fold_idx, (train_idx, heldout_idx) in enumerate(
        fold_splits, start=1
    ):
        train_records = [records[idx] for idx in train_idx]
        heldout_records = [records[idx] for idx in heldout_idx]
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx:02d}")
        completed = _try_resume_spindle_fold(
            fold_dir, fold_idx, [r["subject_id"] for r in train_records],
            [r["subject_id"] for r in heldout_records], cfg)
        if completed is not None:
            fold_results.append(completed)
            print(f"[Spindle Resume] reuse completed fold {fold_idx}/{n_splits}", flush=True)
            continue
        ensure_dir(fold_dir)
        progress_path = os.path.join(fold_dir, "training_in_progress.json")
        with open(progress_path, "w", encoding="utf-8") as handle:
            json.dump({"fold": fold_idx, "status": "training", "restart_policy": "epoch_1"}, handle)
        print(f"[Spindle Resume] train fold {fold_idx}/{n_splits} from epoch 1", flush=True)
        train_samples = build_spindle_point_frames(
            train_records,
            frame_seconds=cfg.spindle_subject_frame_seconds,
            overlap=cfg.spindle_subject_frame_overlap,
        )
        heldout_samples = build_spindle_point_frames(
            heldout_records,
            frame_seconds=cfg.spindle_subject_frame_seconds,
            overlap=cfg.spindle_subject_frame_overlap,
        )
        if not train_samples or not heldout_samples:
            raise ValueError(
                f"Fold {fold_idx} produced an empty train/held-out sample set."
            )
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx:02d}")
        ensure_dir(fold_dir)
        split_info = {
            "fold": fold_idx,
            "expert": str(cfg.spindle_expert).upper(),
            "protocol": (
                "leave_one_subject_out_fixed_threshold_train_loss_plateau"
            ),
            "split_seed": int(cfg.seed),
            "subject_order": [
                records[idx]["subject_id"] for idx in subject_order
            ],
            "train_subjects": [rec["subject_id"] for rec in train_records],
            "test_subjects": [
                rec["subject_id"] for rec in heldout_records
            ],
            "counts": {"train": len(train_records), "test": 1},
            "frame_counts": {
                "train": len(train_samples),
                "test": len(heldout_samples),
            },
            "frame_seconds": float(cfg.spindle_subject_frame_seconds),
            "frame_stride_seconds": float(
                cfg.spindle_subject_frame_seconds
                * (1.0 - cfg.spindle_subject_frame_overlap)
            ),
            "frame_overlap_fraction": float(
                cfg.spindle_subject_frame_overlap
            ),
            "separate_validation_set": False,
            "early_stopping": False,
            "plateau_transition_patience": int(
                cfg.spindle_plateau_transition_patience
            ),
            "best_epoch_selection": False,
            "fixed_training_epochs": int(cfg.spindle_epochs),
            "freeze_backbone_epochs": int(
                cfg.spindle_freeze_backbone_epochs
            ),
            "plateau_second_transformer_stage": bool(
                cfg.spindle_enable_plateau_second_transformer_stage
            ),
            "probability_threshold": float(cfg.spindle_probability_threshold),
            "plateau_monitor": "training_loss",
            "threshold_selection": "fixed_before_training",
            "threshold_selection_uses_test_labels": False,
            "probability_smoothing_seconds": float(
                cfg.spindle_probability_smoothing_seconds
            ),
            "hysteresis_low_ratio": float(
                cfg.spindle_hysteresis_low_ratio
            ),
            "long_event_policy": "split_at_probability_valley",
            "long_event_valley_search_seconds": float(
                cfg.spindle_long_event_valley_search_seconds
            ),
            "checkpoint_selection": "none_fixed_final_epoch",
            "checkpoint_selection_uses_test_labels": False,
            "result_is_oracle_upper_bound": False,
        }
        with open(
            os.path.join(fold_dir, "split.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(split_info, handle, ensure_ascii=False, indent=2)

        print("\n" + "=" * 80)
        print(
            f"[Spindle Fold] {fold_idx}/{n_splits} | expert={cfg.spindle_expert} | "
            f"train={split_info['train_subjects']} | "
            f"test={split_info['test_subjects']} | "
            f"frames={split_info['frame_counts']} | "
            f"frame={cfg.spindle_subject_frame_seconds:.2f}s | "
            f"overlap={cfg.spindle_subject_frame_overlap:.0%} | "
            f"epochs={cfg.spindle_epochs} | "
            f"fixed probability threshold={cfg.spindle_probability_threshold:.2f}"
        )
        set_seed(cfg.seed + fold_idx)
        model = _build_spindle_model(cfg, device)
        _set_spindle_backbone_frozen(model)
        optimizer = _build_spindle_optimizer(model, cfg)
        pos_weight = _spindle_pos_weight(
            train_records, maximum=cfg.spindle_pos_weight_max
        )
        train_loader = DataLoader(
            SpindleWindowDataset(train_samples),
            batch_size=cfg.spindle_batch_size,
            shuffle=True,
            num_workers=0,
        )
        history = []
        best_train_loss = float("inf")
        patience_counter = 0
        second_stage_started = False
        finetune_phase = "head_only"

        for epoch in range(1, int(cfg.spindle_epochs) + 1):
            if epoch == int(cfg.spindle_freeze_backbone_epochs) + 1:
                _unfreeze_final_spindle_transformer_layer(model)
                optimizer = _build_spindle_optimizer(model, cfg)
                finetune_phase = "last_transformer"
                best_train_loss = float("inf")
                patience_counter = 0
            train_loss = float(train_spindle_epoch(
                model, train_loader, optimizer, device, cfg, pos_weight
            ))
            improved = train_loss < best_train_loss
            if improved:
                best_train_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            history.append({
                "epoch": epoch, "train_loss": train_loss,
                "finetune_phase": finetune_phase,
                "plateau_monitor": "training_loss",
                "patience_counter": int(patience_counter),
            })
            print(
                f"[Spindle LOSO Train] fold={fold_idx} | epoch={epoch}/{cfg.spindle_epochs} | "
                f"phase={finetune_phase} | loss={train_loss:.5f}"
            )
            if (
                bool(cfg.spindle_enable_plateau_second_transformer_stage)
                and not second_stage_started
                and finetune_phase == "last_transformer"
                and patience_counter >= int(cfg.spindle_plateau_transition_patience)
                and epoch < int(cfg.spindle_epochs)
            ):
                _enter_second_spindle_transformer_stage(model, optimizer, cfg)
                second_stage_started = True
                finetune_phase = "last_two_transformers"
                best_train_loss = float("inf")
                patience_counter = 0
                print(f"[Spindle Train-Loss Plateau Transition] fold={fold_idx} | epoch={epoch}")

        # No test scoring occurs until all fixed training epochs are complete.
        final_heldout_threshold = float(cfg.spindle_probability_threshold)
        final_heldout_predictions = predict_spindle_subject_probabilities(
            model, heldout_samples, heldout_records, cfg, device
        )
        final_state = copy.deepcopy(model.state_dict())
        iou_results = {}
        for iou_threshold in cfg.spindle_report_iou_thresholds:
            result = evaluate_spindle_predictions(
                final_heldout_predictions,
                cfg,
                threshold=final_heldout_threshold,
                iou_threshold=iou_threshold,
            )
            iou_results[f"{float(iou_threshold):.2f}"] = result
        primary = iou_results[f"{float(cfg.spindle_primary_iou_threshold):.2f}"]
        checkpoint_path = os.path.join(fold_dir, "final_spindle_model.pt")
        torch.save({
            "model_state_dict": final_state,
            "final_epoch": int(cfg.spindle_epochs),
            "fixed_probability_threshold": final_heldout_threshold,
            "test_metrics": primary,
            "selection_policy": (
                "fixed_final_epoch_with_training_loss_plateau_finetuning"
            ),
            "expert": str(cfg.spindle_expert).upper(),
            "split": split_info,
            "config": {
                "sampling_rate": cfg.spindle_sampling_rate,
                "backbone_position_seq_len": cfg.spindle_seq_len,
                "frame_sequence_length": 1,
                "frame_seconds": float(
                    cfg.spindle_subject_frame_seconds
                ),
                "frame_stride_seconds": float(
                    cfg.spindle_subject_frame_seconds
                    * (1.0 - cfg.spindle_subject_frame_overlap)
                ),
                "frame_overlap_fraction": float(
                    cfg.spindle_subject_frame_overlap
                ),
                "fixed_training_epochs": int(cfg.spindle_epochs),
                "freeze_backbone_epochs": int(
                    cfg.spindle_freeze_backbone_epochs
                ),
                "first_unfreeze_scope": "final_transformer_layer_only",
                "plateau_second_unfreeze_enabled": bool(
                    cfg.spindle_enable_plateau_second_transformer_stage
                ),
                "second_unfreeze_scope": (
                    "last_two_transformer_layers"
                ),
                "plateau_transition_patience": int(
                    cfg.spindle_plateau_transition_patience
                ),
                "head_learning_rate": float(cfg.spindle_head_lr),
                "last_transformer_stage1_lr": float(
                    cfg.spindle_backbone_lr
                ),
                "last_transformer_stage2_lr": float(
                    cfg.spindle_lr_last_transformer_stage2
                ),
                "second_last_transformer_lr": float(
                    cfg.spindle_lr_second_last_transformer
                ),
                "probability_threshold": float(cfg.spindle_probability_threshold),
                "plateau_monitor": "training_loss",
                "threshold_selection": (
                    "fixed_before_training"
                ),
                "threshold_selection_uses_test_labels": False,
                "checkpoint_selection": (
                    "none_fixed_final_epoch"
                ),
                "checkpoint_selection_uses_test_labels": False,
                "result_is_oracle_upper_bound": False,
                "separate_validation_set": False,
                "early_stopping": False,
                "primary_iou_threshold": cfg.spindle_primary_iou_threshold,
                "min_duration_seconds": cfg.spindle_min_duration_seconds,
                "max_duration_seconds": cfg.spindle_max_duration_seconds,
                "merge_gap_seconds": cfg.spindle_merge_gap_seconds,
                "merge_gap_comparison": "strictly_less_than",
                "probability_smoothing_seconds": float(
                    cfg.spindle_probability_smoothing_seconds
                ),
                "threshold_method": "high_low_hysteresis",
                "hysteresis_low_ratio": float(
                    cfg.spindle_hysteresis_low_ratio
                ),
                "long_event_policy": "split_at_probability_valley",
                "long_event_valley_search_seconds": float(
                    cfg.spindle_long_event_valley_search_seconds
                ),
                "n2_only_pointwise_segmentation": bool(
                    cfg.spindle_n2_only_segmentation
                ),
                "event_metric": "IoU-F1",
            },
        }, checkpoint_path + ".tmp")
        os.replace(checkpoint_path + ".tmp", checkpoint_path)
        fold_result = {
            "fold": fold_idx,
            "final_epoch": int(cfg.spindle_epochs),
            "second_transformer_stage_started": bool(
                second_stage_started
            ),
            "fixed_probability_threshold": final_heldout_threshold,
            "test_by_iou": iou_results,
            "primary_test": primary,
            "split": split_info,
            "checkpoint": checkpoint_path,
            "history": history,
        }
        fold_results.append(fold_result)
        with open(
            os.path.join(fold_dir, "metrics.json.tmp"), "w", encoding="utf-8"
        ) as handle:
            json.dump(fold_result, handle, ensure_ascii=False, indent=2)
        os.replace(os.path.join(fold_dir, "metrics.json.tmp"), os.path.join(fold_dir, "metrics.json"))
        os.remove(progress_path)
        print(
            f"[Spindle LOSO Final] fold={fold_idx} | "
            f"epoch={cfg.spindle_epochs} | "
            f"fixed_high/low_threshold="
            f"{final_heldout_threshold:.2f}/"
            f"{primary['hysteresis_low_threshold']:.2f} | "
            f"IoU={cfg.spindle_primary_iou_threshold:.2f} | "
            f"IoU-F1={primary['f1']:.4f} | P={primary['precision']:.4f} | "
            f"R={primary['recall']:.4f} | FP/h={primary['fp_per_hour']:.2f}"
        )
        # Dataset/loader retain views of the stacked full-night arrays. Release
        # them before allocating the next fold; deleting only model is insufficient.
        del train_loader, train_samples, heldout_samples
        del optimizer, model, final_state, final_heldout_predictions
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    primary_results = [item["primary_test"] for item in fold_results]
    summary = {
        "task": "spindle_event_detection",
        "dataset": cfg.spindle_dataset_name,
        "expert": str(cfg.spindle_expert).upper(),
        "protocol": (
            f"{n_splits}-fold leave-one-subject-out evaluation; "
            f"{cfg.spindle_subject_frame_seconds:g}-second frames with "
            f"{cfg.spindle_subject_frame_overlap:.0%} overlap inside each "
            "subject; "
            f"staged fine-tuning: {cfg.spindle_freeze_backbone_epochs} head-only epochs, "
            "then the final Transformer layer, then the second-last layer "
            "after a training-loss plateau; training always continues to the fixed "
            "final epoch without best-epoch selection or early stopping; "
            f"fixed probability threshold={cfg.spindle_probability_threshold:.2f}; "
            f"IoU-F1@{cfg.spindle_primary_iou_threshold:.2f}; "
            "smooth probabilities over 0.10s; "
            "high/low hysteresis thresholding; remove events shorter than "
            f"{cfg.spindle_min_duration_seconds:g}s; split events longer than "
            f"{cfg.spindle_max_duration_seconds:g}s at probability valleys; "
            "merge gap <0.1s; "
            "point-wise segmentation inside continuous N2"
        ),
        "fixed_training_epochs": int(cfg.spindle_epochs),
        "frame_seconds": float(cfg.spindle_subject_frame_seconds),
        "frame_stride_seconds": float(
            cfg.spindle_subject_frame_seconds
            * (1.0 - cfg.spindle_subject_frame_overlap)
        ),
        "frame_overlap_fraction": float(
            cfg.spindle_subject_frame_overlap
        ),
        "freeze_backbone_epochs": int(
            cfg.spindle_freeze_backbone_epochs
        ),
        "first_unfreeze_scope": "final_transformer_layer_only",
        "plateau_second_unfreeze_enabled": bool(
            cfg.spindle_enable_plateau_second_transformer_stage
        ),
        "second_unfreeze_scope": "last_two_transformer_layers",
        "plateau_transition_patience": int(
            cfg.spindle_plateau_transition_patience
        ),
        "head_learning_rate": float(cfg.spindle_head_lr),
        "last_transformer_stage1_lr": float(
            cfg.spindle_backbone_lr
        ),
        "last_transformer_stage2_lr": float(
            cfg.spindle_lr_last_transformer_stage2
        ),
        "second_last_transformer_lr": float(
            cfg.spindle_lr_second_last_transformer
        ),
        "probability_threshold": float(cfg.spindle_probability_threshold),
        "plateau_monitor": "training_loss",
        "threshold_selection": "fixed_before_training",
        "threshold_selection_uses_test_labels": False,
        "probability_smoothing_seconds": float(
            cfg.spindle_probability_smoothing_seconds
        ),
        "threshold_method": "high_low_hysteresis",
        "hysteresis_low_ratio": float(
            cfg.spindle_hysteresis_low_ratio
        ),
        "long_event_policy": "split_at_probability_valley",
        "long_event_valley_search_seconds": float(
            cfg.spindle_long_event_valley_search_seconds
        ),
        "checkpoint_selection": "none_fixed_final_epoch",
        "checkpoint_selection_uses_test_labels": False,
        "result_is_oracle_upper_bound": False,
        "separate_validation_set": False,
        "early_stopping": False,
        "best_epoch_selection": False,
        "primary_iou_threshold": float(cfg.spindle_primary_iou_threshold),
        "primary_metric": (
            f"IoU-F1@{cfg.spindle_primary_iou_threshold:.2f}"
        ),
        "mean_fold_f1": float(np.mean([x["f1"] for x in primary_results])),
        "mean_fold_iou_f1": float(np.mean([x["f1"] for x in primary_results])),
        "std_fold_f1": float(np.std([x["f1"] for x in primary_results])),
        "mean_fold_precision": float(
            np.mean([x["precision"] for x in primary_results])
        ),
        "mean_fold_recall": float(
            np.mean([x["recall"] for x in primary_results])
        ),
        "mean_fold_fp_per_hour": float(
            np.mean([x["fp_per_hour"] for x in primary_results])
        ),
        "folds": fold_results,
    }
    with open(
        os.path.join(output_dir, "spindle_cv_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        "\n[Spindle LOSO CV Complete] "
        f"expert={summary['expert']} | "
        f"IoU-F1@{cfg.spindle_primary_iou_threshold:.2f}="
        f"{summary['mean_fold_f1']:.4f}+-{summary['std_fold_f1']:.4f} | "
        f"P={summary['mean_fold_precision']:.4f} | "
        f"R={summary['mean_fold_recall']:.4f} | "
        f"FP/h={summary['mean_fold_fp_per_hour']:.2f}"
    )
    return summary


def main():
    cfg = FinetuneConfig()

    set_seed(cfg.seed)
    # torch.use_deterministic_algorithms(True)

    ensure_dir(cfg.output_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stage_records = None

    # ==============================
    # 下游任务 1：睡眠分期
    # ==============================
    saved_stage_already_ran = False
    if cfg.run_sleep_stage_classification:
        stage_mode = str(
            getattr(cfg, "sleep_stage_run_mode", "train")
        ).lower()
        print(f"\n[Run] sleep_stage_classification | mode={stage_mode}")
        print(
            "[Stage Backbone] source="
            f"{getattr(cfg, 'stage_backbone_checkpoint_source', 'event')} | "
            "generation_model="
            f"{os.path.abspath(cfg.generation_full_model_path)} | "
            "output_dir="
            f"{task_output_dir(cfg, 'sleep_stage_classification')}"
        )
        stage_records = load_subject_records_from_dirs(cfg.stage_train_dirs)
        if stage_mode == "saved_repro":
            saved_stage_loso_test_run(stage_records, cfg, device)
            saved_stage_already_ran = True
        elif stage_mode == "train":
            stage_subject_loso_cv_run(stage_records, cfg, device)
        else:
            raise ValueError(
                "sleep_stage_run_mode must be 'saved_repro' or 'train'; "
                f"got {stage_mode!r}."
            )
    else:
        print("\n[Skip] sleep_stage_classification")

    if (
        getattr(cfg, "run_saved_sleep_stage_test", False)
        and not saved_stage_already_ran
    ):
        print("\n[Run] saved_sleep_stage_test (test only)")
        if stage_records is None:
            stage_records = load_subject_records_from_dirs(cfg.stage_train_dirs)
        saved_stage_loso_test_run(stage_records, cfg, device)
    elif not saved_stage_already_ran:
        print("\n[Skip] saved_sleep_stage_test")

    # ==============================
    # 下游任务 2：睡眠分期生成增强
    # ==============================
    if cfg.run_stage_generation_augmentation:
        print("\n[Run] stage_generation_augmentation")
        generation_stage_records = load_subject_records_from_dirs(
            cfg.generation_stage_train_dirs
        )
        stage_generation_augmentation_task_run(
            generation_stage_records, cfg, device
        )
    else:
        print("\n[Skip] stage_generation_augmentation")

    # ==============================
    # 下游任务 3：外部泛化测试
    # ==============================
    if cfg.run_stage_external_generalization:
        print("\n[Run] stage_external_generalization")

        if stage_records is None:
            stage_records = load_subject_records_from_dirs(cfg.stage_train_dirs)

        external_records = load_subject_records_from_dirs(cfg.stage_external_test_dirs)
        external_samples = build_stage_samples(
            external_records,
            seq_len=cfg.seq_len,
            stride=cfg.stride
        )
        stage_external_generalization_run(stage_records, external_samples, cfg, device)
    else:
        print("\n[Skip] stage_external_generalization")

    # ==============================
    # 下游任务 4：睡眠障碍三分类（CAP + DREAMS HC）
    # ==============================
    if cfg.run_sleep_disorder_classification:
        print("\n[Run] sleep_disorder_classification")
        disorder_records = load_disorder_training_records(cfg)
        save_disorder_cohort_manifest(disorder_records, cfg)
        disorder_subject_cv_run_model_b(disorder_records, cfg, device)
    else:
        print("\n[Skip] sleep_disorder_classification")

    # ==============================
    # 下游任务 5：慢波检测
    # ==============================
    if cfg.run_slow_wave_detection:
        print("\n[Run] slow_wave_detection")
        slow_wave_records = load_subject_records_from_dirs(cfg.slow_wave_train_dirs)
        slow_wave_subject_loso_cv_run(slow_wave_records, cfg, device)
    else:
        print("\n[Skip] slow_wave_detection")

    # ==============================
    # Downstream task 6: expert spindle event detection
    # ==============================
    if cfg.run_spindle_detection:
        print(
            f"\n[Run] spindle_event_detection | expert={cfg.spindle_expert} | "
            f"split_mode={cfg.spindle_split_mode}"
        )

        spindle_records = load_clas_spindle_records(cfg)
        spindle_event_cv_run(spindle_records, cfg, device)
    else:
        print("\n[Skip] spindle_event_detection")


if __name__ == "__main__":
    main()
