"""
sleep_isruc_s1_preprocessor_morphology.py - ISRUC_S1 形态学检测版
专门针对 ISRUC_S1 数据集的预处理，使用形态学波形检测替代特征聚类。
输出为自然语言描述，供后续大语言模型(LLM)进行上游预训练使用。
不参与下游任务。
"""

import os
import glob
import numpy as np
import json
import pandas as pd
import warnings
import gc
import re
import shutil
import tempfile
from pathlib import Path
import mne
import yasa

from morphology_detector_yasa import MorphologyDetector


# 配置参数 (移除了聚类相关参数，调整了描述模板)
class PreprocessConfig:
    """预处理配置参数类"""

    def __init__(self):
        # EEG处理参数
        self.target_fs = 100  # 目标采样率
        self.epoch_duration = 30  # 每个epoch的秒数
        self.label_csv = None

        # 文本描述模板 (针对单个epoch的实时描述)
        # self.text_templates = {
        #     'spindle_kcomplex': "纺锤波与K-复合波共存，是典型的N2睡眠期特征。",
        #     'spindle_only': "出现明显的睡眠纺锤波，提示N2期睡眠。",
        #     'kcomplex_only': "检测到K-复合波，常见于N2睡眠期。",
        #     'slow_wave_rich': "慢波活动丰富，是深度睡眠(N3期)的标志。",
        #     'alpha_prominent': "Alpha节律显著，为安静清醒状态特征。",
        #     'vertex_present': "出现顶尖波，多见于睡眠起始的N1期。",
        #     'rem_detected': "伴有快速眼动，是REM睡眠期的特征。",
        #     'no_significant_wave': "未检测到显著的特征性脑电波形。"
        # }
        self.text_templates = {
        "spindle_kcomplex": "检测到纺锤波与K复合波共存。",
        "spindle_only": "检测到明显纺锤波活动。",
        "kcomplex_only": "检测到K复合波活动。",
        "slow_wave_rich": "检测到丰富的慢波活动。",
        "alpha_prominent": "检测到显著α节律。",
        "vertex_present": "检测到顶尖波活动。",
        "rem_detected": "检测到快速眼动事件。",
        "no_significant_wave": "未检测到显著特征性事件。"
        }

def find_psg_files(data_path):
    # 收集 ISRUC_S1 的 REC 主文件。
    """
    递归查找 ISRUC_S1 数据集中的 PSG 文件
    例如: D:/dataset/ISRUC_S1/9/9/9.rec
    """
    psg_files = []
    for root, _, files in os.walk(data_path):
        for fname in files:
            if fname.lower().endswith('.rec'):
                psg_files.append(os.path.join(root, fname))

    psg_files = sorted(list(set(psg_files)))

    print(f"Found {len(psg_files)} PSG files in ISRUC_S1:")
    for f in psg_files[:5]:
        print(f"  {f}")
    if len(psg_files) > 5:
        print(f"  ... and {len(psg_files) - 5} more")

    return psg_files


def read_raw_edf_or_rec(file_path, preload=True, verbose=False):
    # 优先直接读取 .rec；若后端因扩展名兼容性失败，再临时复制为 .edf 重试。
    # 这里的 fallback 只是字节级复制并改扩展名，不是信号格式转换。
    try:
        return mne.io.read_raw_edf(file_path, preload=preload, verbose=verbose)
    except Exception as e1:
        if not str(file_path).lower().endswith('.rec'):
            raise

        tmp_dir = tempfile.mkdtemp(prefix='isruc_rec_tmp_')
        tmp_edf = os.path.join(tmp_dir, Path(file_path).stem + '.edf')
        try:
            shutil.copyfile(file_path, tmp_edf)
            raw = mne.io.read_raw_edf(tmp_edf, preload=preload, verbose=verbose)
            if preload:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                # Keep the temporary source alive until load_data() finishes.
                raw._isruc_temporary_source_dir = tmp_dir
            return raw
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise e1
def pick_channel(ch_names, candidates):
    # 按候选优先级选通道。
    for ch in candidates:
        if ch in ch_names:
            return ch
    return None


def stage_str_to_int(stage):
    # 分期文本 -> {0,1,2,3,4,-1}。
    stage = str(stage).upper()
    mapping = {
        "WAKE": 0, "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4, "R": 4
    }
    return mapping.get(stage, -1)
def normalize_stage_str(stage):
    # 统一到 W/N1/N2/N3/REM。
    s = str(stage).strip().upper()
    mapping = {
        "WAKE": "W",
        "W": "W",
        "N1": "N1",
        "N2": "N2",
        "N3": "N3",
        "REM": "REM",
        "R": "REM"
    }
    return mapping.get(s, s)


def extract_epoch_signal_array(raw, ch_name, epoch_duration=30, target_n_epochs=None):
    # 连续 EEG -> {n_epochs, 3000} float32。
    """
    从连续 raw 中提取单通道 30 秒 EEG epoch，并尽量统一到 uV 尺度。
    返回形状 [n_epochs, samples_per_epoch] 的 float32 数组。
    """
    x_native = raw.get_data(picks=[ch_name]).squeeze().astype(np.float32)
    x_scaled = x_native * 1e6

    def _robust_std(x):
        x = np.asarray(x).ravel()
        if x.size == 0:
            return 0.0
        lo, hi = np.percentile(x, [5, 95])
        x2 = x[(x >= lo) & (x <= hi)]
        if x2.size == 0:
            return float(np.std(x))
        return float(np.std(x2))

    target_std = 20.0

    def _score(x):
        s = max(_robust_std(x), 1e-12)
        return abs(np.log10(s / target_std))

    signal_uv = x_native if _score(x_native) <= _score(x_scaled) else x_scaled

    sf = float(raw.info["sfreq"])
    samples_per_epoch = int(round(sf * epoch_duration))  # 100 Hz x 30 s = 3000
    n_epochs = len(signal_uv) // samples_per_epoch
    if target_n_epochs is not None:
        n_epochs = min(n_epochs, int(target_n_epochs))

    if n_epochs <= 0:
        return np.zeros((0, samples_per_epoch), dtype=np.float32)

    signal_uv = signal_uv[:n_epochs * samples_per_epoch]
    return signal_uv.reshape(n_epochs, samples_per_epoch).astype(np.float32)  # {n_epochs, 3000}

def load_sleep_edf_data(edf_path, config):
    # 读 REC/EDF、选通道、跑 YASA，返回 epoch 级分期。
    """
    ISRUC_S1 版：
    1) 用 MNE 读连续 .rec 信号
    2) 用 YASA SleepStaging 自动分期
    3) 不再手工切 epoch 做检测
    返回:
        raw, hyp, proba, eeg_name, loc_name, roc_name, emg_name
    """
    try:
        raw = read_raw_edf_or_rec(edf_path, preload=False, verbose=False)
        ch_names = raw.ch_names
        print(f"All channels: {ch_names}")

        eeg_name = pick_channel(ch_names, [
            'C3-M2', 'C4-M1',
            'F3-M2', 'F4-M1',
            'O1-M2', 'O2-M1',
            'C3-A2', 'C4-A1',
            'F3-A2', 'F4-A1',
            'O1-A2', 'O2-A1','C4'
        ])
        if eeg_name is None:
            raise ValueError(f"未找到合适 EEG 通道: {ch_names}")

        loc_name = pick_channel(ch_names, [
            'E1-M2', 'LOC-A2', 'LOC', 'EOG Left Horiz', 'EOG(L)', 'LEOG'
        ])
        roc_name = pick_channel(ch_names, [
            'E2-M1', 'ROC-A1', 'ROC', 'EOG Right Horiz', 'EOG(R)', 'REOG'
        ])
        emg_name = pick_channel(ch_names, [
            'EMG Chin', 'EMG', 'Chin1-Chin2', 'EMG1-EMG2',
            'X1', 'X2', 'X3', 'X4'
        ])

        print(f"EEG channel: {eeg_name}")
        print(f"EOG LOC: {loc_name}, EOG ROC: {roc_name}, EMG: {emg_name}")
        if loc_name is None or roc_name is None:
            print("Warning: LOC/ROC 不完整，YASA REM / SEM 检测能力会下降。")

        needed_channels = list(dict.fromkeys(
            name for name in (eeg_name, loc_name, roc_name, emg_name) if name is not None
        ))
        raw.pick(needed_channels)
        raw.load_data(verbose=False)
        temporary_source_dir = getattr(raw, "_isruc_temporary_source_dir", None)
        if temporary_source_dir:
            shutil.rmtree(temporary_source_dir, ignore_errors=True)
            raw._isruc_temporary_source_dir = None
        target_fs = float(getattr(config, "target_fs", 100))
        if abs(float(raw.info["sfreq"]) - target_fs) > 1e-6:
            print(f"Resampling selected channels: {raw.info['sfreq']} Hz -> {target_fs} Hz")
            raw.resample(target_fs, npad="auto", verbose=False)

        sls = yasa.SleepStaging(
            raw,
            eeg_name=eeg_name,
            eog_name=loc_name,
            emg_name=emg_name
        )

        hyp = sls.predict()
        proba = sls.predict_proba()

        print(f"YASA sleep staging done. Total epochs: {len(hyp)}")

        hyp_series = hyp.hypno if hasattr(hyp, 'hypno') else pd.Series(hyp)
        hyp_series_norm = hyp_series.astype(str).map(normalize_stage_str)
        stage_counts = hyp_series_norm.value_counts()

        print("Sleep stage counts (epochs):")
        for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
            count = int(stage_counts.get(stage, 0))
            ratio = count / len(hyp_series) if len(hyp_series) > 0 else 0
            print(f"  {stage}: {count} ({ratio:.2%})")

        print("All stage counts:")
        for stage_name, cnt in hyp_series.astype(str).value_counts().items():
            print(f"  {stage_name}: {cnt}")

        return raw, hyp, proba, eeg_name, loc_name, roc_name, emg_name

    except Exception as e:
        print(f"Error loading {edf_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None

def visualize_tsne_raw_signal(raw, hyp, eeg_name, output_path, subject_id,
                              epoch_duration=30, max_per_class=500, downsample_to=300):
    # 把原始 epoch 投到 t-SNE 看分布。
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    sfreq = raw.info["sfreq"]
    data = raw.copy().pick([eeg_name]).get_data()[0]   # 单通道原始EEG

    hyp_series = hyp.hypno if hasattr(hyp, "hypno") else pd.Series(hyp)
    n_epochs = len(hyp_series)
    samples_per_epoch = int(epoch_duration * sfreq)

    X = []
    y = []

    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        if end > len(data):
            break

        epoch_signal = data[start:end]

        # 每个epoch做标准化
        std = np.std(epoch_signal)
        if std > 0:
            epoch_signal = (epoch_signal - np.mean(epoch_signal)) / std
        else:
            epoch_signal = epoch_signal - np.mean(epoch_signal)

        # 把3000点左右降采样到较低维，便于t-SNE
        if downsample_to is not None and downsample_to < len(epoch_signal):
            idx = np.linspace(0, len(epoch_signal) - 1, downsample_to).astype(int)
            epoch_signal = epoch_signal[idx]

        stage = normalize_stage_str(hyp_series.iloc[i])
        label = stage_str_to_int(stage)

        if 0 <= label <= 4:
            X.append(epoch_signal)
            y.append(label)

    if len(X) == 0:
        print(f"{subject_id}: 没有可用于原始信号t-SNE的样本")
        return

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    # 类别均衡采样，避免N2过多
    rng = np.random.default_rng(42)
    keep_idx = []
    for cls in range(5):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep_idx.append(idx)

    if len(keep_idx) == 0:
        print(f"{subject_id}: 采样后没有样本")
        return

    keep_idx = np.concatenate(keep_idx)
    X = X[keep_idx]
    y = y[keep_idx]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    for cls, name in enumerate(class_names):
        mask = (y == cls)
        if np.any(mask):
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, alpha=0.7, label=name)

    plt.legend()
    plt.title(f"t-SNE of raw EEG signal ({subject_id})")
    plt.tight_layout()

    save_path = os.path.join(output_path, f"{subject_id}_tsne_raw_signal.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"{subject_id}: 原始信号 t-SNE 已保存到 {save_path}")

def extract_features_for_epoch(epoch, fs, freq_bands=None, waveform_params=None):
    # 兼容旧接口；当前版本不再直接用它。
    """
    【功能已彻底改变】
    提取单个epoch的形态学波形特征。
    此函数名保留，但内部实现改为调用 MorphologyDetector。
    注意: 参数 freq_bands 和 waveform_params 在此版本中不再使用，仅为兼容性保留。
    """
    # 此函数在此版本中已不直接使用，因为检测在 detect_all_in_epoch 中完成。
    # 保留函数定义是为了避免破坏可能的外部调用结构。
    # 实际特征提取在 generate_single_epoch_description 中通过 detector 完成。
    print("Warning: extract_features_for_epoch is deprecated in morphology mode. Use detector directly.")
    return {}


def create_feature_matrix(eeg_epochs, fs, freq_bands=None, waveform_params=None):
    # 兼容旧接口；当前版本不再直接用它。
    """
    【功能已彻底改变】
    为所有epoch创建特征矩阵。
    此函数名保留，但返回的不再是DataFrame，而是包含描述和特征的字典列表。
    """
    # 此函数在此版本中已不直接使用。
    # 保留函数定义是为了避免破坏可能的外部调用结构。
    print("Warning: create_feature_matrix is deprecated in morphology mode.")
    return []


# def generate_single_epoch_description(waveform_features, sleep_stage=None, text_templates=None):
#     """
#     根据单个epoch的波形特征字典，生成自然语言描述。
#     这是新版描述生成的核心。
#     Args:
#         waveform_features: MorphologyDetector._results_to_features 返回的特征字典。
#         sleep_stage: 当前睡眠分期。
#         text_templates: 描述模板字典。
#     Returns:
#         str: 自然语言描述。
#     """
#     if text_templates is None:
#         text_templates = PreprocessConfig().text_templates
#
#     desc_parts = []
#     stage_map = {0: "清醒期(W)", 1: "N1期", 2: "N2期", 3: "N3期", 4: "REM期"}
#
#     # 1. 睡眠分期
#     if sleep_stage is not None and sleep_stage in stage_map:
#         desc_parts.append(f"当前处于{stage_map[sleep_stage]}。")
#
#     # 2. 基于波形存在性的核心描述 (优先级规则)
#     if waveform_features.get('has_spindle', 0) and waveform_features.get('has_kcomplex', 0):
#         desc_parts.append(text_templates['spindle_kcomplex'])
#     elif waveform_features.get('has_spindle', 0):
#         desc_parts.append(text_templates['spindle_only'])
#     elif waveform_features.get('has_kcomplex', 0):
#         desc_parts.append(text_templates['kcomplex_only'])
#     elif waveform_features.get('has_slowwave', 0):
#         density = waveform_features.get('slowwave_density', 0)
#         if density > 2.0:  # 阈值可根据实际情况调整
#             desc_parts.append(text_templates['slow_wave_rich'])
#         else:
#             desc_parts.append("存在少量慢波活动。")
#     elif waveform_features.get('has_alpha', 0):
#         desc_parts.append(text_templates['alpha_prominent'])
#     elif waveform_features.get('has_vertex', 0):
#         desc_parts.append(text_templates['vertex_present'])
#     elif waveform_features.get('has_rem', 0):
#         desc_parts.append(text_templates['rem_detected'])
#     else:
#         desc_parts.append(text_templates['no_significant_wave'])
#
#     # 3. 添加量化指标细节
#     detail_parts = []
#     wave_info = [
#         ('spindle', '纺锤波'),
#         ('kcomplex', 'K-复合波'),
#         ('slowwave', '慢波'),
#         ('alpha', 'α波'),
#         ('vertex', '顶尖波'),
#         ('rem', '快速眼动')
#     ]
#     for wave_key, wave_name in wave_info:
#         density = waveform_features.get(f'{wave_key}_density', 0)
#         count = waveform_features.get(f'{wave_key}_count', 0)
#         if density > 0 or count > 0:
#             detail_parts.append(f"{wave_name}{count}个(密度{density:.1f}/分)")
#
#     if detail_parts:
#         # 将详情作为补充说明，避免主描述过于冗长
#         detail_str = "波形详情：" + "，".join(detail_parts) + "。"
#         # 可以选择性添加，或存为单独字段
#         # desc_parts.append(detail_str)
#         # 本次修改暂不加入主描述，保留在特征中
#
#     # 组合成完整描述
#     full_description = " ".join(desc_parts)
#     return full_description
# def generate_single_epoch_description(waveform_features, sleep_stage=None, text_templates=None):
#     if text_templates is None:
#         text_templates = PreprocessConfig().text_templates
#
#     stage_map = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
#
#     parts = []
#
#     # 1) stage 槽位
#     if sleep_stage is not None and sleep_stage in stage_map:
#         parts.append(f"睡眠阶段: {stage_map[sleep_stage]}。")
#     else:
#         parts.append("睡眠阶段: 未知。")
#
#     # 2) morphology 槽位：允许多事件共存
#     morph_parts = []
#
#     if waveform_features.get("has_spindle", 0):
#         count = waveform_features.get("spindle_count", 0)
#         density = waveform_features.get("spindle_density", 0.0)
#         morph_parts.append(f"检测到纺锤波，数量{count}，密度{density:.2f}/分")
#
#     if waveform_features.get("has_kcomplex", 0):
#         count = waveform_features.get("kcomplex_count", 0)
#         density = waveform_features.get("kcomplex_density", 0.0)
#         morph_parts.append(f"检测到K复合波，数量{count}，密度{density:.2f}/分")
#
#     if waveform_features.get("has_slowwave", 0):
#         count = waveform_features.get("slowwave_count", 0)
#         density = waveform_features.get("slowwave_density", 0.0)
#         morph_parts.append(f"检测到慢波活动，数量{count}，密度{density:.2f}/分")
#
#     if waveform_features.get("has_alpha", 0):
#         count = waveform_features.get("alpha_count", 0)
#         density = waveform_features.get("alpha_density", 0.0)
#         morph_parts.append(f"检测到α波活动，数量{count}，密度{density:.2f}/分")
#
#     if waveform_features.get("has_vertex", 0):
#         count = waveform_features.get("vertex_count", 0)
#         density = waveform_features.get("vertex_density", 0.0)
#         morph_parts.append(f"检测到顶尖波，数量{count}，密度{density:.2f}/分")
#
#     if waveform_features.get("has_rem", 0):
#         count = waveform_features.get("rem_count", 0)
#         density = waveform_features.get("rem_density", 0.0)
#         morph_parts.append(f"检测到快速眼动事件，数量{count}，密度{density:.2f}/分")
#
#     if morph_parts:
#         parts.append("形态学事件: " + "；".join(morph_parts) + "。")
#     else:
#         parts.append("形态学事件: 未检测到显著特征。")
#
#     # 3) stage-event consistency 槽位
#     consistency_parts = []
#
#     if sleep_stage == 2 and waveform_features.get("has_spindle", 0):
#         consistency_parts.append("纺锤波与当前N2阶段一致")
#     if sleep_stage == 2 and waveform_features.get("has_kcomplex", 0):
#         consistency_parts.append("K复合波与当前N2阶段一致")
#     if sleep_stage == 3 and waveform_features.get("has_slowwave", 0):
#         consistency_parts.append("慢波活动与当前N3阶段一致")
#     if sleep_stage == 1 and waveform_features.get("has_vertex", 0):
#         consistency_parts.append("顶尖波与当前N1阶段一致")
#     if sleep_stage == 4 and waveform_features.get("has_rem", 0):
#         consistency_parts.append("快速眼动与当前REM阶段一致")
#     if sleep_stage == 0 and waveform_features.get("has_alpha", 0):
#         consistency_parts.append("α波活动与清醒阶段一致")
#
#     if consistency_parts:
#         parts.append("阶段一致性: " + "；".join(consistency_parts) + "。")
#
#     return " ".join(parts)
# def generate_single_epoch_description(waveform_features, sleep_stage=None, text_templates=None):
#     """
#     方案A：结构化睡眠语言（仍然输出字符串）
#     固定字段 + 固定顺序 + 固定词表
#     """
#     stage_map = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
#     stage_token = stage_map.get(sleep_stage, "UNK")
#
#     def build_slot(name, key):
#         has_flag = int(waveform_features.get(f"has_{key}", 0))
#         count = int(waveform_features.get(f"{key}_count", 0))
#         density = float(waveform_features.get(f"{key}_density", 0.0))
#         return f"{name}={has_flag},count={count},density={density:.2f}"
#
#     parts = [
#         f"STAGE={stage_token}",
#         build_slot("SPINDLE", "spindle"),
#         build_slot("KCOMPLEX", "kcomplex"),
#         build_slot("SLOWWAVE", "slowwave"),
#         build_slot("ALPHA", "alpha"),
#         build_slot("VERTEX", "vertex"),
#         build_slot("REM", "rem"),
#     ]
#
#     consistency = []
#     if sleep_stage == 2 and waveform_features.get("has_spindle", 0):
#         consistency.append("spindle:N2_match")
#     if sleep_stage == 2 and waveform_features.get("has_kcomplex", 0):
#         consistency.append("kcomplex:N2_match")
#     if sleep_stage == 3 and waveform_features.get("has_slowwave", 0):
#         consistency.append("slowwave:N3_match")
#     if sleep_stage == 1 and waveform_features.get("has_vertex", 0):
#         consistency.append("vertex:N1_match")
#     if sleep_stage == 4 and waveform_features.get("has_rem", 0):
#         consistency.append("rem:REM_match")
#     if sleep_stage == 0 and waveform_features.get("has_alpha", 0):
#         consistency.append("alpha:W_match")
#
#     parts.append("CONSISTENCY=" + ("|".join(consistency) if consistency else "none"))
#
#     return "; ".join(parts)
def load_subject_level_label(subject_id, label_csv=None):
    # 读取被试级障碍标签。
    """
    从CSV读取被试级睡眠障碍标签
    必需列:
        subject_id
        sleep_disorder_label
    可选列:
        sleep_disorder_name
    """
    if label_csv is None or (not os.path.exists(label_csv)):
        return -1, "unknown"

    df = pd.read_csv(label_csv)
    if "subject_id" not in df.columns or "sleep_disorder_label" not in df.columns:
        raise ValueError(
            f"标签文件 {label_csv} 缺少必要列，必须包含: subject_id, sleep_disorder_label"
        )

    df["subject_id"] = df["subject_id"].astype(str)
    subject_id = str(subject_id)

    row = df[df["subject_id"] == subject_id]
    if len(row) == 0:
        return -1, "unknown"

    label = int(row.iloc[0]["sleep_disorder_label"])
    if "sleep_disorder_name" in df.columns:
        label_name = str(row.iloc[0]["sleep_disorder_name"])
    else:
        label_name = str(label)

    return label, label_name
def generate_single_epoch_description(waveform_features, sleep_stage=None, text_templates=None, use_stage_token=True):
    # 单个 epoch 特征 -> 1 条文本。
    stage_token = str(sleep_stage) if sleep_stage is not None else "UNK"

    def bucket_count(x):
        x = int(x)
        if x <= 0:
            return "0"
        elif x == 1:
            return "1"
        else:
            return "2plus"

    def bucket_density(x):
        x = float(x)
        if x <= 0:
            return "none"
        elif x <= 2:
            return "low"
        elif x <= 4:
            return "medium"
        else:
            return "high"

    def build_slot(name, key):
        has_flag = int(waveform_features.get(f"has_{key}", 0))
        count = bucket_count(waveform_features.get(f"{key}_count", 0))
        density = bucket_density(waveform_features.get(f"{key}_density", 0.0))
        state = "present" if has_flag else "absent"
        return f"{name} {state} count {count} density {density}"

    parts = []
    if use_stage_token:
        parts.append(f"stage {stage_token}")

    parts.extend([
        build_slot("alpha", "alpha"),
        build_slot("theta", "theta"),
        build_slot("vertex", "vertex"),
        build_slot("sem", "sem"),
        build_slot("spindle", "spindle"),
        build_slot("kcomplex", "kcomplex"),
        build_slot("slowwave", "slowwave"),
        build_slot("rem", "rem"),
        f"alpha_ratio {waveform_features.get('alpha_ratio', 0.0):.4f}",
        f"theta_ratio {waveform_features.get('theta_ratio', 0.0):.4f}",
        f"alpha_theta_ratio {waveform_features.get('alpha_theta_ratio', 0.0):.4f}",
    ])

    consistency = []
    if sleep_stage == 0 and waveform_features.get("has_alpha", 0):
        consistency.append("alpha:W_match")
    if sleep_stage == 1 and waveform_features.get("has_theta", 0):
        consistency.append("theta:N1_match")
    if sleep_stage == 1 and waveform_features.get("has_vertex", 0):
        consistency.append("vertex:N1_match")
    if sleep_stage == 1 and waveform_features.get("has_sem", 0):
        consistency.append("sem:N1_match")
    if sleep_stage == 2 and waveform_features.get("has_kcomplex", 0):
        consistency.append("kcomplex:N2_match")

    parts.append("consistency " + ("|".join(consistency) if consistency else "none"))
    return "; ".join(parts)
def build_epoch_feature_table(
    hyp,
    alpha_df=None,
    theta_df=None,
    sem_df=None,
    kc_df=None,
    sp_df=None,
    sw_df=None,
    rem_df=None
):
    # 连续事件表 -> epoch 表，输出 {n_epochs, n_features}。
    """
    将连续事件表映射到 YASA 30秒分期 bin
    """
    hyp_series = hyp.hypno if hasattr(hyp, "hypno") else pd.Series(hyp)
    n_epochs = len(hyp_series)

    rows = []  # 长度 = n_epochs

    for i in range(n_epochs):
        stage_str = normalize_stage_str(hyp_series.iloc[i])
        rows.append({
            "epoch_index": i,
            "sleep_stage_str": stage_str,
            "sleep_stage": stage_str_to_int(stage_str),
            "has_alpha": 0, "alpha_count": 0, "alpha_density": 0.0,
            "has_theta": 0, "theta_count": 0, "theta_density": 0.0,
            "has_sem": 0, "sem_count": 0, "sem_density": 0.0,
            "has_vertex": 0, "vertex_count": 0, "vertex_density": 0.0,
            "has_kcomplex": 0, "kcomplex_count": 0, "kcomplex_density": 0.0,

            "has_spindle": 0, "spindle_count": 0, "spindle_density": 0.0,
            "has_slowwave": 0, "slowwave_count": 0, "slowwave_density": 0.0,
            "has_rem": 0, "rem_count": 0, "rem_density": 0.0,

            "alpha_power": 0.0,
            "theta_power": 0.0,
            "alpha_ratio": 0.0,
            "theta_ratio": 0.0,
            "alpha_theta_ratio": 0.0,
        })

    def add_events(df, prefix):
        if df is None or len(df) == 0:
            return
        for _, row in df.iterrows():
            start_sec = float(row["Start"])
            end_sec = float(row["End"]) if "End" in row else start_sec
            center_sec = 0.5 * (start_sec + end_sec)
            epoch_idx = int(center_sec // 30)
            if 0 <= epoch_idx < n_epochs:
                rows[epoch_idx][f"{prefix}_count"] += 1

    add_events(alpha_df, "alpha")
    add_events(theta_df, "theta")
    add_events(sem_df, "sem")
    add_events(kc_df, "kcomplex")
    add_events(sp_df, "spindle")
    add_events(sw_df, "slowwave")
    add_events(rem_df, "rem")

    for row in rows:
        for key in ["alpha", "theta", "sem", "kcomplex", "spindle", "slowwave", "rem"]:
            count = row[f"{key}_count"]
            row[f"has_{key}"] = 1 if count > 0 else 0
            row[f"{key}_density"] = count / 0.5

    return pd.DataFrame(rows)  # {n_epochs, n_features}
def generate_text_descriptions(all_features_list, all_sleep_stages, config):
    # 批量把 epoch 特征转成文本。
    """
    【功能已彻底改变】
    为所有epoch生成文本描述。
    此函数名保留，但输入和逻辑已完全不同。
    Args:
        all_features_list: 列表，每个元素是一个epoch的特征字典。
        all_sleep_stages: 列表，对应的睡眠分期标签。
        config: 配置对象。
    Returns:
        list: 文本描述列表，与输入一一对应。
    """
    descriptions = []
    for features, stage in zip(all_features_list, all_sleep_stages):
        desc = generate_single_epoch_description(features, stage, config.text_templates)
        descriptions.append(desc)
    return descriptions


def preprocess_sleep_edf_dataset(data_path, output_path, config=None):
    # 主流程：读 EDF -> 检测事件 -> 对齐标签 -> 保存。
    if config is None:
        config = PreprocessConfig()

    os.makedirs(output_path, exist_ok=True)
    psg_files = find_psg_files(data_path)

    if len(psg_files) == 0:
        print("No PSG files found.")
        return [], [], {}

    all_subjects_data = []
    global_descriptions = []
    global_features = []
    subject_info = []
    total_epochs = 0
    sample_description = ""
    sample_features = {}

    for i, psg_file in enumerate(psg_files):
        subject_id = Path(psg_file).stem
        print(f"\nProcessing {i + 1}/{len(psg_files)}: {subject_id}")

        subject_output_path = os.path.join(output_path, f"{subject_id}_morphology.npz")
        if os.path.exists(subject_output_path):
            try:
                with np.load(subject_output_path, allow_pickle=True) as existing:
                    expected_samples = int(round(
                        float(getattr(config, "target_fs", 100))
                        * float(getattr(config, "epoch_duration", 30))
                    ))
                    if "eeg_epochs" not in existing:
                        raise ValueError("missing eeg_epochs")
                    eeg_shape = existing["eeg_epochs"].shape
                    if len(eeg_shape) != 2 or int(eeg_shape[-1]) != expected_samples:
                        raise ValueError(
                            f"stale epoch shape {eeg_shape}; expected (*, {expected_samples})"
                        )
                    if "waveform_features" not in existing:
                        raise ValueError("missing waveform_features")
                    features = existing["waveform_features"].tolist()
                    required_events = {"has_alpha", "has_kcomplex", "has_vertex"}
                    if features and not required_events.issubset(features[0]):
                        raise ValueError("missing updated alpha/kcomplex/vertex labels")
                    if "true_sleep_stages" in existing:
                        existing_epochs = int(len(existing["true_sleep_stages"]))
                    elif "eeg_epochs" in existing:
                        existing_epochs = int(len(existing["eeg_epochs"]))
                    else:
                        existing_epochs = 0
                all_subjects_data.append({
                    "subject_id": subject_id,
                    "output_path": subject_output_path,
                    "num_epochs": existing_epochs,
                    "status": "existing",
                })
                subject_info.append(subject_id)
                total_epochs += existing_epochs
                print(f"  Already processed, skipping: {subject_output_path}")
                continue
            except Exception as exc:
                print(f"  Existing output is stale or invalid; rebuilding it: {exc}")

        raw, hyp, proba, eeg_name, loc_name, roc_name, emg_name = load_sleep_edf_data(psg_file, config)
        if raw is not None and hyp is not None and eeg_name is not None:
            visualize_tsne_raw_signal(
                raw=raw,
                hyp=hyp,
                eeg_name=eeg_name,
                output_path=output_path,
                subject_id=subject_id,
                epoch_duration=config.epoch_duration,
                max_per_class=500,
                downsample_to=300
            )
        if raw is None:
            print("  Skipping due to load error")
            continue

        # 新增：读取真实标签
        true_sleep_stages = load_hypnogram_data(psg_file, config)
        if true_sleep_stages is None:
            print("  Warning: true labels not found, this subject will use -1 labels")

        detector = MorphologyDetector(fs=raw.info["sfreq"], channel_names=eeg_name)
        print("  Running YASA three-event detection on continuous signal...")

        sp_df, sw_df, rem_df = detector.detect_yasa_three_events(
            raw=raw,
            hyp=hyp,
            eeg_name=eeg_name,
            loc_name=loc_name,
            roc_name=roc_name
        )
        alpha_df = detector.detect_alpha_events_continuous(raw, hyp, eeg_name)
        kc_df = detector.detect_k_complex_events_continuous(raw, hyp, eeg_name)
        theta_df = detector.detect_theta_events_continuous(raw, hyp, eeg_name)
        sem_df = pd.DataFrame()
        if loc_name is not None and roc_name is not None:
            sem_df = detector.detect_sem_events_continuous(raw, hyp, loc_name, roc_name)

        # ===== 新增：打印三类事件检测总数 =====
        alpha_count = len(alpha_df) if alpha_df is not None else 0
        kc_count = len(kc_df) if kc_df is not None else 0
        theta_count = len(theta_df) if theta_df is not None else 0
        sem_count = len(sem_df) if sem_df is not None else 0
        sp_count = len(sp_df) if sp_df is not None else 0
        sw_count = len(sw_df) if sw_df is not None else 0
        rem_count = len(rem_df) if rem_df is not None else 0

        print("  Detected event counts:")
        print(f"    Alpha    : {alpha_count}")
        print(f"    KComplex : {kc_count}")
        print(f"    Theta    : {theta_count}")
        print(f"    SEMs     : {sem_count}")
        print(f"    Spindles : {sp_count}")
        print(f"    SlowWaves: {sw_count}")
        print(f"    REMs     : {rem_count}")





        epoch_df = build_epoch_feature_table(
            hyp,
            alpha_df=alpha_df,
            kc_df=kc_df,
            theta_df=theta_df,
            sem_df=sem_df,
            sp_df=sp_df,
            sw_df=sw_df,
            rem_df=rem_df
        )  # {n_epochs, n_features}
        eeg = raw.copy().pick([eeg_name]).get_data(units="uV").squeeze()
        epoch_len = int(raw.info["sfreq"] * 30)  # 3000

        for i in range(len(epoch_df)):
            s = i * epoch_len
            e = s + epoch_len
            eeg_epoch = eeg[s:e]  # {3000}

            # 1) band ratio
            band_feat = detector.compute_band_features(eeg_epoch)
            for k, v in band_feat.items():
                epoch_df.loc[i, k] = v

            # 2) vertex：只在 YASA = N1 时检测
            stage_val = int(epoch_df.loc[i, "sleep_stage"])
            if stage_val == 1:
                vertex_count, vertex_events = detector.detect_vertex_waves(eeg_epoch)
            else:
                vertex_count, vertex_events = 0, []

            epoch_df.loc[i, "has_vertex"] = int(vertex_count > 0)
            epoch_df.loc[i, "vertex_count"] = int(vertex_count)
            epoch_df.loc[i, "vertex_density"] = float(vertex_count / 0.5)
        # 新增：将真实标签对齐到当前 epoch_df
        if true_sleep_stages is not None:
            min_len = min(len(epoch_df), len(true_sleep_stages))
            if len(epoch_df) != len(true_sleep_stages):
                print(
                    f"  Aligning YASA epochs and true labels: {len(epoch_df)} vs {len(true_sleep_stages)} -> {min_len}")

            epoch_df = epoch_df.iloc[:min_len].copy()
            true_sleep_stages = true_sleep_stages[:min_len]
        else:
            true_sleep_stages = [-1] * len(epoch_df)

        stage_event_summary = (
            epoch_df.groupby("sleep_stage_str")[[
                "alpha_count", "theta_count", "sem_count", "vertex_count", "kcomplex_count",
                "spindle_count", "slowwave_count", "rem_count"
            ]].sum()
        )

        print("  Event counts by sleep stage:")
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            if stage in stage_event_summary.index:
                row = stage_event_summary.loc[stage]
                print(
                    f"    {stage}: "
                    f"alpha={int(row['alpha_count'])}, "
                    f"theta={int(row['theta_count'])}, "
                    f"sem={int(row['sem_count'])}, "
                    f"vertex={int(row['vertex_count'])}, "
                    f"kcomplex={int(row['kcomplex_count'])}, "
                    f"spindle={int(row['spindle_count'])}, "
                    f"slowwave={int(row['slowwave_count'])}, "
                    f"rem={int(row['rem_count'])}"
                )
            else:
                print(f"    {stage}: alpha=0, theta=0, sem=0, vertex=0, kcomplex=0, spindle=0, slowwave=0, rem=0")
        # 1) 生成两套描述：
        #    plain: 下游正式训练用
        #    with_stage: 仅分析 / 可视化 / 消融
        epoch_df["description_plain"] = epoch_df.apply(
            lambda row: generate_single_epoch_description(
                row.to_dict(),
                sleep_stage=row["sleep_stage"],
                text_templates=config.text_templates,
                use_stage_token=False
            ),
            axis=1
        )

        epoch_df["description_with_stage"] = epoch_df.apply(
            lambda row: generate_single_epoch_description(
                row.to_dict(),
                sleep_stage=row["sleep_stage"],
                text_templates=config.text_templates,
                use_stage_token=True
            ),
            axis=1
        )

        descriptions_plain = epoch_df["description_plain"].tolist()
        descriptions_with_stage = epoch_df["description_with_stage"].tolist()

        all_waveform_features = epoch_df.to_dict("records")
        sleep_stages = epoch_df["sleep_stage"].tolist()
        sleep_stage_str = epoch_df["sleep_stage_str"].tolist()

        # 2) 读取被试级睡眠障碍标签（没有CSV就写-1）
        sleep_disorder_label, sleep_disorder_name = load_subject_level_label(
            subject_id,
            getattr(config, "label_csv", None)
        )

        eeg_epochs = extract_epoch_signal_array(
            raw,
            eeg_name,
            epoch_duration=int(getattr(config, "epoch_duration", 30)),
            target_n_epochs=len(true_sleep_stages)
        )  # {n_epochs, 3000}
        eog_loc_epochs = (
            extract_epoch_signal_array(
                raw, loc_name,
                epoch_duration=int(getattr(config, "epoch_duration", 30)),
                target_n_epochs=len(true_sleep_stages)
            )
            if loc_name is not None else np.zeros_like(eeg_epochs)
        )
        eog_roc_epochs = (
            extract_epoch_signal_array(
                raw, roc_name,
                epoch_duration=int(getattr(config, "epoch_duration", 30)),
                target_n_epochs=len(true_sleep_stages)
            )
            if roc_name is not None else np.zeros_like(eeg_epochs)
        )
        eog_available = np.asarray(
            [int(loc_name is not None), int(roc_name is not None)],
            dtype=np.float32
        )

        subject_data = {  # 主要按 epoch 对齐
            "subject_id": subject_id,

            # 兼容旧代码：descriptions 默认指向 plain 版本
            "descriptions": descriptions_plain,
            "descriptions_plain": descriptions_plain,
            "descriptions_with_stage": descriptions_with_stage,

            "waveform_features": all_waveform_features,

            # epoch级真实标签
            "sleep_stages": np.array(true_sleep_stages),  # {n_epochs}
            "true_sleep_stages": np.array(true_sleep_stages),  # {n_epochs}

            # epoch级YASA标签
            "yasa_sleep_stages": np.array(sleep_stages),  # {n_epochs}
            "yasa_sleep_stage_str": np.array(sleep_stage_str),  # {n_epochs}

            # subject级标签
            "sleep_disorder_label": np.array(sleep_disorder_label),
            "sleep_disorder_name": np.array(sleep_disorder_name),

            "yasa_proba": proba.to_dict("list") if hasattr(proba, "to_dict") else None,  # 5 类概率
            "channel": eeg_name,
            "eeg_epochs": eeg_epochs,  # {n_epochs, 3000}
            "eog_loc_epochs": eog_loc_epochs,
            "eog_roc_epochs": eog_roc_epochs,
            "eog_available": eog_available,
            "sampling_rate": raw.info["sfreq"],
            "epoch_duration": 30
        }

        np.savez_compressed(subject_output_path, **subject_data)

        # 3) 预训练语料默认保存 plain description，避免阶段标签泄漏到正式下游输入
        jsonl_path = os.path.join(output_path, f"{subject_id}_descriptions.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx, (_, row) in enumerate(zip(range(len(epoch_df)), epoch_df.iterrows())):
                _, row = row
                record = {
                    "epoch_index": int(row["epoch_index"]),
                    "sleep_stage": int(true_sleep_stages[idx]),
                    "true_sleep_stage": int(true_sleep_stages[idx]),
                    "yasa_sleep_stage": int(row["sleep_stage"]),
                    "yasa_sleep_stage_str": str(row["sleep_stage_str"]),
                    "description": row["description_plain"],  # 预训练默认用 plain
                    "description_plain": row["description_plain"],
                    "description_with_stage": row["description_with_stage"],
                    "features": row.to_dict(),
                    "sleep_disorder_label": int(sleep_disorder_label),
                    "sleep_disorder_name": str(sleep_disorder_name)
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 可选：单独保存事件表
        sp_df.to_csv(os.path.join(output_path, f"{subject_id}_spindles.csv"), index=False)
        kc_df.to_csv(os.path.join(output_path, f"{subject_id}_kcomplex.csv"), index=False)
        sw_df.to_csv(os.path.join(output_path, f"{subject_id}_slowwaves.csv"), index=False)
        rem_df.to_csv(os.path.join(output_path, f"{subject_id}_rem.csv"), index=False)

        all_subjects_data.append({
            "subject_id": subject_id,
            "output_path": subject_output_path,
            "num_epochs": int(len(true_sleep_stages)),
            "status": "processed",
        })
        total_epochs += len(descriptions_plain)
        if not sample_description and descriptions_plain:
            sample_description = descriptions_plain[0]
        if not sample_features and all_waveform_features:
            sample_features = all_waveform_features[0]
        subject_info.append(subject_id)

        print(f"  Saved: {subject_output_path}")
        print(f"  Saved: {jsonl_path}")

        raw.close()
        del raw, subject_data, eeg_epochs, eog_loc_epochs, eog_roc_epochs
        gc.collect()

    if all_subjects_data:
        global_info = {
            "subject_ids": subject_info,
            "total_epochs": int(total_epochs),
            "config": config.__dict__,
            "sample_description": sample_description,
            "sample_features": sample_features,
        }
        np.savez_compressed(os.path.join(output_path, "global_morphology_info.npz"), **global_info)

    print("\n预处理完成!")
    print(f"处理了 {len(all_subjects_data)} 个被试")
    print(f"总生成 {total_epochs} 条文本描述")
    return all_subjects_data, global_descriptions, global_features

# ===================== 以下为与原版兼容的辅助函数，未作核心修改 =====================

def load_hypnogram_data(edf_path, config):
    # 读取 ISRUC_S1 真实分期，输出 {n_epochs}。
    """
    加载 ISRUC_S1 睡眠阶段标注数据（同目录 _1.txt 标签格式）
    例如：
        PSG:   D:/dataset/ISRUC_S1/9/9/9.rec
        Label: D:/dataset/ISRUC_S1/9/9/9_1.txt

    每一行通常对应一个 30 秒 epoch。
    常见标签可能是：
        0, 1, 2, 3, 5
        W, N1, N2, N3, REM
        Sleep stage W / N1 / N2 / N3 / R
    """
    try:
        rec_path = Path(edf_path)
        subject_base = rec_path.stem
        label_path = rec_path.with_name(f"{subject_base}_1.txt")

        if not label_path.exists():
            print(f"  Warning: label file not found for {rec_path.name}")
            print(f"  Expected: {label_path}")
            return None

        print(f"  Found label file: {label_path.name}")

        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_lines = f.readlines()

        sleep_stages = []
        skipped_lines = 0

        for raw_line in raw_lines:
            line = str(raw_line).replace('﻿', '').strip()
            if not line:
                continue

            mapped = map_annotation_to_stage(line)
            if mapped != -1:
                sleep_stages.append(mapped)
                continue

            found_any = False
            for token in re.split(r'[	,; ]+', line):
                token = token.strip()
                if not token:
                    continue
                mapped_tok = map_annotation_to_stage(token)
                if mapped_tok != -1:
                    sleep_stages.append(mapped_tok)
                    found_any = True

            if not found_any:
                skipped_lines += 1

        valid_count = sum(1 for x in sleep_stages if x != -1)
        invalid_count = sum(1 for x in sleep_stages if x == -1)

        print(f"  Loaded {len(sleep_stages)} total stage epochs from TXT")
        print(f"  Valid epochs: {valid_count}, invalid/unknown epochs: {invalid_count}")
        if skipped_lines > 0:
            print(f"  Skipped non-stage lines: {skipped_lines}")

        return sleep_stages

    except Exception as e:
        print(f"Error loading ISRUC_S1 hypnogram data: {e}")
        import traceback
        traceback.print_exc()
        return None



def parse_sleep_stages(annotations, epoch_duration):
    # 注释列表 -> 每个 30s 1 个标签。
    """解析睡眠阶段标注 (与原版相同)"""
    stages = []
    for onset, duration, description in zip(annotations[0], annotations[1], annotations[2]):
        stage_code = map_annotation_to_stage(description)
        num_epochs = int(duration / epoch_duration)
        stages.extend([stage_code] * num_epochs)
    return stages


def map_annotation_to_stage(annotation):
    # 原始注释文本 -> {0,1,2,3,4,-1}。
    """
    将 ISRUC_S1 标签映射到标准睡眠阶段:
    W=0, N1=1, N2=2, N3=3, REM=4
    未知/无效标签返回 -1

    兼容以下写法：
        0/1/2/3/4/5
        W/N1/N2/N3/R/REM
        Sleep stage W / N1 / N2 / N3 / R
        Stage 1/2/3/4
    其中 4 和 Stage 4 会并入 N3；5 会映射到 REM。
    """
    s = str(annotation).replace('﻿', '').strip().upper()
    if s == '':
        return -1

    s = s.replace('_', ' ').replace('-', ' ')
    s = ' '.join(s.split())

    direct_map = {
        '0': 0, 'W': 0, 'WAKE': 0, 'STAGE W': 0, 'SLEEP STAGE W': 0, 'SLEEP STAGE WAKE': 0,
        '1': 1, 'N1': 1, 'S1': 1, 'STAGE 1': 1, 'SLEEP STAGE 1': 1, 'SLEEP STAGE N1': 1,
        '2': 2, 'N2': 2, 'S2': 2, 'STAGE 2': 2, 'SLEEP STAGE 2': 2, 'SLEEP STAGE N2': 2,
        '3': 3, '4': 3, 'N3': 3, 'S3': 3, 'S4': 3, 'STAGE 3': 3, 'STAGE 4': 3,
        'SLEEP STAGE 3': 3, 'SLEEP STAGE 4': 3, 'SLEEP STAGE N3': 3,
        '5': 4, 'R': 4, 'REM': 4, 'STAGE R': 4, 'SLEEP STAGE R': 4, 'SLEEP STAGE REM': 4,
    }
    if s in direct_map:
        return direct_map[s]

    if 'SLEEP STAGE W' in s or s.endswith(' W'):
        return 0
    if 'SLEEP STAGE N1' in s or s.endswith(' N1'):
        return 1
    if 'SLEEP STAGE N2' in s or s.endswith(' N2'):
        return 2
    if 'SLEEP STAGE N3' in s or s.endswith(' N3'):
        return 3
    if 'SLEEP STAGE R' in s or 'SLEEP STAGE REM' in s or s.endswith(' REM') or s.endswith(' R'):
        return 4

    nums = re.findall(r'\b[0-5]\b', s)
    if len(nums) == 1 and nums[0] in direct_map:
        return direct_map[nums[0]]

    return -1



def load_preprocessed_data(data_path, subject_id=None):
    # 回读保存好的 *_morphology.npz。
    """加载预处理后的数据 (适配新格式)"""
    if subject_id:
        file_path = os.path.join(data_path, f"{subject_id}_morphology.npz")
        if os.path.exists(file_path):
            return np.load(file_path, allow_pickle=True)
        else:
            raise FileNotFoundError(f"Preprocessed data for {subject_id} not found at {file_path}")
    else:
        global_path = os.path.join(data_path, "global_morphology_info.npz")
        if not os.path.exists(global_path):
            raise FileNotFoundError("Global info not found. Run preprocessing first.")
        return np.load(global_path, allow_pickle=True)


if __name__ == "__main__":
    config = PreprocessConfig()
    config.label_csv = None

    data_path = r"D:/dataset/ISRUC_S1/"
    output_path = r"D:/dataset/ISRUC_S1/processed_morphology_CLAS_EOG_five/"

    try:
        all_subjects_data, all_descriptions, all_features = preprocess_sleep_edf_dataset(
            data_path, output_path, config
        )
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
