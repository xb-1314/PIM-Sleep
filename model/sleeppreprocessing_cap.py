"""
sleep_edf_preprocessor_morphology.py - 形态学检测版
专门针对Sleep-EDF扩展数据集的预处理，使用形态学波形检测替代特征聚类。
输出为自然语言描述，供后续大语言模型(LLM)使用。
"""

import os
import glob
import re
import gc
import numpy as np
import json
import pandas as pd
import warnings
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
        # 断点续跑：已有输出包含有效专家分期时直接复用；全为-1的旧输出会重建。
        self.resume_existing = True
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
    # 收集 CAP 的 PSG/EDF 主文件。
    """
    查找MASS3数据集中的PSG文件
    例如: SS3_0001 PSG.edf
    """
    patterns = [
        "* PSG.edf",
        "*.edf"
    ]

    psg_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(data_path, pattern))
        files = [f for f in files if 'Hypnogram' not in f and 'hypnogram' not in f.lower()]
        psg_files.extend(files)

    psg_files = list(set(psg_files))
    psg_files.sort()

    print(f"Found {len(psg_files)} CAP PSG files:")
    for f in psg_files[:5]:
        print(f"  {os.path.basename(f)}")
    if len(psg_files) > 5:
        print(f"  ... and {len(psg_files) - 5} more")

    return psg_files
def pick_channel(ch_names, candidates):
    # 按候选优先级选通道。
    normalized = {
        re.sub(r"[^A-Z0-9]", "", str(name).upper()): name
        for name in ch_names
    }
    for ch in candidates:
        if ch in ch_names:
            return ch
        key = re.sub(r"[^A-Z0-9]", "", str(ch).upper())
        if key in normalized:
            return normalized[key]
    return None


def stage_str_to_int(stage):
    # 分期文本 -> {0,1,2,3,4,-1}。
    stage = str(stage).upper()
    mapping = {
        "WAKE": 0, "W": 0,
        "S0": 0,
        "N1": 1, "S1": 1,
        "N2": 2, "S2": 2,
        "N3": 3, "S3": 3, "S4": 3,
        "REM": 4, "R": 4
    }
    return mapping.get(stage, -1)
def normalize_stage_str(stage):
    # 统一到 W/N1/N2/N3/REM。
    s = str(stage).strip().upper()
    mapping = {
        "WAKE": "W",
        "W": "W", "S0": "W",
        "N1": "N1", "S1": "N1",
        "N2": "N2", "S2": "N2",
        "N3": "N3", "S3": "N3", "S4": "N3",
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
    # 读 EDF、选通道、跑 YASA，返回 epoch 级分期。
    """
    新版：
    1) 用 MNE 读连续信号
    2) 用 YASA SleepStaging 自动分期
    3) 不再手工切 epoch 做检测
    返回:
        raw, hyp, proba, eeg_name, loc_name, roc_name, emg_name
    """
    raw = None
    try:
        # CAP recordings can contain 20-36 high-rate channels. Loading and
        # resampling every channel as float64 needs several GiB per subject.
        # Read only the header first, select the channels actually consumed by
        # this pipeline, and preload/resample that small subset afterwards.
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        ch_names = raw.ch_names
        print(f"All channels: {ch_names}")

        eeg_name = pick_channel(ch_names, [
            'C4-A1', 'C4-M1', 'EEG C4-A1', 'EEG C4', 'C4','C4A1','C3-A2'
        ])
        if eeg_name is None:
            raise ValueError(f"未找到合适 EEG 通道: {ch_names}")

        loc_name = pick_channel(ch_names, [
            'EOG sin','LOC-A1','EOG-L','LOC','LOC / A2','EOG Left', 'LOC-A2', 'EOG1', 'EOG'
        ])

        roc_name = pick_channel(ch_names, [
            'EOG dx','ROC-A2','EOG-R','ROC','ROC / A1','EOG Right', 'ROC-A1', 'EOG2'
        ])

        # CAP channel spelling is not fully uniform across recordings. If an
        # exact alias was unavailable, use remaining channels explicitly
        # marked as EOG/LOC/ROC without ever duplicating one channel twice.
        eog_like = [
            name for name in ch_names
            if re.search(r"EOG|LOC|ROC", str(name), flags=re.IGNORECASE)
        ]
        if loc_name is None and eog_like:
            loc_name = eog_like[0]
        if roc_name is None:
            roc_name = next(
                (name for name in eog_like if name != loc_name), None
            )

        emg_name = pick_channel(ch_names, [
            'EMG', 'Chin EMG', 'EMG Chin', 'EMG Chin1', 'EMG Chin2'
        ])

        print(f"EEG channel: {eeg_name}")
        print(f"EOG LOC: {loc_name}, EOG ROC: {roc_name}, EMG: {emg_name}")
        if loc_name is None or roc_name is None:
            print("Warning: LOC/ROC 不完整，YASA REM 检测将被跳过。")

        selected_channels = list(dict.fromkeys(
            name for name in [eeg_name, loc_name, roc_name, emg_name]
            if name is not None
        ))
        raw.pick(selected_channels)
        raw.load_data()
        target_fs = float(getattr(config, "target_fs", 100))
        if not np.isclose(float(raw.info["sfreq"]), target_fs):
            print(
                f"Resampling selected channels {raw.info['sfreq']:.2f} Hz "
                f"-> {target_fs:.2f} Hz"
            )
            raw.resample(target_fs, npad="auto")

        # 不要对 raw 先手工 bandpass 再分期
        sls = yasa.SleepStaging(
            raw,
            eeg_name=eeg_name,
            eog_name=loc_name,
            emg_name=emg_name
        )

        hyp = sls.predict()  # {n_epochs}
        proba = sls.predict_proba()  # {n_epochs, 5}

        print(f"YASA sleep staging done. Total epochs: {len(hyp)}")

        # ===== 新增：统计每个睡眠期有多少个 epoch =====
        hyp_series = hyp.hypno if hasattr(hyp, "hypno") else pd.Series(hyp)
        hyp_series_norm = hyp_series.astype(str).map(normalize_stage_str)
        stage_counts = hyp_series_norm.value_counts()

        print("Sleep stage counts (epochs):")
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            print(f"  {stage}: {int(stage_counts.get(stage, 0))}")
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
        if raw is not None:
            raw.close()
        del raw
        gc.collect()
        return None, None, None, None, None, None, None


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
    CAP Model B three-class labels:
    - n[0-9]*    -> HC(0)
    - nfle*      -> NFLE(1)
    - rbd*       -> RBD(2)

    注意顺序必须先判断 nfle，再判断 n，
    否则 nfle25 会被误判成 normal。
    """
    sid = str(subject_id).strip().lower()

    if sid.startswith("nfle"):
        return 1, "NFLE"

    if sid.startswith("rbd"):
        return 2, "RBD"

    if re.match(r"^n\d+$", sid):
        return 0, "HC"

    return -1, "unknown"
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

    parts.append("consistency " + ("|".join(consistency) if consistency else "none"))
    return "; ".join(parts)
def build_epoch_feature_table(
    hyp,
    alpha_df=None,
    theta_df=None,
    sem_df=None,
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

        for _, evt in df.iterrows():
            start = float(evt["Start"])
            end = float(evt["End"])

            start_epoch = int(start // 30)
            end_epoch = int(end // 30)

            for ei in range(start_epoch, end_epoch + 1):
                if 0 <= ei < len(rows):
                    rows[ei][f"has_{prefix}"] = 1
                    rows[ei][f"{prefix}_count"] += 1

        for r in rows:
            r[f"{prefix}_density"] = r[f"{prefix}_count"] / 0.5

    add_events(alpha_df, "alpha")
    add_events(theta_df, "theta")
    add_events(sem_df, "sem")
    add_events(sp_df, "spindle")
    add_events(sw_df, "slowwave")
    add_events(rem_df, "rem")

    for row in rows:
        for key in ["spindle", "slowwave", "rem"]:
            count = row[f"{key}_count"]
            row[f"has_{key}"] = 1 if count > 0 else 0
            row[f"{key}_density"] = count / 0.5  # 30秒 = 0.5分钟

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


def inspect_existing_cap_output(output_path):
    """Return epoch count when an existing subject cache is complete and usable."""
    if not os.path.exists(output_path):
        return None
    required_keys = {
        "true_sleep_stages",
        "eeg_epochs",
        "eog_loc_epochs",
        "eog_roc_epochs",
        "sleep_disorder_label",
    }
    try:
        with np.load(output_path, allow_pickle=True) as cached:
            if not required_keys.issubset(set(cached.files)):
                return None
            stages = np.asarray(
                cached["true_sleep_stages"], dtype=np.int64
            ).reshape(-1)
            if stages.size == 0 or not np.isin(stages, [0, 1, 2, 3, 4]).any():
                return None
            return int(stages.size)
    except Exception as exc:
        print(f"  Existing cache is unreadable and will be rebuilt: {exc}")
        return None


def preprocess_sleep_edf_dataset(data_path, output_path, config=None):
    # 主流程：读 EDF -> 检测事件 -> 对齐标签 -> 保存。
    if config is None:
        config = PreprocessConfig()

    os.makedirs(output_path, exist_ok=True)
    psg_files = find_psg_files(data_path)

    if len(psg_files) == 0:
        print("No PSG files found.")
        return [], [], {}

    # Keep only lightweight summaries in RAM. The former implementation kept
    # every subject's full-night arrays and every epoch dictionary alive until
    # the end of the run, which exhausted memory midway through CAP.
    all_subjects_data = []
    total_epoch_count = 0
    sample_description = ""
    sample_features = {}
    subject_info = []

    for i, psg_file in enumerate(psg_files):
        subject_id = os.path.basename(psg_file).replace('-PSG.edf', '').replace('.edf', '')
        print(f"\nProcessing {i + 1}/{len(psg_files)}: {subject_id}")

        sleep_disorder_label, sleep_disorder_name = load_subject_level_label(
            subject_id,
            getattr(config, "label_csv", None),
        )
        if sleep_disorder_label not in {0, 1, 2}:
            print("  Skip: CAP Model B uses only HC, NFLE, and RBD subjects.")
            continue

        subject_output_path = os.path.join(
            output_path, f"{subject_id}_morphology.npz"
        )
        if bool(getattr(config, "resume_existing", True)):
            cached_epochs = inspect_existing_cap_output(subject_output_path)
            if cached_epochs is not None:
                print(
                    f"  Reuse existing valid cache: {subject_output_path} "
                    f"(epochs={cached_epochs})"
                )
                all_subjects_data.append({
                    "subject_id": subject_id,
                    "output_path": subject_output_path,
                    "num_epochs": int(cached_epochs),
                    "reused": True,
                })
                subject_info.append(subject_id)
                total_epoch_count += int(cached_epochs)
                continue

        raw, hyp, proba, eeg_name, loc_name, roc_name, emg_name = load_sleep_edf_data(psg_file, config)
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
        theta_df = detector.detect_theta_events_continuous(raw, hyp, eeg_name)

        sem_df = pd.DataFrame()
        if loc_name is not None and roc_name is not None:
            sem_df = detector.detect_sem_events_continuous(raw, hyp, loc_name, roc_name)

        # 这里再统计，不能提前
        alpha_count = len(alpha_df) if alpha_df is not None else 0
        theta_count = len(theta_df) if theta_df is not None else 0
        sem_count = len(sem_df) if sem_df is not None else 0
        sp_count = len(sp_df) if sp_df is not None else 0
        sw_count = len(sw_df) if sw_df is not None else 0
        rem_count = len(rem_df) if rem_df is not None else 0

        print("  Detected event counts:")
        print(f"    Alpha    : {alpha_count}")
        print(f"    Theta    : {theta_count}")
        print(f"    SEMs     : {sem_count}")
        print(f"    Spindles : {sp_count}")
        print(f"    SlowWaves: {sw_count}")
        print(f"    REMs     : {rem_count}")

        epoch_df = build_epoch_feature_table(
            hyp,
            alpha_df=alpha_df,
            theta_df=theta_df,
            sem_df=sem_df,
            sp_df=sp_df,
            sw_df=sw_df,
            rem_df=rem_df
        )  # {n_epochs, n_features}

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
            "alpha_count", "theta_count", "sem_count", "vertex_count",
            "spindle_count", "slowwave_count", "rem_count"
        ]]
            .sum()
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
                    f"spindle={int(row['spindle_count'])}, "
                    f"slowwave={int(row['slowwave_count'])}, "
                    f"rem={int(row['rem_count'])}"
                )
            else:
                print(f"{stage}: alpha=0, theta=0, sem=0, vertex=0, spindle=0, slowwave=0, rem=0")

        eeg = raw.copy().pick([eeg_name]).get_data(units="uV").squeeze()
        epoch_len = int(raw.info["sfreq"] * 30)  # 3000

        for i in range(len(epoch_df)):
            s = i * epoch_len
            e = s + epoch_len
            eeg_epoch = eeg[s:e]  # {3000}

            band_feat = detector.compute_band_features(eeg_epoch)
            for k, v in band_feat.items():
                epoch_df.loc[i, k] = v
        for i in range(len(epoch_df)):
            s = i * epoch_len
            e = s + epoch_len
            eeg_epoch = eeg[s:e]  # {3000}

            stage_val = int(epoch_df.loc[i, "sleep_stage"])
            if stage_val == 1:
                vertex_count, vertex_events = detector.detect_vertex_waves(eeg_epoch)
            else:
                vertex_count, vertex_events = 0, []

            epoch_df.loc[i, "has_vertex"] = int(vertex_count > 0)
            epoch_df.loc[i, "vertex_count"] = int(vertex_count)
            epoch_df.loc[i, "vertex_density"] = float(vertex_count / 0.5)
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

        eeg_epochs = extract_epoch_signal_array(
            raw,
            eeg_name,
            epoch_duration=int(getattr(config, "epoch_duration", 30)),
            target_n_epochs=len(true_sleep_stages)
        )  # {n_epochs, 3000}
        if loc_name is not None:
            eog_loc_epochs = extract_epoch_signal_array(
                raw,
                loc_name,
                epoch_duration=int(getattr(config, "epoch_duration", 30)),
                target_n_epochs=len(true_sleep_stages),
            )
        else:
            eog_loc_epochs = np.zeros_like(eeg_epochs)
        if roc_name is not None:
            eog_roc_epochs = extract_epoch_signal_array(
                raw,
                roc_name,
                epoch_duration=int(getattr(config, "epoch_duration", 30)),
                target_n_epochs=len(true_sleep_stages),
            )
        else:
            eog_roc_epochs = np.zeros_like(eeg_epochs)
        eog_available = np.asarray(
            [float(loc_name is not None), float(roc_name is not None)],
            dtype=np.float32,
        )

        aligned_n = min(
            len(true_sleep_stages),
            len(eeg_epochs),
            len(eog_loc_epochs),
            len(eog_roc_epochs),
        )
        true_sleep_stages = true_sleep_stages[:aligned_n]
        eeg_epochs = eeg_epochs[:aligned_n]
        eog_loc_epochs = eog_loc_epochs[:aligned_n]
        eog_roc_epochs = eog_roc_epochs[:aligned_n]
        descriptions_plain = descriptions_plain[:aligned_n]
        descriptions_with_stage = descriptions_with_stage[:aligned_n]
        all_waveform_features = all_waveform_features[:aligned_n]
        sleep_stages = sleep_stages[:aligned_n]
        sleep_stage_str = sleep_stage_str[:aligned_n]
        epoch_df = epoch_df.iloc[:aligned_n].copy()

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
            "eeg_channel": np.array(eeg_name),
            "eog_loc_channel": np.array(loc_name or ""),
            "eog_roc_channel": np.array(roc_name or ""),
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
        sw_df.to_csv(os.path.join(output_path, f"{subject_id}_slowwaves.csv"), index=False)
        rem_df.to_csv(os.path.join(output_path, f"{subject_id}_rem.csv"), index=False)

        all_subjects_data.append({
            "subject_id": subject_id,
            "output_path": subject_output_path,
            "num_epochs": int(aligned_n),
            "reused": False,
        })
        total_epoch_count += int(aligned_n)
        if not sample_description and descriptions_plain:
            sample_description = descriptions_plain[0]
        if not sample_features and all_waveform_features:
            sample_features = all_waveform_features[0]
        subject_info.append(subject_id)

        print(f"  Saved: {subject_output_path}")
        print(f"  Saved: {jsonl_path}")

        # The compressed NPZ is now authoritative. Release the current
        # subject's continuous recording, epoch arrays and feature tables
        # before opening the next EDF.
        raw.close()
        del (
            raw, hyp, proba, detector, epoch_df, subject_data,
            eeg, eeg_epochs, eog_loc_epochs, eog_roc_epochs,
            descriptions_plain, descriptions_with_stage,
            all_waveform_features,
        )
        gc.collect()

    if all_subjects_data:
        global_info = {
            "subject_ids": subject_info,
            "total_epochs": int(total_epoch_count),
            "config": config.__dict__,
            "sample_description": sample_description,
            "sample_features": sample_features,
        }
        np.savez_compressed(os.path.join(output_path, "global_morphology_info.npz"), **global_info)

    print("\n预处理完成!")
    print(f"处理了 {len(all_subjects_data)} 个被试")
    print(f"总生成/复用 {total_epoch_count} 个epoch")
    return all_subjects_data, [], []

# ===================== 以下为与原版兼容的辅助函数，未作核心修改 =====================

def load_hypnogram_data(edf_path, config):
    """Read CAP expert 30-second R&K stages from the RemLogic text export.

    Only rows whose Event column starts with ``SLEEP-`` are stage rows; CAP
    event rows (MCAP, apnea, movements, and so on) are ignored. S3 and S4 are
    merged into the standard five-class N3 label used by the downstream model.
    """
    try:
        subject_base = os.path.splitext(os.path.basename(edf_path))[0]
        label_path = os.path.join(os.path.dirname(edf_path), f"{subject_base}.txt")
        if not os.path.exists(label_path):
            print(f"  Warning: CAP expert stage file not found for {subject_base}")
            print(f"  Expected: {label_path}")
            return None
        sleep_stages = []
        with open(label_path, "r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                fields = [field.strip() for field in line.rstrip("\r\n").split("\t")]
                if len(fields) < 3:
                    continue
                stage_text = fields[0]
                # CAP RemLogic exports are not column-uniform. Most files use
                # ``Stage, Position, Time, Event, ...`` while some omit the
                # Position column and put Event at index 2. Locate the SLEEP-*
                # field by content instead of assuming a fixed column index.
                event_text = next(
                    (
                        field.upper() for field in fields[1:]
                        if field.upper().startswith("SLEEP-")
                    ),
                    "",
                )
                if not event_text:
                    continue
                sleep_stages.append(map_annotation_to_stage(stage_text))
        if not sleep_stages:
            raise ValueError(f"No SLEEP-* stage rows found in {label_path}")
        valid_count = sum(1 for x in sleep_stages if x != -1)
        invalid_count = sum(1 for x in sleep_stages if x == -1)
        print(
            f"  Loaded {len(sleep_stages)} CAP expert stage epochs from "
            f"{os.path.basename(label_path)}"
        )
        print(f"  Valid epochs: {valid_count}, invalid/unknown epochs: {invalid_count}")
        return sleep_stages
    except Exception as e:
        print(f"Error loading CAP hypnogram data: {e}")
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
    将MASS3标签映射到标准睡眠阶段:
    W=0, N1=1, N2=2, N3=3, REM=4
    未知/无效标签返回 -1
    """
    annotation = str(annotation).strip().upper()

    if annotation in {'W', 'S0', '0', 'WAKE', 'SLEEP-S0'}:
        return 0
    elif annotation in {'1', 'S1', 'N1', 'SLEEP-S1'}:
        return 1
    elif annotation in {'2', 'S2', 'N2', 'SLEEP-S2'}:
        return 2
    elif annotation in {'3', '4', 'S3', 'S4', 'N3', 'SLEEP-S3', 'SLEEP-S4'}:
        return 3
    elif annotation in {'R', 'REM', 'SLEEP-REM'}:
        return 4
    elif annotation in {'?', 'MT', 'MOVEMENT', 'UNSCORED', 'SLEEP-UNSCORED'}:
        return -1
    else:
        # 兼容其它可能写法
        if annotation in ['WAKE', 'STAGE W', 'SLEEP STAGE W']:
            return 0
        elif annotation in ['N1', 'S1', 'STAGE 1', 'SLEEP STAGE 1']:
            return 1
        elif annotation in ['N2', 'S2', 'STAGE 2', 'SLEEP STAGE 2']:
            return 2
        elif annotation in ['N3', 'S3', 'STAGE 3', 'STAGE 4', 'SLEEP STAGE 3', 'SLEEP STAGE 4']:
            return 3
        elif annotation in ['REM', 'STAGE R', 'SLEEP STAGE R']:
            return 4

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

    # 必须提供被试级二分类标签
    config.label_csv = None

    data_path = r"C:/Users/1/Desktop/cap/cap-sleep-database-1.0.0/"
    output_path = r"D:/dataset/cap/"

    try:
        all_subjects_data, all_descriptions, all_features = preprocess_sleep_edf_dataset(
            data_path, output_path, config
        )
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
