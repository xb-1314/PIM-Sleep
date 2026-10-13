import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import mne
import yasa

from morphology_detector_yasa import MorphologyDetector

warnings.filterwarnings("ignore")


# =========================
# 配置
# =========================
class PreprocessConfig:
    def __init__(self):
        self.target_fs = 100
        self.epoch_duration = 30
        self.label_csv = None

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


# =========================
# 基础工具
# =========================
def find_psg_files(data_path):
    # 收集 CLAS 的 EDF 主文件。
    patterns = ["*.edf", "*.EDF"]
    psg_files = []
    for pattern in patterns:
        psg_files.extend(glob.glob(os.path.join(data_path, pattern)))
    psg_files = sorted(list(set(psg_files)))
    print(f"Found {len(psg_files)} EDF files in {data_path}:")
    for f in psg_files[:10]:
        print(f"  {os.path.basename(f)}")
    if len(psg_files) > 10:
        print(f"  ... and {len(psg_files) - 10} more")
    return psg_files


def pick_channel(ch_names, candidates):
    # 按候选优先级选通道。
    for ch in candidates:
        if ch in ch_names:
            return ch

    lower_map = {c.lower(): c for c in ch_names}
    for ch in candidates:
        if ch.lower() in lower_map:
            return lower_map[ch.lower()]

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


def robust_std(x):
    # 用 5%~95% 区间估计稳健标准差。
    x = np.asarray(x).ravel()
    lo, hi = np.percentile(x, [5, 95])
    x2 = x[(x >= lo) & (x <= hi)]
    if x2.size == 0:
        return np.std(x)
    return np.std(x2)


def crop_raw_to_full_epochs(raw, epoch_sec=30):
    # 裁成整段 30s epoch。
    sf = raw.info["sfreq"]
    n_epochs = int(np.floor(raw.n_times / sf / epoch_sec))
    if n_epochs <= 0:
        raise ValueError("EDF 时长不足，无法切成完整 30 秒 epoch。")

    tmax = n_epochs * epoch_sec - 1.0 / sf
    raw = raw.copy().crop(tmin=0, tmax=tmax)
    return raw, n_epochs


def choose_signal_in_uv(raw, ch_name, kind="eeg"):
    # 自动判断是否需要乘 1e6 转到 uV。
    x_native = raw.get_data(picks=[ch_name]).squeeze()
    x_scaled = x_native * 1e6

    target_std = 20.0 if kind == "eeg" else 40.0

    def score(x):
        s = max(robust_std(x), 1e-12)
        return abs(np.log10(s / target_std))

    candidates = {
        "native_as_uV": x_native,
        "native_times_1e6": x_scaled,
    }

    mode = min(candidates.keys(), key=lambda k: score(candidates[k]))
    signal_uv = candidates[mode]

    print(
        f"{ch_name} -> 使用 {mode} 作为 uV；"
        f" trimmed_std={robust_std(signal_uv):.3f} uV；"
        f" ptp={np.ptp(signal_uv):.3f} uV"
    )
    return signal_uv, mode


# =========================
# CLAS 标签读取
# =========================
def get_clas_txt_path(edf_path):
    # 从 EDF 文件名推 CLAS 同名 txt 路径。
    base_dir = os.path.dirname(edf_path)
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    txt_path = os.path.join(base_dir, f"{stem}.txt")
    return txt_path


def map_clas_stage_to_int(stage_name):
    # CLAS 标签 -> {0,1,2,3,4,-1}。
    s = str(stage_name).strip().upper()

    if s in ["WAKE", "W"]:
        return 0
    if s in ["NONREM1", "N1", "REM1", "S1"]:
        return 1
    if s in ["NONREM2", "N2", "REM2", "S2"]:
        return 2
    if s in ["NONREM3", "N3", "S3", "S4", "NONREM4"]:
        return 3
    if s in ["REM", "R"]:
        return 4

    return -1


def load_hypnogram_data(edf_path, config):
    # 读取 CLAS 真实分期，输出 {n_epochs}。
    """
    读取 CLAS 同名 txt:
    每行格式: StageName <tab> start_sec <tab> end_sec
    例如: Wake  0  2730
    转成 30 秒 epoch 标签
    """
    txt_path = get_clas_txt_path(edf_path)

    if not os.path.exists(txt_path):
        print(f"  Warning: CLAS txt label file not found: {txt_path}")
        return None

    try:
        intervals = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                stage_name = parts[0]
                try:
                    start_sec = float(parts[1])
                    end_sec = float(parts[2])
                except Exception:
                    continue

                intervals.append((stage_name, start_sec, end_sec))

        if len(intervals) == 0:
            print(f"  Warning: no valid rows in {txt_path}")
            return None

        max_end = max(x[2] for x in intervals)
        n_epochs = int(np.floor(max_end / config.epoch_duration))
        sleep_stages = np.full(n_epochs, -1, dtype=int)

        for stage_name, start_sec, end_sec in intervals:
            stage_code = map_clas_stage_to_int(stage_name)

            start_epoch = int(np.floor(start_sec / config.epoch_duration))
            end_epoch_exclusive = int(np.ceil(end_sec / config.epoch_duration))

            start_epoch = max(start_epoch, 0)
            end_epoch_exclusive = min(end_epoch_exclusive, n_epochs)

            if start_epoch >= end_epoch_exclusive:
                continue

            sleep_stages[start_epoch:end_epoch_exclusive] = stage_code

        sleep_stages = sleep_stages.tolist()

        valid_count = sum(1 for x in sleep_stages if x != -1)
        invalid_count = sum(1 for x in sleep_stages if x == -1)

        print(f"  Found txt file: {os.path.basename(txt_path)}")
        print(f"  Parsed {len(intervals)} label intervals")
        print(f"  Converted to {len(sleep_stages)} labels at 30s resolution")
        print(f"  Valid epochs: {valid_count}, invalid/unknown epochs: {invalid_count}")

        stage_counts = pd.Series(sleep_stages).value_counts().to_dict()
        print("  Stage counts (30s):")
        print(f"    W   : {stage_counts.get(0, 0)}")
        print(f"    N1  : {stage_counts.get(1, 0)}")
        print(f"    N2  : {stage_counts.get(2, 0)}")
        print(f"    N3  : {stage_counts.get(3, 0)}")
        print(f"    REM : {stage_counts.get(4, 0)}")
        print(f"    UNK : {stage_counts.get(-1, 0)}")

        return sleep_stages

    except Exception as e:
        print(f"Error loading CLAS hypnogram data: {e}")
        import traceback
        traceback.print_exc()
        return None


# =========================
# EDF 读取与自动分期
# =========================


def extract_epoch_signal_array(
    raw,
    ch_name,
    epoch_duration=30,
    target_n_epochs=None,
    signal_kind="eeg",
):
    # 连续 EEG/EOG -> {n_epochs, 3000} float32。
    """
    从连续 raw 中提取单通道 30 秒 EEG/EOG epoch，并尽量统一到 uV 尺度。
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

    target_std = 20.0 if str(signal_kind).lower() == "eeg" else 40.0

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
    # 读 EDF、重采样、裁整段、跑 YASA。
    """
    CLAS 版本：
    1) 先 preload=False 读头，降低内存风险
    2) 选 EEG/EOG/EMG 通道
    3) 再 load_data + resample
    4) 裁成完整 30 秒 epoch
    5) YASA 自动分期
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        ch_names = raw.ch_names
        print(f"All channels: {ch_names}")

        eeg_name = pick_channel(ch_names, [
            "C3-A2", "C3-A1", "C4-A1", "C4-M1",
            "EEG C3-A2", "EEG C3-A1", "EEG C4-A1", "EEG C4",
            "C3", "C4", "CZ-A1", "Cz-A1", "Cz", "CZ"
        ])
        if eeg_name is None:
            raise ValueError(f"未找到合适 EEG 通道: {ch_names}")

        loc_name = pick_channel(ch_names, [
            "EOG_L","EOG Left", "LOC-A2", "LOC", "EOG1", "LEOG", "Left EOG"
        ])
        roc_name = pick_channel(ch_names, [
            "EOG_R","EOG Right", "ROC-A1", "ROC", "EOG2", "REOG", "Right EOG"
        ])
        emg_name = pick_channel(ch_names, [
            "EMG", "EMG Chin", "Chin EMG", "EMG Chin1", "EMG Chin2", "Submental EMG"
        ])

        print(f"EEG channel: {eeg_name}")
        print(f"EOG LOC: {loc_name}, EOG ROC: {roc_name}, EMG: {emg_name}")

        keep_channels = [eeg_name]
        if loc_name is not None:
            keep_channels.append(loc_name)
        if roc_name is not None and roc_name not in keep_channels:
            keep_channels.append(roc_name)
        if emg_name is not None and emg_name not in keep_channels:
            keep_channels.append(emg_name)

        raw.pick(keep_channels)
        print(f"Picked channels only: {raw.ch_names}")

        raw.load_data()

        if raw.info["sfreq"] > config.target_fs:
            print(f"Resampling from {raw.info['sfreq']} Hz to {config.target_fs} Hz")
            raw.resample(config.target_fs, npad="auto")

        print(f"Current sfreq after resample: {raw.info['sfreq']} Hz")

        raw, n_epochs = crop_raw_to_full_epochs(raw, epoch_sec=config.epoch_duration)
        print(f"Complete 30s epochs from raw: {n_epochs}")

        sls = yasa.SleepStaging(
            raw,
            eeg_name=eeg_name,
            eog_name=loc_name,
            emg_name=emg_name
        )

        hyp = sls.predict()  # {n_epochs}
        proba = sls.predict_proba()  # {n_epochs, 5}

        print(f"YASA sleep staging done. Total epochs: {len(hyp)}")

        hyp_series = hyp.hypno if hasattr(hyp, "hypno") else pd.Series(hyp)
        hyp_series_norm = hyp_series.astype(str).map(normalize_stage_str)
        stage_counts = hyp_series_norm.value_counts()

        print("YASA sleep stage counts (epochs):")
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            count = int(stage_counts.get(stage, 0))
            ratio = count / len(hyp_series) if len(hyp_series) > 0 else 0
            print(f"  {stage}: {count} ({ratio:.2%})")

        return raw, hyp, proba, eeg_name, loc_name, roc_name, emg_name

    except Exception as e:
        print(f"Error loading {edf_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None


# =========================
# 描述与事件表
# =========================
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
            start_sec = float(evt["Start"])
            end_sec = float(evt["End"]) if "End" in evt else start_sec
            center_sec = 0.5 * (start_sec + end_sec)
            epoch_idx = int(center_sec // 30)
            if 0 <= epoch_idx < n_epochs:
                rows[epoch_idx][f"{prefix}_count"] += 1

    add_events(alpha_df, "alpha")
    add_events(theta_df, "theta")
    add_events(sem_df, "sem")
    add_events(sp_df, "spindle")
    add_events(sw_df, "slowwave")
    add_events(rem_df, "rem")

    for row in rows:
        for key in ["alpha", "theta", "sem", "spindle", "slowwave", "rem"]:
            count = row[f"{key}_count"]
            row[f"has_{key}"] = 1 if count > 0 else 0
            row[f"{key}_density"] = count / 0.5

    return pd.DataFrame(rows)  # {n_epochs, n_features}


def load_subject_level_label(subject_id, label_csv=None):
    # 读取被试级障碍标签。
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


# =========================
# 主预处理流程
# =========================
def preprocess_sleep_edf_dataset(data_path, output_path, config=None, only_subject_id=None):
    # 主流程：读 EDF -> 检测事件 -> 对齐标签 -> 保存。
    if config is None:
        config = PreprocessConfig()

    os.makedirs(output_path, exist_ok=True)
    psg_files = find_psg_files(data_path)

    if only_subject_id is not None:
        psg_files = [
            f for f in psg_files
            if os.path.basename(f).replace(".edf", "").replace(".EDF", "") == only_subject_id
        ]
        print(f"仅处理指定被试: {only_subject_id}")
        print(f"筛选后文件数: {len(psg_files)}")

    if len(psg_files) == 0:
        print("No PSG files found.")
        return [], [], {}

    all_subjects_data = []
    global_descriptions = []
    global_features = []
    subject_info = []

    for i, psg_file in enumerate(psg_files):
        subject_id = os.path.basename(psg_file).replace(".edf", "").replace(".EDF", "")
        print(f"\nProcessing {i + 1}/{len(psg_files)}: {subject_id}")

        raw, hyp, proba, eeg_name, loc_name, roc_name, emg_name = load_sleep_edf_data(psg_file, config)
        if raw is None:
            print("  Skipping due to load error")
            continue

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

        eeg, _ = choose_signal_in_uv(raw, eeg_name, kind="eeg")
        epoch_len = int(raw.info["sfreq"] * 30)

        for j in range(len(epoch_df)):
            s = j * epoch_len
            e = s + epoch_len
            eeg_epoch = eeg[s:e]

            band_feat = detector.compute_band_features(eeg_epoch)
            for k, v in band_feat.items():
                epoch_df.loc[j, k] = v

            stage_val = int(epoch_df.loc[j, "sleep_stage"])
            if stage_val == 1:
                vertex_count, _ = detector.detect_vertex_waves(eeg_epoch)
            else:
                vertex_count = 0

            epoch_df.loc[j, "has_vertex"] = int(vertex_count > 0)
            epoch_df.loc[j, "vertex_count"] = int(vertex_count)
            epoch_df.loc[j, "vertex_density"] = float(vertex_count / 0.5)

        if true_sleep_stages is not None:
            min_len = min(len(epoch_df), len(true_sleep_stages))
            if len(epoch_df) != len(true_sleep_stages):
                print(f"  Aligning YASA epochs and true labels: {len(epoch_df)} vs {len(true_sleep_stages)} -> {min_len}")
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
                print(f"    {stage}: alpha=0, theta=0, sem=0, vertex=0, spindle=0, slowwave=0, rem=0")

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
        slow_wave_labels = epoch_df["has_slowwave"].astype(np.int64).to_numpy()  # {n_epochs}
        slow_wave_event_counts = epoch_df["slowwave_count"].astype(np.int64).to_numpy()  # {n_epochs}
        slow_wave_densities = epoch_df["slowwave_density"].astype(np.float32).to_numpy()  # {n_epochs}

        sleep_disorder_label, sleep_disorder_name = load_subject_level_label(
            subject_id,
            getattr(config, "label_csv", None)
        )

        eeg_epochs = extract_epoch_signal_array(
            raw,
            eeg_name,
            epoch_duration=int(getattr(config, "epoch_duration", 30)),
            target_n_epochs=len(true_sleep_stages),
            signal_kind="eeg",
        )  # {n_epochs, 3000}

        # CLAS 多模态输出：LOC/ROC 与 EEG 使用同样的 30 s 边界分段。
        # 某个 EOG 通道缺失时保存全零占位，并通过 eog_available 明确标记，
        # 避免下游把占位数据当成真实 EOG。
        eog_loc_epochs = (
            extract_epoch_signal_array(
                raw,
                loc_name,
                epoch_duration=int(getattr(config, "epoch_duration", 30)),
                target_n_epochs=len(true_sleep_stages),
                signal_kind="eog",
            )
            if loc_name is not None else np.zeros_like(eeg_epochs)
        )
        eog_roc_epochs = (
            extract_epoch_signal_array(
                raw,
                roc_name,
                epoch_duration=int(getattr(config, "epoch_duration", 30)),
                target_n_epochs=len(true_sleep_stages),
                signal_kind="eog",
            )
            if roc_name is not None else np.zeros_like(eeg_epochs)
        )
        eog_available = np.asarray(
            [int(loc_name is not None), int(roc_name is not None)],
            dtype=np.float32,
        )

        # 所有 epoch 级字段必须严格对齐。正常情况下三路信号长度相同；
        # 若 EDF 时长短于标签时长，统一截断到共同的完整 epoch 数。
        aligned_n_epochs = min(
            len(true_sleep_stages),
            len(eeg_epochs),
            len(eog_loc_epochs),
            len(eog_roc_epochs),
        )
        if aligned_n_epochs <= 0:
            print("  Skipping: no aligned EEG/EOG epochs available")
            continue
        if any(
            n != aligned_n_epochs
            for n in (
                len(true_sleep_stages),
                len(eeg_epochs),
                len(eog_loc_epochs),
                len(eog_roc_epochs),
            )
        ):
            print(
                "  Aligning saved modalities: "
                f"labels={len(true_sleep_stages)}, EEG={len(eeg_epochs)}, "
                f"LOC={len(eog_loc_epochs)}, ROC={len(eog_roc_epochs)} "
                f"-> {aligned_n_epochs}"
            )

        epoch_df = epoch_df.iloc[:aligned_n_epochs].reset_index(drop=True)
        true_sleep_stages = true_sleep_stages[:aligned_n_epochs]
        descriptions_plain = descriptions_plain[:aligned_n_epochs]
        descriptions_with_stage = descriptions_with_stage[:aligned_n_epochs]
        all_waveform_features = all_waveform_features[:aligned_n_epochs]
        sleep_stages = sleep_stages[:aligned_n_epochs]
        sleep_stage_str = sleep_stage_str[:aligned_n_epochs]
        slow_wave_labels = slow_wave_labels[:aligned_n_epochs]
        slow_wave_event_counts = slow_wave_event_counts[:aligned_n_epochs]
        slow_wave_densities = slow_wave_densities[:aligned_n_epochs]
        eeg_epochs = eeg_epochs[:aligned_n_epochs]
        eog_loc_epochs = eog_loc_epochs[:aligned_n_epochs]
        eog_roc_epochs = eog_roc_epochs[:aligned_n_epochs]
        proba_aligned = (
            proba.iloc[:aligned_n_epochs]
            if hasattr(proba, "iloc") else proba
        )

        subject_data = {  # 主要按 epoch 对齐
            "subject_id": subject_id,

            "descriptions": descriptions_plain,
            "descriptions_plain": descriptions_plain,
            "descriptions_with_stage": descriptions_with_stage,

            "waveform_features": all_waveform_features,

            "sleep_stages": np.array(true_sleep_stages),  # {n_epochs}
            "true_sleep_stages": np.array(true_sleep_stages),

            "yasa_sleep_stages": np.array(sleep_stages),  # {n_epochs}
            "yasa_sleep_stage_str": np.array(sleep_stage_str),

            "sleep_disorder_label": np.array(sleep_disorder_label),
            "sleep_disorder_name": np.array(sleep_disorder_name),

            "slow_wave_labels": slow_wave_labels,  # {n_epochs}
            "slow_wave_event_counts": slow_wave_event_counts,  # {n_epochs}
            "slow_wave_densities": slow_wave_densities,  # {n_epochs}

            "yasa_proba": (
                proba_aligned.to_dict("list")
                if hasattr(proba_aligned, "to_dict") else None
            ),  # 5 类概率
            "channel": eeg_name,
            "eeg_epochs": eeg_epochs,  # {n_epochs, 3000}
            "eog_loc_epochs": eog_loc_epochs,  # {n_epochs, 3000}
            "eog_roc_epochs": eog_roc_epochs,  # {n_epochs, 3000}
            "eog_available": eog_available,  # [LOC available, ROC available]
            "eog_channels": np.asarray(
                [loc_name or "", roc_name or ""], dtype=str
            ),
            "sampling_rate": raw.info["sfreq"],
            "epoch_duration": 30
        }

        subject_output_path = os.path.join(output_path, f"{subject_id}_morphology.npz")
        np.savez_compressed(subject_output_path, **subject_data)
        print(
            "  Multimodal epochs: "
            f"EEG={eeg_epochs.shape}, LOC={eog_loc_epochs.shape}, "
            f"ROC={eog_roc_epochs.shape}, available={eog_available.astype(int).tolist()}"
        )

        jsonl_path = os.path.join(output_path, f"{subject_id}_descriptions.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx, row in epoch_df.iterrows():
                record = {
                    "epoch_index": int(row["epoch_index"]),
                    "sleep_stage": int(true_sleep_stages[idx]),
                    "true_sleep_stage": int(true_sleep_stages[idx]),
                    "yasa_sleep_stage": int(row["sleep_stage"]),
                    "yasa_sleep_stage_str": str(row["sleep_stage_str"]),
                    "description": row["description_plain"],
                    "description_plain": row["description_plain"],
                    "description_with_stage": row["description_with_stage"],
                    "features": row.to_dict(),
                    "slow_wave_label": int(row["has_slowwave"]),
                    "slow_wave_event_count": int(row["slowwave_count"]),
                    "slow_wave_density": float(row["slowwave_density"]),
                    "sleep_disorder_label": int(sleep_disorder_label),
                    "sleep_disorder_name": str(sleep_disorder_name)
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        sp_df.to_csv(os.path.join(output_path, f"{subject_id}_spindles.csv"), index=False)
        sw_df.to_csv(os.path.join(output_path, f"{subject_id}_slowwaves.csv"), index=False)
        rem_df.to_csv(os.path.join(output_path, f"{subject_id}_rem.csv"), index=False)

        all_subjects_data.append(subject_data)
        global_descriptions.extend(descriptions_plain)
        global_features.extend(all_waveform_features)
        subject_info.append(subject_id)

        print(f"  Saved: {subject_output_path}")
        print(f"  Saved: {jsonl_path}")

    if all_subjects_data:
        global_info = {
            "subject_ids": subject_info,
            "total_epochs": len(global_descriptions),
            "config": config.__dict__,
            "sample_description": global_descriptions[0] if global_descriptions else "",
            "sample_features": global_features[0] if global_features else {}
        }
        np.savez_compressed(os.path.join(output_path, "global_morphology_info.npz"), **global_info)

    print("\n预处理完成!")
    print(f"处理了 {len(all_subjects_data)} 个被试")
    print(f"总生成 {len(global_descriptions)} 条文本描述")
    return all_subjects_data, global_descriptions, global_features


# =========================
# 运行入口
# =========================
if __name__ == "__main__":
    config = PreprocessConfig()
    config.label_csv = None

    data_path = r"D:\dataset\CLAS\sleep_base"
    # 使用新目录，避免覆盖旧的 EEG-only CLAS 预处理结果。
    output_path = r"D:\dataset\CLAS\sleep_base\processed_morphology_CLAS_EOG"

    try:
        all_subjects_data, all_descriptions, all_features = preprocess_sleep_edf_dataset(
            data_path=data_path,
            output_path=output_path,
            config=config,
        )
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
