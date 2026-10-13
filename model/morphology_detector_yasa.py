# morphology_detector.py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks, medfilt, hilbert
from scipy.ndimage import label, find_objects
import warnings
import logging
import os
import contextlib
import sys
# 设置环境变量
os.environ['JOBLIB_TQDM'] = 'False'

# 配置日志
logging.basicConfig(level=logging.ERROR)  # 全局日志级别设为 ERROR
logging.getLogger('yasa').setLevel(logging.ERROR)
logging.getLogger('joblib').setLevel(logging.ERROR)
logging.getLogger('yasa').propagate = False
logging.getLogger('joblib').propagate = False

warnings.filterwarnings('ignore')

@contextlib.contextmanager
def suppress_stdout_stderr():
    """临时屏蔽第三方库在控制台的输出"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
def robust_std(x):
    x = np.asarray(x).ravel()
    lo, hi = np.percentile(x, [5, 95])
    x2 = x[(x >= lo) & (x <= hi)]
    if x2.size == 0:
        return np.std(x)
    return np.std(x2)


def choose_signal_in_uv_from_raw(raw, ch_name, kind="eeg"):
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
class MorphologyDetector:
    """
    睡眠脑电特征波形整合检测器。
    整合了α波、K-复合波、纺锤波、慢波、顶尖波、快速眼动的检测逻辑。
    注意：REM检测需要单独的左右眼电（EOG）通道数据。
    """

    def __init__(self, fs=100,channel_names=None):
        """
        初始化检测器，设置默认参数。
        Args:
            fs: 采样率 (Hz)，默认为100Hz，与您的预处理流程一致。
        """
        self.fs = fs
        self.channel_names = channel_names

        # 定义各波形检测的默认参数（基于您提供的文档，已优化）
        self.waveform_params = {
            'alpha': {
                'freq_band': (8, 13),  # α波频率带
                'target_stages': [0, 1],  # 检测目标分期：W(0), N1(1)
                'min_duration': 0.5,  # 最小持续时长(秒)
                'max_duration': 3.0,
                'amp_threshold': 1,  # 幅值阈值：均值 + N倍标准差
                'butter_order': 4
            },
            'k_complex': {
                'freq_band': (0.3, 4.0),  # K-复合波慢波成分
                'target_stages': [2],  # 检测目标分期：N2(2)
                'min_negative_amp': 40.0,
                'min_positive_amp': 20.0,
                'min_peak_to_peak_amp': 75.0,
                'min_duration': 0.5,
                'max_duration': 1.5,
                'peak_distance': 0.7,  # 峰间距最小(秒)
                'positive_delay_min': 0.05,
                'positive_delay_max': 1.0,
                'peak_prominence': 20.0,
                'butter_order': 4
            },
            'spindle': {  # 参数需结合YASA或自实现逻辑，此处为示例
                'freq_band': (12, 15),
                'min_duration': 0.5,
                'max_duration': 2.0,
                # 'threshold_std': 1.5,
                'target_stages':[1,2,3],
                'butter_order': 4
            },
            'slow_wave': {  # 参数需结合YASA或自实现逻辑，此处为示例
                'freq_band': (0.3, 1.5),
                'min_duration': 0.5,
                'max_duration': 2.0,
                'height_std': 1.8,
                'target_stages':[2,3],
                'butter_order': 4
            },
            'vertex': {  # 来自 vertex_sharp_detect.py
                'freq_band': (0.5, 20.0),
                'target_stages': [1],  # 主要出现在N1期
                'min_duration': 0.05,  # 最短持续时间(秒)
                'max_duration': 0.5,  # 最长持续时间(秒)
                'neg_threshold_uv': 50,  # 负向尖峰最小幅值(μV)
                'threshold_std': 2.5,
                'slope_threshold': 150.0,  # 斜率阈值
                'peak_prominence': 15,
                'peak_distance': 0.2,
                'butter_order': 2,
            },
            'rem': {
                'filter_band': (0.1, 30),
                'amplitude_thr': 8,
                'instant_slope_thr': 80,
                'avg_slope_thr': 80,
                'amplitude_time_win': 0.8,
                'avg_slope_time_win': 0.2,
                'moving_avg_window': 10,
                'em_gap': 0.05,
                'final_merge_gap': 0.1,
                'onset_max_move': 0.1,
                'conjugate_slope_thr': 15,
                'conjugate_pre_win': 0.05,
                'conjugate_post_win': 0.15,
                'event_duration': 0.4,
                'slope_product_thr': -100
            },
            'theta': {
                'freq_band': (3, 7),
                'target_stages': [1],  # N1
                'min_duration': 0.5,
                'amp_threshold': 1.0,
                'butter_order': 4
            },
            'sem': {
                'freq_band': (0.1, 1.0),
                'target_stages': [1],  # N1
                'min_duration': 0.5,
                'max_duration': 2.0,
                'amp_threshold': 0.8,
                'butter_order': 4
            },
        }

    def detect_yasa_three_events(self, raw, hyp, eeg_name, loc_name=None, roc_name=None):
        """
        在连续整夜信号上做 YASA 三类事件检测:
        1) spindle
        2) slow wave
        3) REM

        返回三个 DataFrame。
        如果 hyp 中不存在目标阶段，则返回空 DataFrame，不报错。
        """
        import yasa
        import numpy as np
        import pandas as pd

        sf = raw.info["sfreq"]

        # 用自适应方式拿到真正的 uV 信号
        eeg, _ = choose_signal_in_uv_from_raw(raw, eeg_name, kind="eeg")

        if hasattr(hyp, "hypno"):
            hyp_arr = np.asarray(hyp.hypno).astype(str)
        else:
            hyp_arr = np.asarray(hyp).astype(str)

        present_stages = set(hyp_arr.tolist())

        # =========================
        # Spindles: 改成 N1/N2/N3
        # =========================
        sp_df = pd.DataFrame()
        if ("N1" in present_stages) or ("N2" in present_stages) or ("N3" in present_stages):
            with suppress_stdout_stderr():
                sp = yasa.spindles_detect(
                    data=eeg,
                    sf=sf,
                    hypno=hyp,
                    include=["N1", "N2", "N3"],
                    ch_names=[eeg_name],
                    verbose=False
                )
            sp_df = sp.summary() if sp is not None else pd.DataFrame()
        else:
            print("Warning: hyp 中没有 N1/N2/N3，Spindle 检测跳过。")

        # =========================
        # Slow waves: 仍然只在 N2/N3
        # =========================
        sw_df = pd.DataFrame()
        if ("N2" in present_stages) or ("N3" in present_stages):
            with suppress_stdout_stderr():
                sw = yasa.sw_detect(
                    data=eeg,
                    sf=sf,
                    hypno=hyp,
                    include=["N2", "N3"],
                    ch_names=[eeg_name],
                    coupling=False,
                    verbose=False
                )
            sw_df = sw.summary() if sw is not None else pd.DataFrame()
        else:
            print("Warning: hyp 中没有 N2/N3，Slow wave 检测跳过。")

        # =========================
        # REM: EOG 也改成单位自适应
        # =========================
        rem_df = pd.DataFrame()
        if loc_name is not None and roc_name is not None:
            if "REM" in present_stages:
                loc, _ = choose_signal_in_uv_from_raw(raw, loc_name, kind="eog")
                roc, _ = choose_signal_in_uv_from_raw(raw, roc_name, kind="eog")

                with suppress_stdout_stderr():
                    rem = yasa.rem_detect(
                        loc=loc,
                        roc=roc,
                        sf=sf,
                        hypno=hyp,
                        include="REM",
                        verbose=False
                    )

                rem_df = rem.summary() if rem is not None else pd.DataFrame()
            else:
                print("Warning: hyp 中没有 REM，REM 检测跳过。")
        else:
            print("Warning: 缺少 LOC/ROC，REM 检测已跳过。")

        return sp_df, sw_df, rem_df

    def detect_alpha_events_continuous(self, raw, hyp, eeg_name):
        eeg, _ = choose_signal_in_uv_from_raw(raw, eeg_name, kind="eeg")
        sf = raw.info["sfreq"]

        hyp_series = hyp.hypno if hasattr(hyp, "hypno") else pd.Series(hyp)
        hyp_series = pd.Series(hyp_series).astype(str).str.upper()

        epoch_len = int(sf * 30)
        n = min(len(hyp_series), len(eeg) // epoch_len)

        stage_mask = np.zeros(len(eeg), dtype=bool)
        for i in range(n):
            if hyp_series.iloc[i] in ["W", "WAKE", "N1"]:
                s = i * epoch_len
                e = s + epoch_len
                stage_mask[s:e] = True

        count, events = self.detect_alpha_waves(eeg, stage_mask=stage_mask)

        rows = []
        for evt in events:
            rows.append({
                "Start": evt["start"],
                "End": evt["end"],
                "Duration": evt["duration"],
                "StageHint": "W/N1",
                "MeanAmp": evt.get("mean_amp", 0.0),
                "MaxAmp": evt.get("max_amp", 0.0)
            })
        return pd.DataFrame(rows)

    def detect_theta_events_continuous(self, raw, hyp, eeg_name):
        eeg, _ = choose_signal_in_uv_from_raw(raw, eeg_name, kind="eeg")
        sf = raw.info["sfreq"]

        hyp_series = hyp.hypno if hasattr(hyp, "hypno") else pd.Series(hyp)
        hyp_series = pd.Series(hyp_series).astype(str).str.upper()

        epoch_len = int(sf * 30)
        n = min(len(hyp_series), len(eeg) // epoch_len)

        stage_mask = np.zeros(len(eeg), dtype=bool)
        for i in range(n):
            if hyp_series.iloc[i] == "N1":
                s = i * epoch_len
                e = s + epoch_len
                stage_mask[s:e] = True

        count, events = self.detect_theta_waves(eeg, stage_mask=stage_mask)

        rows = []
        for evt in events:
            rows.append({
                "Start": evt["start"],
                "End": evt["end"],
                "Duration": evt["duration"],
                "StageHint": "N1",
                "MeanAmp": evt.get("mean_amp", 0.0),
                "MaxAmp": evt.get("max_amp", 0.0)
            })
        return pd.DataFrame(rows)

    def detect_sem_events(self, loc_signal, roc_signal, stage_mask=None):
        """
        仅在有双EOG时检测 N1 的慢眼动 (SEM)。
        如果缺少任一EOG通道，直接返回 0 个事件。
        """
        params = self.waveform_params['sem']

        if loc_signal is None or roc_signal is None:
            return 0, []

        if stage_mask is None:
            stage_mask = np.ones(len(loc_signal), dtype=bool)

        # 1) 低频滤波：慢眼动
        loc_f = self._butter_bandpass_filter(
            loc_signal, params['freq_band'][0], params['freq_band'][1],
            order=params['butter_order']
        )
        roc_f = self._butter_bandpass_filter(
            roc_signal, params['freq_band'][0], params['freq_band'][1],
            order=params['butter_order']
        )

        # 2) SEM 通常表现为左右眼反相缓慢运动
        #    用 LOC - ROC 增强反相成分
        diff_sig = loc_f - roc_f
        env = np.abs(hilbert(diff_sig))

        valid_env = env[stage_mask]
        if len(valid_env) == 0:
            return 0, []

        thr = np.mean(valid_env) + params['amp_threshold'] * np.std(valid_env)
        active = (env > thr) & stage_mask

        labeled, _ = label(active)
        objects = find_objects(labeled)

        events = []
        for sl in objects:
            if sl is None:
                continue

            start = sl[0].start
            end = sl[0].stop
            dur = (end - start) / self.fs

            if dur < params['min_duration'] or dur > params['max_duration']:
                continue

            loc_seg = loc_f[start:end]
            roc_seg = roc_f[start:end]
            diff_seg = diff_sig[start:end]

            # 3) 反相约束：左右眼相关系数应偏负
            corr = 0.0
            if len(loc_seg) > 5 and np.std(loc_seg) > 1e-6 and np.std(roc_seg) > 1e-6:
                corr = float(np.corrcoef(loc_seg, roc_seg)[0, 1])

            # SEM 更接近缓慢反相眼动，因此要求相关性偏负
            if corr > -0.2:
                continue

            events.append({
                'start': start / self.fs,
                'end': end / self.fs,
                'duration': dur,
                'max_amp': float(np.max(np.abs(diff_seg))),
                'mean_amp': float(np.mean(np.abs(diff_seg))),
                'corr': corr
            })

        return len(events), events

    def detect_sem_events_continuous(self, raw, hyp, loc_name, roc_name):
        """
        仅双EOG数据集使用。
        没有双EOG时直接返回空表。
        """
        if loc_name is None or roc_name is None:
            return pd.DataFrame()

        loc, _ = choose_signal_in_uv_from_raw(raw, loc_name, kind="eog")
        roc, _ = choose_signal_in_uv_from_raw(raw, roc_name, kind="eog")
        sf = raw.info["sfreq"]

        hyp_series = hyp.hypno if hasattr(hyp, "hypno") else pd.Series(hyp)
        hyp_series = pd.Series(hyp_series).astype(str).str.upper()

        epoch_len = int(sf * 30)
        n = min(len(hyp_series), len(loc) // epoch_len, len(roc) // epoch_len)

        stage_mask = np.zeros(len(loc), dtype=bool)
        for i in range(n):
            if hyp_series.iloc[i] == "N1":
                s = i * epoch_len
                e = s + epoch_len
                stage_mask[s:e] = True

        count, events = self.detect_sem_events(loc, roc, stage_mask)

        rows = []
        for evt in events:
            rows.append({
                "Start": evt["start"],
                "End": evt["end"],
                "Duration": evt["duration"],
                "StageHint": "N1",
                "MeanAmp": evt.get("mean_amp", 0.0),
                "MaxAmp": evt.get("max_amp", 0.0),
                "Corr": evt.get("corr", 0.0)
            })

        return pd.DataFrame(rows)
    def compute_band_features(self, eeg_signal):
        def band_power(sig, low, high, order=4):
            filtered = self._butter_bandpass_filter(sig, low, high, order=order)
            return float(np.mean(filtered ** 2))

        total_power = band_power(eeg_signal, 0.5, 30.0)
        alpha_power = band_power(eeg_signal, 8.0, 13.0)
        theta_power = band_power(eeg_signal, 3.0, 7.0)

        return {
            'alpha_power': alpha_power,
            'theta_power': theta_power,
            'alpha_ratio': alpha_power / (total_power + 1e-6),
            'theta_ratio': theta_power / (total_power + 1e-6),
            'alpha_theta_ratio': alpha_power / (theta_power + 1e-6)
        }

    def detect_theta_waves(self, eeg_signal, stage_mask=None):
        params = self.waveform_params['theta']
        low, high = params['freq_band']

        if stage_mask is None:
            stage_mask = np.ones(len(eeg_signal), dtype=bool)

        filtered = self._butter_bandpass_filter(
            eeg_signal, low, high, order=params['butter_order']
        )

        envelope = np.abs(hilbert(filtered))
        valid_env = envelope[stage_mask]

        if len(valid_env) == 0:
            return 0, []

        thr = np.mean(valid_env) + params['amp_threshold'] * np.std(valid_env)
        active = (envelope > thr) & stage_mask

        labeled, _ = label(active)
        objects = find_objects(labeled)

        events = []
        for sl in objects:
            if sl is None:
                continue

            start = sl[0].start
            end = sl[0].stop
            dur = (end - start) / self.fs

            if dur < params['min_duration']:
                continue

            seg = filtered[start:end]
            events.append({
                'start': start / self.fs,
                'end': end / self.fs,
                'duration': dur,
                'max_amp': float(np.max(np.abs(seg))),
                'mean_amp': float(np.mean(np.abs(seg)))
            })

        return len(events), events

    # ---------- 工具函数 ----------
    def _butter_bandpass_filter(self, data, lowcut, highcut, order=4):
        """带通滤波器（零相位）"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
        return filtered

    def _calculate_slope(self, signal):
        """对低幅值信号先轻度去噪再计算斜率"""
        signal_denoise = np.convolve(signal, [1 / 3, 1 / 3, 1 / 3], mode='same')
        dx = 1 / self.fs
        slope = np.diff(signal_denoise) / dx
        slope = np.append(slope, slope[-1])
        return slope

    def _linear_weighted_moving_avg(self, data, window_size):
        """对低幅值信号使用更轻的加权"""
        if window_size <= 1:
            return data.copy()

        weights = np.linspace(1, window_size / 2, window_size)
        weights = weights / weights.sum()

        data_padded = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
        smoothed = np.convolve(data_padded, weights, mode='same')
        return smoothed[window_size // 2:-window_size // 2] if window_size // 2 > 0 else smoothed

    # ---------- 核心检测方法 (基于您的文档1: alpha_wave_detect.py) ----------
    def detect_alpha_waves(self, eeg_signal, stage_mask=None):
        """
        检测α波 (8-13Hz节律)。
        基于文档1中的 `detect_alpha_wave` 函数逻辑。
        Args:
            eeg_signal: 单通道EEG信号，长度为 N 的数组。
            stage_mask: 与信号等长的布尔数组，True表示目标分期(W/N1)。如果为None，则在全部信号中检测。
        Returns:
            tuple: (事件数量, 事件列表)
                   事件列表中的每个元素为字典，包含'start','end','duration','max_amp','mean_amp'等。
        """
        params = self.waveform_params['alpha']
        low, high = params['freq_band']
        min_dur = params['min_duration']
        max_dur = params['max_duration']
        amp_thr = params['amp_threshold']

        if stage_mask is None:
            stage_mask = np.ones(len(eeg_signal), dtype=bool)

        # 1. 带通滤波提取α节律
        alpha_signal = self._butter_bandpass_filter(eeg_signal, low, high, params['butter_order'])
        alpha_amp = np.abs(hilbert(alpha_signal))  # 包络

        # 2. 仅对目标分期信号计算阈值
        target_amp = alpha_amp[stage_mask]
        if len(target_amp) == 0:
            return 0, []
        threshold = np.mean(target_amp) + amp_thr * np.std(target_amp)

        # 3. 事件的整个区间都必须位于目标分期。
        alpha_mask = (alpha_amp > threshold) & stage_mask

        # 4. 标记连续的超阈值段
        events = []
        labeled, n_events = label(alpha_mask)
        objects = find_objects(labeled)

        for i in range(1, n_events + 1):
            if i - 1 >= len(objects):
                continue
            event_slice = objects[i - 1][0]
            start_idx = event_slice.start
            end_idx = event_slice.stop

            event_dur = (end_idx - start_idx) / self.fs
            if event_dur < min_dur or event_dur > max_dur:
                continue

            # 计算事件特征
            max_amp = np.max(alpha_amp[start_idx:end_idx])
            mean_amp = np.mean(alpha_amp[start_idx:end_idx])

            events.append({
                'start': start_idx / self.fs,
                'end': end_idx / self.fs,
                'duration': event_dur,
                'max_amp': max_amp,
                'mean_amp': mean_amp
            })
        return len(events), events

    def detect_k_complexes(self, eeg_signal, stage_mask=None):
        """
        检测K-复合波 (高幅值负向尖峰+后续正向波)。
        基于文档1中的 `detect_k_complex` 函数逻辑。
        """
        params = self.waveform_params['k_complex']
        low, high = params['freq_band']
        eeg_signal = np.asarray(eeg_signal, dtype=float)

        if stage_mask is None:
            stage_mask = np.ones(len(eeg_signal), dtype=bool)
        else:
            stage_mask = np.asarray(stage_mask, dtype=bool)
            if len(stage_mask) != len(eeg_signal):
                raise ValueError("stage_mask and eeg_signal must have the same length")
        if not np.any(stage_mask):
            return 0, []

        # 1. 带通滤波提取慢波成分
        slow_signal = self._butter_bandpass_filter(eeg_signal, low, high, params['butter_order'])

        peak_distance = max(1, int(params['peak_distance'] * self.fs))
        neg_peak_idx, _ = signal.find_peaks(
            -slow_signal,
            height=params['min_negative_amp'],
            prominence=params['peak_prominence'],
            distance=peak_distance,
        )

        zero_crossings = np.flatnonzero(np.diff(np.signbit(slow_signal)))
        delay_min = max(1, int(params['positive_delay_min'] * self.fs))
        delay_max = max(delay_min + 1, int(params['positive_delay_max'] * self.fs))
        events = []
        for neg_idx in neg_peak_idx:
            if not stage_mask[neg_idx]:
                continue
            search_start = neg_idx + delay_min
            search_end = min(len(slow_signal), neg_idx + delay_max + 1)
            if search_start >= search_end:
                continue

            positive_segment = slow_signal[search_start:search_end]
            pos_idx = search_start + int(np.argmax(positive_segment))
            negative_amp = float(-slow_signal[neg_idx])
            positive_amp = float(slow_signal[pos_idx])
            peak_to_peak = negative_amp + positive_amp
            if not stage_mask[pos_idx]:
                continue
            if positive_amp < params['min_positive_amp']:
                continue
            if peak_to_peak < params['min_peak_to_peak_amp']:
                continue

            previous_crossings = zero_crossings[zero_crossings < neg_idx]
            following_crossings = zero_crossings[zero_crossings >= pos_idx]
            if previous_crossings.size == 0 or following_crossings.size == 0:
                continue
            event_start = int(previous_crossings[-1] + 1)
            event_end = int(following_crossings[0] + 1)
            duration = (event_end - event_start) / self.fs
            if not (params['min_duration'] <= duration <= params['max_duration']):
                continue
            if not np.all(stage_mask[event_start:event_end]):
                continue

            events.append({
                'start': event_start / self.fs,
                'negative_peak': neg_idx / self.fs,
                'positive_peak': pos_idx / self.fs,
                'end': event_end / self.fs,
                'duration': duration,
                'peak_amp': negative_amp,
                'positive_amp': positive_amp,
                'peak_to_peak_amp': peak_to_peak,
            })
        return len(events), events

    def detect_k_complex_events_continuous(self, raw, hyp, eeg_name):
        eeg, _ = choose_signal_in_uv_from_raw(raw, eeg_name, kind="eeg")
        sf = raw.info["sfreq"]
        hyp_series = hyp.hypno if hasattr(hyp, "hypno") else pd.Series(hyp)
        hyp_series = pd.Series(hyp_series).astype(str).str.upper()
        epoch_len = int(sf * 30)
        n_epochs = min(len(hyp_series), len(eeg) // epoch_len)

        stage_mask = np.zeros(len(eeg), dtype=bool)
        for epoch_idx in range(n_epochs):
            if hyp_series.iloc[epoch_idx] == "N2":
                start = epoch_idx * epoch_len
                stage_mask[start:start + epoch_len] = True

        _, events = self.detect_k_complexes(eeg, stage_mask=stage_mask)
        return pd.DataFrame([
            {
                "Start": event["start"],
                "End": event["end"],
                "Duration": event["duration"],
                "StageHint": "N2",
                "NegativeAmp": event["peak_amp"],
                "PositiveAmp": event["positive_amp"],
                "PeakToPeakAmp": event["peak_to_peak_amp"],
            }
            for event in events
        ])

    # ---------- 基于YASA的检测 (来自您的文档3: spindle_slow_wave_detect.py) ----------
    def detect_spindles_yasa(self, eeg_signal,stage_mask=None,sleep_stage=None):
        """
        使用YASA库检测纺锤波。
        注意：此方法需要单独安装 yasa 库 (pip install yasa)。
        此处返回YASA的检测结果对象，您可以在主流程中提取所需特征。
        """
        try:
            target_stages = self.waveform_params['spindle']['target_stages']
            # 如果提供了sleep_stage，但当前分期不在目标分期内，则直接返回无纺锤波
            if sleep_stage is not None and sleep_stage not in target_stages:
                # 当前epoch并非N1、N2、N3期，按定义不应有纺锤波，直接返回
                return 0, []

            import yasa
            sleep_stage_array = np.full(len(eeg_signal), sleep_stage, dtype=int)
            # # 应用分期掩码：将非目标分期的信号置零
            # signal_to_analyze = eeg_signal.copy()
            # if stage_mask is not None:
            #     # 将非目标分期（stage_mask为False）的信号值设为0
            #     signal_to_analyze[~stage_mask] = 0
            # print(f"通道是：{self.channel_names}")
            # print(f"eeg信号：{eeg_signal.shape}")
            # print(f"标签：{sleep_stage_array.shape}")
            # 假设传入的是单通道信号
            with suppress_stdout_stderr():
                sp = yasa.spindles_detect(
                    data=eeg_signal,
                    sf=self.fs,
                    hypno=sleep_stage_array,
                    include=(1, 2, 3),
                    ch_names=[self.channel_names],
                    freq_sp=self.waveform_params['spindle']['freq_band'],
                    duration=(
                        self.waveform_params['spindle']['min_duration'],
                        self.waveform_params['spindle']['max_duration']
                    ),
                    verbose=False,
                )
            if sp is not None:
                summary = sp.summary()
                return len(summary), summary.to_dict('records')
            return 0, []
        except ImportError:
            print("警告：未安装YASA库，纺锤波检测不可用。请运行 `pip install yasa`")
            return 0, []
        except Exception as e:
            print(f"纺锤波检测出错: {e}")
            return 0, []

    def detect_slow_waves_yasa(self, eeg_signal,stage_mask=None,sleep_stage=None):
        """
        使用YASA库检测慢波。
        """
        try:
            target_stages = self.waveform_params['slow_wave']['target_stages']
            # 如果提供了sleep_stage，但当前分期不在目标分期内，则直接返回无纺锤波
            if sleep_stage is not None and sleep_stage not in target_stages:
                # 当前epoch并非N1、N2、N3期，按定义不应有纺锤波，直接返回
                return 0, []

            import yasa
            sleep_stage_array = np.full(len(eeg_signal), sleep_stage, dtype=int)
            # # 应用分期掩码：将非目标分期的信号置零
            # signal_to_analyze = eeg_signal.copy()
            # if stage_mask is not None:
            #     signal_to_analyze[~stage_mask] = 0

            with suppress_stdout_stderr():
                sw = yasa.sw_detect(
                    data=eeg_signal,
                    sf=self.fs,
                    hypno=sleep_stage_array,
                    ch_names=[self.channel_names],
                    freq_sw=self.waveform_params['slow_wave']['freq_band'],
                    dur_neg=[0.3, 1.5],
                    dur_pos=(0.1, 1),
                    include=(2, 3),
                    amp_neg=(40, 200),
                    amp_pos=(10, 150),
                    amp_ptp=(75, 350),
                    verbose=False,
                )
            if sw is not None:
                summary = sw.summary()
                return len(summary), summary.to_dict('records')
            return 0, []
        except ImportError:
            print("警告：未安装YASA库，慢波检测不可用。")
            return 0, []
        except Exception as e:
            print(f"慢波检测出错: {e}")
            return 0, []

    # ---------- 顶尖波检测 (基于您的文档4: vertex_sharp_detect.py) ----------
    def detect_vertex_waves(self, eeg_signal, stage_mask=None):
        """
        检测顶尖波 (Vertex Sharp Waves)，主要出现在N1期。
        基于文档4中的核心逻辑。
        """
        params = self.waveform_params['vertex']
        min_dur = params['min_duration']
        max_dur = params['max_duration']
        slope_thr = params['slope_threshold']
        eeg_signal = np.asarray(eeg_signal, dtype=float)

        if stage_mask is None:
            stage_mask = np.ones(len(eeg_signal), dtype=bool)
        else:
            stage_mask = np.asarray(stage_mask, dtype=bool)
            if len(stage_mask) != len(eeg_signal):
                raise ValueError("stage_mask and eeg_signal must have the same length")
        if not np.any(stage_mask):
            return 0, []

        low, high = params['freq_band']
        filtered = self._butter_bandpass_filter(
            eeg_signal, low, high, params['butter_order']
        )
        target_values = filtered[stage_mask]
        dynamic_amp = params['threshold_std'] * robust_std(target_values)
        min_amp = max(float(params['neg_threshold_uv']), float(dynamic_amp))
        peaks, _ = signal.find_peaks(
            -filtered,
            height=min_amp,
            prominence=params['peak_prominence'],
            distance=max(1, int(params['peak_distance'] * self.fs)),
        )

        zero_crossings = np.flatnonzero(np.diff(np.signbit(filtered)))
        events = []
        for pk_idx in peaks:
            if not stage_mask[pk_idx]:
                continue
            previous_crossings = zero_crossings[zero_crossings < pk_idx]
            following_crossings = zero_crossings[zero_crossings >= pk_idx]
            if previous_crossings.size == 0 or following_crossings.size == 0:
                continue
            start_idx = int(previous_crossings[-1] + 1)
            end_idx = int(following_crossings[0] + 1)
            duration = (end_idx - start_idx) / self.fs
            if not (min_dur <= duration <= max_dur):
                continue
            if not np.all(stage_mask[start_idx:end_idx]):
                continue

            window_data = filtered[start_idx:end_idx]
            if len(window_data) < 2:
                continue
            max_slope = np.max(np.abs(np.diff(window_data))) * self.fs
            if max_slope < slope_thr:
                continue
            peak_amp = float(-filtered[pk_idx])

            events.append({
                'start': start_idx / self.fs,
                'peak': pk_idx / self.fs,
                'end': end_idx / self.fs,
                'duration': duration,
                'amplitude': peak_amp,
                'max_slope': float(max_slope),
                'threshold_uv': float(min_amp),
            })
        return len(events), events

    # ---------- 快速眼动检测 (基于您的文档2: rapid_eyes_movement.py) ----------
    def detect_rem_events(self, eog_loc, eog_roc, stage_mask=None):
        """
        检测快速眼动（双通道EOG：LOC-A2 + ROC-A1）
        借鉴 rapid_eyes_movement双眼.py 的思路：
        1. 双通道带通滤波
        2. 基线校正
        3. 构造左右眼差分信号
        4. 幅度 + 斜率筛选候选点
        5. onset 精修
        6. 左右眼共轭性验证
        7. 输出 REM 事件
        """
        params = self.waveform_params['rem']

        # 输入保护
        if eog_loc is None or eog_roc is None:
            return 0, []

        eog_loc = np.asarray(eog_loc, dtype=float)
        eog_roc = np.asarray(eog_roc, dtype=float)

        if len(eog_loc) == 0 or len(eog_roc) == 0:
            return 0, []

        # 长度对齐
        n = min(len(eog_loc), len(eog_roc))
        eog_loc = eog_loc[:n]
        eog_roc = eog_roc[:n]

        if stage_mask is not None:
            stage_mask = np.asarray(stage_mask).astype(bool)[:n]
            if len(stage_mask) < n:
                padded_mask = np.zeros(n, dtype=bool)
                padded_mask[:len(stage_mask)] = stage_mask
                stage_mask = padded_mask

        # 1. 双通道滤波
        loc_filt = self._butter_bandpass_filter(
            eog_loc,
            params['filter_band'][0],
            params['filter_band'][1],
            order=3
        )
        roc_filt = self._butter_bandpass_filter(
            eog_roc,
            params['filter_band'][0],
            params['filter_band'][1],
            order=3
        )

        # 轻微增强
        loc_enhanced = loc_filt * 1.2
        roc_enhanced = roc_filt * 1.2

        # 2. 基线校正
        if stage_mask is not None and np.any(stage_mask):
            loc_baseline = loc_enhanced - np.mean(loc_enhanced[stage_mask])
            roc_baseline = roc_enhanced - np.mean(roc_enhanced[stage_mask])
        else:
            loc_baseline = loc_enhanced - np.mean(loc_enhanced)
            roc_baseline = roc_enhanced - np.mean(roc_enhanced)

        # 3. 构造左右眼差分信号
        eog_diff = loc_baseline - roc_baseline

        loc_smoothed = self._linear_weighted_moving_avg(
            loc_baseline, params['moving_avg_window']
        )
        roc_smoothed = self._linear_weighted_moving_avg(
            roc_baseline, params['moving_avg_window']
        )
        eog_diff_smoothed = self._linear_weighted_moving_avg(
            eog_diff, params['moving_avg_window']
        )

        if len(eog_diff_smoothed) < 5:
            return 0, []

        # 4. 计算斜率、一阶导、二阶导
        slope = self._calculate_slope(eog_diff_smoothed)

        first_deriv = np.diff(eog_diff_smoothed) / (1 / self.fs)
        if len(first_deriv) == 0:
            return 0, []
        first_deriv = np.append(first_deriv, first_deriv[-1])

        second_deriv = np.diff(first_deriv) / (1 / self.fs)
        if len(second_deriv) == 0:
            return 0, []
        second_deriv = np.append(second_deriv, second_deriv[-1])

        # 5. 候选点筛选
        candidate_indices = []
        amp_win = max(1, int(params['amplitude_time_win'] * self.fs))
        avg_slope_win = max(1, int(params['avg_slope_time_win'] * self.fs))

        for t_idx in range(amp_win, len(eog_diff_smoothed) - avg_slope_win):
            if stage_mask is not None and not stage_mask[t_idx]:
                continue

            local_seg = eog_diff_smoothed[t_idx - amp_win:t_idx + 1]
            amp_change = np.max(local_seg) - np.min(local_seg)
            if amp_change < params['amplitude_thr']:
                continue

            if -params['instant_slope_thr'] < slope[t_idx] < params['instant_slope_thr']:
                continue

            avg_slope = np.mean(np.abs(slope[t_idx:t_idx + avg_slope_win]))
            if avg_slope < params['avg_slope_thr']:
                continue

            candidate_indices.append(t_idx)

        if len(candidate_indices) == 0:
            return 0, []

        # 6. 根据 em_gap 合并邻近候选点，每组保留二阶导最大的点
        filtered_candidates = []
        candidate_indices = np.array(candidate_indices, dtype=int)
        candidate_times = candidate_indices / self.fs
        groups = np.round(candidate_times / params['em_gap'])

        for g in np.unique(groups):
            group_idx = candidate_indices[groups == g]
            if len(group_idx) == 0:
                continue

            best_idx = group_idx[np.argmax(second_deriv[group_idx])]
            filtered_candidates.append(int(best_idx))

        if len(filtered_candidates) == 0:
            return 0, []

        # 7. onset 精修：在附近窗口里找二阶导绝对值最大的点
        refined_candidates = []
        onset_search_samples = max(1, int(params['onset_max_move'] * self.fs))

        for idx in filtered_candidates:
            search_start = max(0, idx - onset_search_samples)
            search_end = min(len(second_deriv), idx + onset_search_samples + 1)
            if search_end <= search_start:
                continue

            local_idx = search_start + np.argmax(
                np.abs(second_deriv[search_start:search_end])
            )
            refined_candidates.append(int(local_idx))

        if len(refined_candidates) == 0:
            return 0, []

        # 8. 左右眼共轭性验证
        conjugate_validated = []
        conjugate_pre = max(1, int(params['conjugate_pre_win'] * self.fs))
        conjugate_post = max(1, int(params['conjugate_post_win'] * self.fs))

        for idx in refined_candidates:
            win_start = max(0, idx - conjugate_pre)
            win_end = min(len(loc_smoothed), idx + conjugate_post)
            if win_end - win_start < 5:
                continue

            loc_win = loc_smoothed[win_start:win_end]
            roc_win = roc_smoothed[win_start:win_end]

            duration_sec = (win_end - win_start) / self.fs
            if duration_sec <= 0:
                continue

            slope_loc = (loc_win[-1] - loc_win[0]) / duration_sec
            slope_roc = (roc_win[-1] - roc_win[0]) / duration_sec

            slope_product = slope_loc * slope_roc
            slope_abs_ratio = max(abs(slope_loc), abs(slope_roc)) / (min(abs(slope_loc), abs(slope_roc)) + 1e-6)

            # 满足其一：
            # 1) 左右眼方向相反（斜率乘积足够负）
            # 2) 左右眼变化幅度差异明显（比例足够大）
            if (
                    (abs(slope_loc) > params['conjugate_slope_thr'] or abs(slope_roc) > params['conjugate_slope_thr'])
                    and
                    (slope_product < params['slope_product_thr'] or slope_abs_ratio > 1.5)
            ):
                conjugate_validated.append(int(idx))

        if len(conjugate_validated) == 0:
            return 0, []

        # 9. 最终唯一化，避免重复事件
        unique_events = []
        conjugate_validated = sorted(set(conjugate_validated))
        validated_times = np.array(conjugate_validated) / self.fs

        if len(validated_times) > 0:
            unique_events.append(int(conjugate_validated[0]))
            last_time = validated_times[0]

            for i in range(1, len(validated_times)):
                if validated_times[i] - last_time > 0.1:  # 合并 0.1 秒内重复事件
                    unique_events.append(int(conjugate_validated[i]))
                    last_time = validated_times[i]

        # 10. 生成事件
        events = []
        for idx in unique_events:
            onset = idx / self.fs
            end = onset + params['event_duration']

            if end > len(eog_diff_smoothed) / self.fs:
                continue

            events.append({
                'onset': onset,
                'end': end,
                'duration': params['event_duration'],
                'peak_time': onset,
                'peak_amp': float(np.abs(eog_diff_smoothed[idx])),
                'eog_loc_idx': (int(onset * self.fs), int(end * self.fs)),
                'eog_roc_idx': (int(onset * self.fs), int(end * self.fs))
            })

        return len(events), events

    # ---------- 统一入口函数 ----------
    def detect_all_in_epoch(self, eeg_epoch, sleep_stage=None, eog_loc_epoch=None, eog_roc_epoch=None):
        """
        对单个30秒EEG epoch执行所有适用的波形检测。
        这是您替代特征提取和聚类的主要调用函数。
        Args:
            eeg_epoch: 单通道EEG信号 (30秒数据，长度应为 fs*30 )。
            sleep_stage: 当前epoch的睡眠分期 (0=W,1=N1,2=N2,3=N3,4=REM)。用于创建分期掩码，提高检测效率。
            eog_loc_epoch: (可选) 当前epoch的LOC-A2眼电信号，用于REM检测。
            eog_roc_epoch: (可选) 当前epoch的ROC-A1眼电信号，用于REM检测。
        Returns:
            dict: 包含所有波形检测结果的特征字典，可直接用于生成文本描述。
                  格式: {'has_alpha':1, 'alpha_count':2, 'alpha_density':4.0, ...}
        """
        if len(eeg_epoch) != self.fs * 30:
            print(f"警告：epoch长度{len(eeg_epoch)}不等于30秒的数据点({self.fs * 30})")

        # 根据睡眠分期创建各波形的目标分期掩码
        stage_masks = {}
        if sleep_stage is not None:
            # 创建与eeg_epoch等长的全False数组
            full_mask = np.zeros(len(eeg_epoch), dtype=bool)
            # 对于目标分期，将整个epoch设为True
            stage_masks['alpha'] = full_mask.copy()
            if sleep_stage in self.waveform_params['alpha']['target_stages']:
                stage_masks['alpha'][:] = True

            stage_masks['k_complex'] = full_mask.copy()
            if sleep_stage in self.waveform_params['k_complex']['target_stages']:
                stage_masks['k_complex'][:] = True

            stage_masks['vertex'] = full_mask.copy()
            if sleep_stage in self.waveform_params['vertex']['target_stages']:
                stage_masks['vertex'][:] = True

            stage_masks['theta'] = full_mask.copy()
            if sleep_stage in self.waveform_params['theta']['target_stages']:
                stage_masks['theta'][:] = True

            stage_masks['sem'] = full_mask.copy()
            if sleep_stage in self.waveform_params['sem']['target_stages']:
                stage_masks['sem'][:] = True

            # === 新增：为纺锤波和慢波创建掩码 ===
            stage_masks['spindle'] = full_mask.copy()
            if sleep_stage in self.waveform_params['spindle']['target_stages']:
                stage_masks['spindle'][:] = True

            stage_masks['slow_wave'] = full_mask.copy()
            if sleep_stage in self.waveform_params['slow_wave']['target_stages']:
                stage_masks['slow_wave'][:] = True

            stage_masks['rem'] = full_mask.copy()
            if sleep_stage == 4:  # REM期
                stage_masks['rem'][:] = True
        else:
            # 如果未提供分期，则在全部信号中检测
            for key in ['alpha', 'theta', 'k_complex', 'vertex', 'sem', 'rem', 'spindle', 'slow_wave']:
                stage_masks[key] = None

        # 执行检测
        results = {}

        # 1. α波
        alpha_count, alpha_events = self.detect_alpha_waves(eeg_epoch, stage_masks['alpha'])
        results['alpha'] = (alpha_count, alpha_events)

        # 2. K-复合波
        kc_count, kc_events = self.detect_k_complexes(eeg_epoch, stage_masks['k_complex'])
        results['k_complex'] = (kc_count, kc_events)

        # 3. 纺锤波 (使用YASA)
        # print(f"[DEBUG] 输入YASA的eeg_epoch形状: {eeg_epoch.shape}, 类型: {eeg_epoch.dtype}")
        # print(f"[DEBUG] 输入YASA的eeg_epoch统计: mean={np.mean(eeg_epoch):.2f}, std={np.std(eeg_epoch):.2f}")
        spindle_count, spindle_events = self.detect_spindles_yasa(eeg_epoch, stage_masks['spindle'],sleep_stage)
        results['spindle'] = (spindle_count, spindle_events)

        # 4. 慢波 (使用YASA)
        sw_count, sw_events = self.detect_slow_waves_yasa(eeg_epoch,stage_masks['slow_wave'],sleep_stage)
        results['slow_wave'] = (sw_count, sw_events)

        # 5. 顶尖波
        vertex_count, vertex_events = self.detect_vertex_waves(eeg_epoch, stage_masks['vertex'])
        results['vertex'] = (vertex_count, vertex_events)

        # 缓慢眼动 SEM (需要双EOG，仅MASS主实验启用)
        sem_count, sem_events = (0, [])
        if eog_loc_epoch is not None and eog_roc_epoch is not None:
            sem_count, sem_events = self.detect_sem_events(
                eog_loc_epoch,
                eog_roc_epoch,
                stage_masks['sem']
            )
        results['sem'] = (sem_count, sem_events)

        # 6. 快速眼动 (需要EOG)
        rem_count, rem_events = (0, [])
        if eog_loc_epoch is not None and eog_roc_epoch is not None:
            rem_count, rem_events = self.detect_rem_events(
                eog_loc_epoch,
                eog_roc_epoch,
                stage_masks['rem']
            )
        results['rem'] = (rem_count, rem_events)

        theta_count, theta_events = self.detect_theta_waves(eeg_epoch, stage_masks['theta'])
        results['theta'] = (theta_count, theta_events)

        # 转换为统一的特征字典 (用于生成文本描述)
        features = self._results_to_features(results, epoch_duration_sec=30.0)
        # {
        #     # K复合波特征
        #     'has_kcomplex': 1,
        #     'kcomplex_count': 1,
        #     'kcomplex_density': 2.0,
        #     'mean_kcomplex_amp': 18.2,
        #     'mean_kcomplex_dur': 0.6,
        #
        #     # 纺锤波特征
        #     'has_spindle': 1,
        #     'spindle_count': 2,
        #     'spindle_density': 4.0,
        #     'mean_spindle_amp': 13.8,
        #     'mean_spindle_dur': 0.85
        # }
        return features

    def _results_to_features(self, detection_results, epoch_duration_sec=30.0):
        """
        将检测结果转换为统一的特征字典。
        """
        features = {}
        epoch_duration_min = epoch_duration_sec / 60.0

        wave_name_map = {
            'alpha': 'alpha',
            'theta': 'theta',
            'k_complex': 'kcomplex',
            'spindle': 'spindle',
            'slow_wave': 'slowwave',
            'vertex': 'vertex',
            'sem': 'sem',
            'rem': 'rem'
        }

        for key, (count, events) in detection_results.items():
            feat_prefix = wave_name_map.get(key, key)
            features[f'has_{feat_prefix}'] = 1 if count > 0 else 0
            features[f'{feat_prefix}_count'] = count
            features[f'{feat_prefix}_density'] = count / epoch_duration_min if epoch_duration_min > 0 else 0.0


            # 可选：添加平均幅度、持续时间等特征
            if events and len(events) > 0:
                if any(name in events[0] for name in ('amplitude', 'max_amp', 'peak_amp')):
                    amps = []
                    for evt in events:
                        if 'amplitude' in evt:
                            amps.append(evt['amplitude'])
                        elif 'max_amp' in evt:
                            amps.append(evt['max_amp'])
                        elif 'peak_amp' in evt:
                            amps.append(evt['peak_amp'])
                    if amps:
                        features[f'mean_{feat_prefix}_amp'] = np.mean(amps)

                if 'duration' in events[0]:
                    durs = [evt['duration'] for evt in events if 'duration' in evt]
                    if durs:
                        features[f'mean_{feat_prefix}_dur'] = np.mean(durs)
            else:
                features[f'mean_{feat_prefix}_amp'] = 0.0
                features[f'mean_{feat_prefix}_dur'] = 0.0


        return features


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 1. 初始化检测器
    detector = MorphologyDetector(fs=100)

    # 2. 模拟一个30秒的EEG epoch (随机数据，仅用于演示)
    epoch_duration = 30
    fs = 100
    t = np.linspace(0, epoch_duration, epoch_duration * fs, endpoint=False)
    # 生成模拟信号：包含一些振荡
    simulated_eeg = 5 * np.sin(2 * np.pi * 10 * t) + 3 * np.sin(2 * np.pi * 0.8 * t) + np.random.randn(len(t)) * 2

    # 3. 模拟一个睡眠分期 (例如 N2期)
    simulated_stage = 2  # N2

    # 4. 执行检测
    features = detector.detect_all_in_epoch(
        eeg_epoch=simulated_eeg,
        sleep_stage=simulated_stage,
        eog_loc_epoch=None,  # 模拟中没有EOG
        eog_roc_epoch=None
    )

    # 5. 打印特征
    print("检测到的形态学特征:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    # 6. 这些特征可以直接用于您之前设计的 `generate_description_from_waveforms` 函数
    # 生成文本描述示例：
    stage_map = {0: "清醒期", 1: "N1期", 2: "N2期", 3: "N3期", 4: "REM期"}
    desc_parts = []

    if simulated_stage in stage_map:
        desc_parts.append(f"当前处于{stage_map[simulated_stage]}。")

    if features.get('has_spindle', 0) and features.get('has_kcomplex', 0):
        desc_parts.append("纺锤波与K-复合波频繁出现，这是N2睡眠期的典型特征。")
    elif features.get('has_spindle', 0):
        desc_parts.append("出现明显的睡眠纺锤波。")
    elif features.get('has_kcomplex', 0):
        desc_parts.append("检测到K-复合波。")

    if features.get('spindle_density', 0) > 0:
        desc_parts.append(f"纺锤波密度为{features['spindle_density']:.1f}个/分钟。")

    full_description = " ".join(desc_parts)
    print("\n生成的文本描述:")
    print(full_description)
