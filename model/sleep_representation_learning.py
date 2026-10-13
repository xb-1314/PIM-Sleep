import os
import copy
import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# 配置
# =========================
class EEGPretrainConfig:
    def __init__(self):
        # 预处理后的 npz 目录，可按需增减
        self.pretrain_dirs = [
            r"D:/dataset/mass3/signals_edf/processed_morphology_CLAS_EOG_five/",
            r"D:/dataset/ISRUC_S1/processed_morphology_CLAS_EOG_five/"
        ]

        self.output_dir = r"./sleep_eeg_pretrained_backbone_event_detached_bert_context"

        self.seed = 42
        self.batch_size = 8
        self.num_epochs = 30
        self.learning_rate = 8e-5
        self.weight_decay = 1e-4
        self.num_workers = 0
        self.use_amp = True

        # 与下游 04 保持一致
        self.eeg_hidden_dim = 256
        self.eeg_kernel_size = 7
        self.dropout = 0.1
        self.seq_num_layers = 2
        self.seq_num_heads = 8
        self.seq_len = 20
        self.stride = 10

        # 输入长度：100 Hz * 30 s = 3000
        self.sample_length = 3000

        # 数据量控制
        self.max_subjects = None
        self.max_windows = None
        self.min_valid_epochs_per_subject = 20

        # 事件标签: [alpha, slowwave, spindle, rem, vertex, kcomplex]
        self.event_names = ["alpha", "slowwave", "spindle", "rem", "vertex", "kcomplex"]

        # 单阶段联合预训练：Full Event 为主任务，BERT Masked Event 为弱上下文辅助任务。
        # 仍然保持 30 s = 1 epoch token，20个epoch = 10 min上下文。
        self.mask_ratio = 0.10
        self.min_masked_epochs = 1
        self.mask_token_prob = 0.80
        self.keep_original_prob = 0.10
        self.random_replace_prob = 0.10
        self.lambda_full_event = 1.0
        self.lambda_masked_event = 0.05
        self.validation_mask_seed = 2026

        # 关键设计：Masked分支从epoch embedding开始，并detach，
        # 因此Masked loss只更新position/Global Transformer/[MASK]/masked head，
        # 不更新EEG encoder、EOG encoder和fusion。
        self.detach_masked_epoch_embeddings = True

        # 为了与原Event-only基线尽量公平，两个分支都使用普通BCE，不使用pos_weight。
        self.pretrain_type = "full_event_plus_detached_bert_context"

        self.save_epoch_checkpoints = True
        self.save_every_epochs = 10


# =========================
# 工具函数
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)



def list_morphology_files(processed_dir):
    files = []
    if not os.path.isdir(processed_dir):
        return files
    for fname in os.listdir(processed_dir):
        if fname.endswith("_morphology.npz"):
            files.append(os.path.join(processed_dir, fname))
    files.sort()
    return files



def standardize_epoch(x):
    x = np.asarray(x, dtype=np.float32)
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-6:
        return x - mean
    return (x - mean) / std



def fix_length(x, target_len):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:len(x)] = x
    return out



def build_window_starts(n, seq_len, stride):
    if n <= seq_len:
        return [0]
    starts = list(range(0, n - seq_len + 1, stride))
    last_start = n - seq_len
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts



def pad_eeg_epochs(eeg_epochs, target_len, sample_length):
    eeg_epochs = np.asarray(eeg_epochs, dtype=np.float32)
    if eeg_epochs.ndim != 2:
        raise ValueError(f"eeg_epochs 期望二维数组，实际形状: {eeg_epochs.shape}")
    if len(eeg_epochs) >= target_len:
        return eeg_epochs[:target_len]
    pad = np.zeros((target_len - len(eeg_epochs), sample_length), dtype=np.float32)
    return np.concatenate([eeg_epochs, pad], axis=0)


def pad_psg_epochs(psg_epochs, target_len, sample_length):
    psg_epochs = np.asarray(psg_epochs, dtype=np.float32)
    if psg_epochs.ndim != 3 or psg_epochs.shape[1] != 3:
        raise ValueError(f"psg_epochs expected [epochs, 3, samples], got {psg_epochs.shape}")
    if len(psg_epochs) >= target_len:
        return psg_epochs[:target_len]
    pad = np.zeros((target_len - len(psg_epochs), 3, sample_length), dtype=np.float32)
    return np.concatenate([psg_epochs, pad], axis=0)



def pad_valid_mask(valid_len, target_len):
    valid_len = int(min(max(valid_len, 0), target_len))
    return np.asarray([1] * valid_len + [0] * (target_len - valid_len), dtype=np.int64)



def pad_event_matrix(values, target_len, num_events, pad_value=-100):
    values = np.asarray(values, dtype=np.int64)
    if values.ndim == 1:
        values = values.reshape(-1, num_events)
    if len(values) >= target_len:
        return values[:target_len]
    out = np.full((target_len, num_events), pad_value, dtype=np.int64)
    out[:len(values)] = values
    return out



def normalize_eeg_window(window_eeg, valid_mask=None):
    window_eeg = np.asarray(window_eeg, dtype=np.float32).copy()
    if valid_mask is None:
        valid_mask = np.ones(len(window_eeg), dtype=np.int64)
    else:
        valid_mask = np.asarray(valid_mask, dtype=np.int64)

    for i in range(len(window_eeg)):
        if i >= len(valid_mask) or valid_mask[i] == 0:
            continue
        window_eeg[i] = standardize_epoch(window_eeg[i])
    return window_eeg


def normalize_psg_window(window_psg, valid_mask=None):
    window_psg = np.asarray(window_psg, dtype=np.float32).copy()
    if valid_mask is None:
        valid_mask = np.ones(len(window_psg), dtype=np.int64)
    for i in range(len(window_psg)):
        if i >= len(valid_mask) or valid_mask[i] == 0:
            continue
        for channel_idx in range(window_psg.shape[1]):
            channel = window_psg[i, channel_idx]
            if np.any(channel):
                window_psg[i, channel_idx] = standardize_epoch(channel)
    return window_psg



def _safe_int(x, default=-100):
    try:
        return int(x)
    except Exception:
        return default



def _extract_event_targets(data, n_epochs, event_names):
    num_events = len(event_names)
    event_map = {name: i for i, name in enumerate(event_names)}
    out = np.full((n_epochs, num_events), -100, dtype=np.int64)

    if "waveform_features" in data:
        try:
            waveform_features = data["waveform_features"].tolist()
            for i in range(min(n_epochs, len(waveform_features))):
                feat = waveform_features[i]
                if not isinstance(feat, dict):
                    continue
                key_map = {
                    "alpha": ["has_alpha"],
                    "slowwave": ["has_slowwave"],
                    "spindle": ["has_spindle"],
                    "rem": ["has_rem"],
                    "vertex": ["has_vertex"],
                    "kcomplex": ["has_kcomplex"],
                }
                for name, keys in key_map.items():
                    for key in keys:
                        if key in feat:
                            out[i, event_map[name]] = _safe_int(feat.get(key), -100)
                            break
        except Exception:
            pass

    if "slow_wave_labels" in data:
        try:
            slow_wave_labels = np.asarray(data["slow_wave_labels"], dtype=np.int64).reshape(-1)
            for i in range(min(n_epochs, len(slow_wave_labels))):
                out[i, event_map["slowwave"]] = int(slow_wave_labels[i])
        except Exception:
            pass

    return out



def load_preprocessed_sequences(processed_dirs, cfg):
    all_windows = []
    subject_meta = []
    subject_counter = 0

    for processed_dir in processed_dirs:
        files = list_morphology_files(processed_dir)
        print(f"扫描目录: {processed_dir}，发现 {len(files)} 个 npz")

        for file_path in files:
            try:
                data = np.load(file_path, allow_pickle=True)
            except Exception as e:
                print(f"跳过损坏文件 {file_path}: {e}")
                continue

            if "eeg_epochs" not in data:
                print(f"跳过 {file_path}，因为没有 eeg_epochs")
                continue

            eeg_epochs = np.asarray(data["eeg_epochs"], dtype=np.float32)
            if eeg_epochs.ndim == 1:
                eeg_epochs = eeg_epochs.reshape(1, -1)
            eog_loc_epochs = np.asarray(
                data["eog_loc_epochs"] if "eog_loc_epochs" in data else np.zeros_like(eeg_epochs),
                dtype=np.float32
            )
            eog_roc_epochs = np.asarray(
                data["eog_roc_epochs"] if "eog_roc_epochs" in data else np.zeros_like(eeg_epochs),
                dtype=np.float32
            )
            eog_available = np.asarray(
                data["eog_available"] if "eog_available" in data else [0.0, 0.0],
                dtype=np.float32
            ).reshape(2)
            n_signal_epochs = min(len(eeg_epochs), len(eog_loc_epochs), len(eog_roc_epochs))

            subject_id = str(data["subject_id"]) if "subject_id" in data else os.path.basename(file_path)

            clean_epochs = []
            kept_indices = []
            for i in range(n_signal_epochs):
                eeg_epoch = standardize_epoch(fix_length(eeg_epochs[i], cfg.sample_length))
                loc_epoch = fix_length(eog_loc_epochs[i], cfg.sample_length)
                roc_epoch = fix_length(eog_roc_epochs[i], cfg.sample_length)
                loc_epoch = standardize_epoch(loc_epoch) if eog_available[0] > 0 else np.zeros_like(loc_epoch)
                roc_epoch = standardize_epoch(roc_epoch) if eog_available[1] > 0 else np.zeros_like(roc_epoch)
                epoch = np.stack([eeg_epoch, loc_epoch, roc_epoch], axis=0)
                if np.any(np.isnan(epoch)) or np.any(np.isinf(epoch)):
                    continue
                clean_epochs.append(epoch)
                kept_indices.append(i)

            if len(clean_epochs) < cfg.min_valid_epochs_per_subject:
                print(f"跳过 {subject_id}，有效 epoch 太少: {len(clean_epochs)}")
                continue

            clean_epochs = np.asarray(clean_epochs, dtype=np.float32)
            event_targets_all = _extract_event_targets(data, n_signal_epochs, cfg.event_names)[kept_indices]

            starts = build_window_starts(len(clean_epochs), cfg.seq_len, cfg.stride)
            window_count = 0

            for start in starts:
                end = min(start + cfg.seq_len, len(clean_epochs))
                window = pad_psg_epochs(clean_epochs[start:end], cfg.seq_len, cfg.sample_length)
                valid_mask = pad_valid_mask(end - start, cfg.seq_len)
                window = normalize_psg_window(window, valid_mask)
                event_window = pad_event_matrix(event_targets_all[start:end], cfg.seq_len, len(cfg.event_names), pad_value=-100)
                if not np.any(event_window >= 0):
                    continue

                all_windows.append({
                    "subject_id": subject_id,
                    "eeg": window,
                    "eog_available": eog_available,
                    "valid_mask": valid_mask,
                    "event_targets": event_window,
                })
                window_count += 1

                if cfg.max_windows is not None and len(all_windows) >= cfg.max_windows:
                    break

            subject_meta.append({
                "file_path": file_path,
                "subject_id": subject_id,
                "valid_epochs": int(len(clean_epochs)),
                "eog_available": eog_available.tolist(),
                "num_windows": int(window_count),
            })
            subject_counter += 1

            if cfg.max_subjects is not None and subject_counter >= cfg.max_subjects:
                break
            if cfg.max_windows is not None and len(all_windows) >= cfg.max_windows:
                break

        if cfg.max_subjects is not None and subject_counter >= cfg.max_subjects:
            break
        if cfg.max_windows is not None and len(all_windows) >= cfg.max_windows:
            break

    print(f"EEG event-only upstream pretraining windows: {len(all_windows)}")
    return all_windows, subject_meta


# =========================
# 数据集
# =========================
class EEGPretrainDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        return {
            "eeg": torch.tensor(item["eeg"], dtype=torch.float32),
            "eog_available": torch.tensor(item["eog_available"], dtype=torch.float32),
            "valid_mask": torch.tensor(item["valid_mask"], dtype=torch.long),
            "event_targets": torch.tensor(item["event_targets"], dtype=torch.float32),
            "subject_id": item["subject_id"],
        }


def summarize_event_targets(windows, event_names):
    stats = {}
    for event_idx, event_name in enumerate(event_names):
        values = []
        for item in windows:
            targets = np.asarray(item["event_targets"][:, event_idx])
            valid = np.asarray(item["valid_mask"]) > 0
            labeled = targets[valid & (targets >= 0)]
            if labeled.size > 0:
                values.append(labeled)
        merged = np.concatenate(values) if values else np.asarray([], dtype=np.int64)
        stats[event_name] = {
            "labeled": int(merged.size),
            "positive": int(np.sum(merged == 1)),
            "negative": int(np.sum(merged == 0)),
        }
    return stats


# =========================
# 模型
# =========================
def split_windows_by_subject(windows, valid_ratio=0.1, seed=42):
    subject_ids = sorted({str(item["subject_id"]) for item in windows})
    if len(subject_ids) <= 1:
        return list(windows), list(windows), subject_ids, subject_ids

    rng = np.random.RandomState(seed)
    shuffled = np.asarray(subject_ids, dtype=object)
    rng.shuffle(shuffled)

    n_valid_subjects = max(1, int(round(len(shuffled) * valid_ratio)))
    n_valid_subjects = min(n_valid_subjects, len(shuffled) - 1)
    valid_subjects = set(str(x) for x in shuffled[:n_valid_subjects].tolist())
    train_subjects = set(str(x) for x in shuffled[n_valid_subjects:].tolist())

    train_windows = [item for item in windows if str(item["subject_id"]) in train_subjects]
    valid_windows = [item for item in windows if str(item["subject_id"]) in valid_subjects]

    if len(train_windows) == 0 or len(valid_windows) == 0:
        raise RuntimeError(
            "subject-level pretrain split produced an empty train or valid set; "
            "check subject_id values and valid_ratio."
        )

    return (
        train_windows,
        valid_windows,
        sorted(train_subjects),
        sorted(valid_subjects),
    )


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
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x).squeeze(-1)
        x = self.proj(x)
        return x


class SleepTextBackbone(nn.Module):
    """
    保留 04 中的类名、模块名和 forward 接口。

    关键拆分：
      1) encode_epochs: 原始 EEG/LOC/ROC -> 30秒 epoch embedding [B,S,H]
      2) encode_context: epoch embedding -> 位置编码 -> Global Transformer

    这样 Full 分支可以正常端到端更新全部 backbone；
    Masked 分支则可以在 epoch embedding 处 detach 后，只训练 Global Transformer 的上下文能力。
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
        eeg_kernel_size=7,
    ):
        super().__init__()
        self.eeg_encoder = EEGEpochEncoder(
            out_dim=eeg_hidden_dim,
            kernel_size=eeg_kernel_size,
            dropout=dropout,
        )
        self.eog_encoder = EEGEpochEncoder(
            out_dim=eeg_hidden_dim,
            kernel_size=eeg_kernel_size,
            dropout=dropout,
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
            activation="gelu",
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=seq_num_layers)

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

    def encode_context(self, epoch_embeddings, valid_mask=None, epoch_mask=None, replacement_emb=None):
        """只做 token 级扰动 + 位置编码 + Global Transformer，不重新跑CNN/Fusion。"""
        x = epoch_embeddings

        if epoch_mask is not None:
            mask = epoch_mask.unsqueeze(-1)
            if replacement_emb is None:
                replacement_emb = self.mask_token.expand(x.size(0), x.size(1), -1)
            replacement_emb = replacement_emb.to(device=x.device, dtype=x.dtype)
            x = torch.where(mask, replacement_emb, x)

        seq_len = x.size(1)
        x = self.drop(self.norm(x + self.pos_embed[:, :seq_len, :]))

        if valid_mask is not None:
            key_padding_mask = (valid_mask == 0)
            x = self.seq_encoder(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.seq_encoder(x)
        return x

    def forward(self, eeg, eog_available=None, valid_mask=None, epoch_mask=None, replacement_emb=None):
        epoch_embeddings = self.encode_epochs(eeg, eog_available=eog_available)
        return self.encode_context(
            epoch_embeddings,
            valid_mask=valid_mask,
            epoch_mask=epoch_mask,
            replacement_emb=replacement_emb,
        )


class FullPlusDetachedBERTPretrainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = SleepTextBackbone(
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
        h = cfg.eeg_hidden_dim

        # 两个任务条件分布不同，因此不再共享最后的事件分类头。
        self.full_event_head = nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, len(cfg.event_names)),
        )
        self.masked_event_head = nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, len(cfg.event_names)),
        )

    def encode_epoch_tokens(self, eeg, eog_available=None):
        return self.backbone.encode_epochs(eeg, eog_available=eog_available)

    def full_forward_from_tokens(self, epoch_tokens, valid_mask=None):
        contextual = self.backbone.encode_context(epoch_tokens, valid_mask=valid_mask)
        return self.full_event_head(contextual)

    def masked_forward_from_tokens(self, epoch_tokens, valid_mask=None, mask_token_mask=None):
        contextual = self.backbone.encode_context(
            epoch_tokens,
            valid_mask=valid_mask,
            epoch_mask=mask_token_mask,
            replacement_emb=None,
        )
        return self.masked_event_head(contextual)


# =========================
# Full Event + detached BERT Masked Event training
# =========================
def _rand(shape, device, generator=None):
    if generator is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=generator, device="cpu").to(device)


def _randperm(n, device, generator=None):
    if generator is None:
        return torch.randperm(n, device=device)
    return torch.randperm(n, generator=generator, device="cpu").to(device)


def _randint(high, shape, device, generator=None):
    if generator is None:
        return torch.randint(high, shape, device=device)
    return torch.randint(high, shape, generator=generator, device="cpu").to(device)


def generate_target_mask(valid_mask, event_targets, mask_ratio, min_masked_epochs=1, generator=None):
    """选择需要做masked-event预测的epoch；target_mask只决定损失位置。"""
    valid_positions = valid_mask > 0
    labeled_positions = (event_targets >= 0).any(dim=-1)
    candidates = valid_positions & labeled_positions

    target_mask = candidates & (_rand(candidates.shape, candidates.device, generator) < float(mask_ratio))

    # 每个样本至少选择 min_masked_epochs 个目标，同时至少留下1个可见上下文epoch。
    for batch_idx in range(target_mask.size(0)):
        candidate_indices = torch.where(candidates[batch_idx])[0]
        n_candidates = int(candidate_indices.numel())
        if n_candidates == 0:
            continue

        required = min(max(int(min_masked_epochs), 1), n_candidates)
        current = int(target_mask[batch_idx].sum().item())
        if current < required:
            available = candidate_indices[~target_mask[batch_idx, candidate_indices]]
            need = required - current
            if available.numel() > 0:
                order = _randperm(available.numel(), available.device, generator)
                target_mask[batch_idx, available[order[:need]]] = True

        valid_count = int(valid_positions[batch_idx].sum().item())
        if valid_count > 1 and int(target_mask[batch_idx].sum().item()) >= valid_count:
            target_indices = torch.where(target_mask[batch_idx])[0]
            keep_idx = int(_randint(target_indices.numel(), (1,), target_indices.device, generator).item())
            target_mask[batch_idx, target_indices[keep_idx]] = False

    return target_mask


def split_bert_masks(target_mask, cfg, generator=None):
    """目标epoch内部执行80% [MASK] / 10% keep / 10% random replacement。"""
    p_mask = float(cfg.mask_token_prob)
    p_keep = float(cfg.keep_original_prob)
    p_random = float(cfg.random_replace_prob)
    total = p_mask + p_keep + p_random
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"BERT replacement probabilities must sum to 1, got {total:.6f}")

    values = _rand(target_mask.shape, target_mask.device, generator)
    mask_token_mask = target_mask & (values < p_mask)
    keep_original_mask = target_mask & (values >= p_mask) & (values < p_mask + p_keep)
    random_replace_mask = target_mask & (values >= p_mask + p_keep)
    return mask_token_mask, keep_original_mask, random_replace_mask


def apply_random_token_replacement(epoch_tokens, valid_mask, random_replace_mask, generator=None):
    """
    在30秒 epoch embedding层做随机token替换，更接近BERT的token-level corruption。
    输入应当已经detach，因此不会通过随机源token向CNN/Fusion回传masked梯度。
    """
    corrupted = epoch_tokens.clone()
    replace_count = int(random_replace_mask.sum().item())
    if replace_count == 0:
        return corrupted

    pool = epoch_tokens[valid_mask > 0]
    if pool.size(0) == 0:
        return corrupted

    sampled = _randint(pool.size(0), (replace_count,), epoch_tokens.device, generator)
    corrupted[random_replace_mask] = pool[sampled]
    return corrupted


def full_active_mask(valid_mask, event_targets):
    return (valid_mask > 0).unsqueeze(-1).expand_as(event_targets) & (event_targets >= 0)


def masked_active_mask(valid_mask, event_targets, target_mask):
    return (
        (valid_mask > 0).unsqueeze(-1)
        & target_mask.unsqueeze(-1)
        & (event_targets >= 0)
    )


def compute_bce_on_active(logits, targets, active):
    labeled_count = int(active.sum().item())
    if labeled_count == 0:
        return logits.sum() * 0.0, 0
    return F.binary_cross_entropy_with_logits(logits[active], targets[active]), labeled_count


@dataclass
class TrainOutput:
    loss: float
    full_event_loss: float
    masked_event_loss: float
    full_labeled_targets: int
    masked_labeled_targets: int
    target_epochs: int
    mask_token_epochs: int
    kept_original_epochs: int
    random_replace_epochs: int


def compute_combined_loss(model, batch, cfg, mask_generator=None):
    eeg = batch["eeg"]
    eog_available = batch["eog_available"]
    valid_mask = batch["valid_mask"]
    event_targets = batch["event_targets"]

    # 先只编码一次：原始PSG -> 30秒epoch token。
    epoch_tokens = model.encode_epoch_tokens(eeg, eog_available=eog_available)

    # -------- 分支1：Full Event --------
    # 不detach，因此Full loss端到端更新EEG/EOG encoder、fusion、Global Transformer。
    full_logits = model.full_forward_from_tokens(epoch_tokens, valid_mask=valid_mask)
    full_active = full_active_mask(valid_mask, event_targets)
    full_loss, full_count = compute_bce_on_active(full_logits, event_targets, full_active)

    # -------- 分支2：BERT Masked Event --------
    # 关键：从同一批epoch token开始，但先detach。
    # 因此Masked loss无法回到EEG/EOG encoder和fusion，只会更新后面的Global Transformer等模块。
    if bool(getattr(cfg, "detach_masked_epoch_embeddings", True)):
        masked_tokens = epoch_tokens.detach()
    else:
        masked_tokens = epoch_tokens

    target_mask = generate_target_mask(
        valid_mask=valid_mask,
        event_targets=event_targets,
        mask_ratio=cfg.mask_ratio,
        min_masked_epochs=cfg.min_masked_epochs,
        generator=mask_generator,
    )
    mask_token_mask, keep_original_mask, random_replace_mask = split_bert_masks(
        target_mask,
        cfg,
        generator=mask_generator,
    )

    # 10% random：在embedding层替换为其他有效30秒epoch token。
    masked_tokens = apply_random_token_replacement(
        masked_tokens,
        valid_mask=valid_mask,
        random_replace_mask=random_replace_mask,
        generator=mask_generator,
    )

    # 80% [MASK]：由backbone.encode_context用可学习mask_token替换。
    # 10% keep：不做任何改变，但仍属于target并计算masked loss。
    masked_logits = model.masked_forward_from_tokens(
        masked_tokens,
        valid_mask=valid_mask,
        mask_token_mask=mask_token_mask,
    )
    masked_active = masked_active_mask(valid_mask, event_targets, target_mask)
    masked_loss, masked_count = compute_bce_on_active(masked_logits, event_targets, masked_active)

    total_loss = (
        float(cfg.lambda_full_event) * full_loss
        + float(cfg.lambda_masked_event) * masked_loss
    )

    stats = {
        "full_loss": full_loss,
        "masked_loss": masked_loss,
        "full_count": full_count,
        "masked_count": masked_count,
        "target_epochs": int(target_mask.sum().item()),
        "mask_token_epochs": int(mask_token_mask.sum().item()),
        "kept_original_epochs": int(keep_original_mask.sum().item()),
        "random_replace_epochs": int(random_replace_mask.sum().item()),
    }
    return total_loss, stats


def _finalize_output(total_loss_sum, sample_count, full_loss_sum, full_count,
                     masked_loss_sum, masked_count, target_epochs, mask_token_epochs,
                     kept_original_epochs, random_replace_epochs):
    if sample_count == 0 or full_count == 0 or masked_count == 0:
        raise RuntimeError("split contains no valid full-event or masked-event targets")
    return TrainOutput(
        loss=total_loss_sum / sample_count,
        full_event_loss=full_loss_sum / full_count,
        masked_event_loss=masked_loss_sum / masked_count,
        full_labeled_targets=full_count,
        masked_labeled_targets=masked_count,
        target_epochs=target_epochs,
        mask_token_epochs=mask_token_epochs,
        kept_original_epochs=kept_original_epochs,
        random_replace_epochs=random_replace_epochs,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, cfg):
    model.train()
    total_loss_sum = 0.0
    sample_count = 0
    full_loss_sum = 0.0
    full_count = 0
    masked_loss_sum = 0.0
    masked_count = 0
    target_epochs = 0
    mask_token_epochs = 0
    kept_original_epochs = 0
    random_replace_epochs = 0

    for batch in tqdm(loader, desc="EEG Full + detached BERT context pretrain", leave=False):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.type == "cuda")):
            loss, stats = compute_combined_loss(model, batch, cfg, mask_generator=None)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(batch["eeg"].size(0))
        total_loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        full_loss_sum += float(stats["full_loss"].item()) * stats["full_count"]
        full_count += stats["full_count"]
        masked_loss_sum += float(stats["masked_loss"].item()) * stats["masked_count"]
        masked_count += stats["masked_count"]
        target_epochs += stats["target_epochs"]
        mask_token_epochs += stats["mask_token_epochs"]
        kept_original_epochs += stats["kept_original_epochs"]
        random_replace_epochs += stats["random_replace_epochs"]

    return _finalize_output(
        total_loss_sum, sample_count,
        full_loss_sum, full_count,
        masked_loss_sum, masked_count,
        target_epochs, mask_token_epochs,
        kept_original_epochs, random_replace_epochs,
    )


@torch.no_grad()
def evaluate_one_epoch(model, loader, device, cfg):
    model.eval()
    total_loss_sum = 0.0
    sample_count = 0
    full_loss_sum = 0.0
    full_count = 0
    masked_loss_sum = 0.0
    masked_count = 0
    target_epochs = 0
    mask_token_epochs = 0
    kept_original_epochs = 0
    random_replace_epochs = 0

    # 固定验证集target和80/10/10抽样，避免每轮验证因随机mask不同而抖动。
    validation_generator = torch.Generator(device="cpu")
    validation_generator.manual_seed(int(cfg.validation_mask_seed))

    for batch in tqdm(loader, desc="EEG Full + detached BERT context valid", leave=False):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.type == "cuda")):
            loss, stats = compute_combined_loss(
                model, batch, cfg, mask_generator=validation_generator
            )

        batch_size = int(batch["eeg"].size(0))
        total_loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        full_loss_sum += float(stats["full_loss"].item()) * stats["full_count"]
        full_count += stats["full_count"]
        masked_loss_sum += float(stats["masked_loss"].item()) * stats["masked_count"]
        masked_count += stats["masked_count"]
        target_epochs += stats["target_epochs"]
        mask_token_epochs += stats["mask_token_epochs"]
        kept_original_epochs += stats["kept_original_epochs"]
        random_replace_epochs += stats["random_replace_epochs"]

    return _finalize_output(
        total_loss_sum, sample_count,
        full_loss_sum, full_count,
        masked_loss_sum, masked_count,
        target_epochs, mask_token_epochs,
        kept_original_epochs, random_replace_epochs,
    )


# =========================
# 主流程
# =========================
def build_lightweight_backbone_checkpoint(model, cfg, epoch, valid_full_loss, valid_total_loss):
    return {
        "backbone_state_dict": copy.deepcopy(model.backbone.state_dict()),
        "eeg_encoder_state_dict": copy.deepcopy(model.backbone.eeg_encoder.state_dict()),
        "config": cfg.__dict__,
        "epoch": int(epoch),
        "valid_loss": float(valid_full_loss),
        "valid_full_event_loss": float(valid_full_loss),
        "valid_total_loss": float(valid_total_loss),
        "pretrain_type": cfg.pretrain_type,
        "event_names": cfg.event_names,
        "mask_ratio": float(cfg.mask_ratio),
        "lambda_masked_event": float(cfg.lambda_masked_event),
        "detach_masked_epoch_embeddings": bool(cfg.detach_masked_epoch_embeddings),
    }


def train_sleep_eeg_pretrain(config=None):
    cfg = config if config is not None else EEGPretrainConfig()
    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    windows, subject_meta = load_preprocessed_sequences(cfg.pretrain_dirs, cfg)
    if len(windows) == 0:
        raise RuntimeError("没有找到含有效 event 标签的 EEG 上游预训练窗口。")

    event_target_stats = summarize_event_targets(windows, cfg.event_names)
    print(f"[Event Targets] {json.dumps(event_target_stats, ensure_ascii=False)}")

    train_windows, valid_windows, train_subject_ids, valid_subject_ids = split_windows_by_subject(
        windows,
        valid_ratio=0.1,
        seed=cfg.seed,
    )
    print(
        "[Pretrain Split] subject_level | "
        f"train_subjects={len(train_subject_ids)}, valid_subjects={len(valid_subject_ids)}, "
        f"train_windows={len(train_windows)}, valid_windows={len(valid_windows)}"
    )
    print(f"[Pretrain Split] valid subjects: {valid_subject_ids}")
    print(
        "[Objective] "
        f"Full={cfg.lambda_full_event} + Masked={cfg.lambda_masked_event} | "
        f"mask_ratio={cfg.mask_ratio} | BERT={cfg.mask_token_prob}/{cfg.keep_original_prob}/{cfg.random_replace_prob} | "
        f"masked_tokens_detach={cfg.detach_masked_epoch_embeddings}"
    )
    checkpoint_selection = str(
        getattr(cfg, "checkpoint_selection_metric", "valid_full_event_loss")
    ).strip().lower()
    if checkpoint_selection not in {
        "valid_full_event_loss", "valid_masked_event_loss"
    }:
        raise ValueError(
            "checkpoint_selection_metric must be 'valid_full_event_loss' or "
            f"'valid_masked_event_loss', got {checkpoint_selection!r}"
        )
    print(
        "[Checkpoint Selection] best backbone is selected by "
        f"{checkpoint_selection}."
    )

    train_ds = EEGPretrainDataset(train_windows)
    valid_ds = EEGPretrainDataset(valid_windows)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = FullPlusDetachedBERTPretrainer(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    history = []
    best_valid_score = float("inf")
    best_state = None
    best_epoch = 0

    backbone_path = os.path.join(cfg.output_dir, "sleep_eeg_bert_backbone_pretrained.pt")
    legacy_encoder_path = os.path.join(cfg.output_dir, "eeg_encoder_pretrained.pt")
    full_model_path = os.path.join(cfg.output_dir, "sleep_eeg_full_detached_bert_pretrainer_best.pt")

    for epoch in range(cfg.num_epochs):
        train_out = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
        valid_out = evaluate_one_epoch(model, valid_loader, device, cfg)

        row = {
            "epoch": epoch + 1,
            "train_total_loss": float(train_out.loss),
            "valid_total_loss": float(valid_out.loss),
            "train_full_event_loss": float(train_out.full_event_loss),
            "valid_full_event_loss": float(valid_out.full_event_loss),
            "train_masked_event_loss": float(train_out.masked_event_loss),
            "valid_masked_event_loss": float(valid_out.masked_event_loss),
            "train_full_labeled_targets": int(train_out.full_labeled_targets),
            "valid_full_labeled_targets": int(valid_out.full_labeled_targets),
            "train_masked_labeled_targets": int(train_out.masked_labeled_targets),
            "valid_masked_labeled_targets": int(valid_out.masked_labeled_targets),
            "train_target_epochs": int(train_out.target_epochs),
            "valid_target_epochs": int(valid_out.target_epochs),
            "train_mask_token_epochs": int(train_out.mask_token_epochs),
            "valid_mask_token_epochs": int(valid_out.mask_token_epochs),
            "train_kept_original_epochs": int(train_out.kept_original_epochs),
            "valid_kept_original_epochs": int(valid_out.kept_original_epochs),
            "train_random_replace_epochs": int(train_out.random_replace_epochs),
            "valid_random_replace_epochs": int(valid_out.random_replace_epochs),
        }
        history.append(row)

        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"train_total={train_out.loss:.6f} "
            f"(full={train_out.full_event_loss:.6f}, masked={train_out.masked_event_loss:.6f}) | "
            f"valid_total={valid_out.loss:.6f} "
            f"(full={valid_out.full_event_loss:.6f}, masked={valid_out.masked_event_loss:.6f})"
        )
        print(
            "[Valid BERT Replacement] "
            f"targets={valid_out.target_epochs}, mask={valid_out.mask_token_epochs}, "
            f"keep={valid_out.kept_original_epochs}, random={valid_out.random_replace_epochs}"
        )

        save_every = int(getattr(cfg, "save_every_epochs", 0))
        if getattr(cfg, "save_epoch_checkpoints", False) and save_every > 0 and (epoch + 1) % save_every == 0:
            epoch_backbone_path = os.path.join(
                cfg.output_dir,
                f"sleep_eeg_bert_backbone_epoch_{epoch + 1:03d}.pt",
            )
            epoch_state = build_lightweight_backbone_checkpoint(
                model,
                cfg,
                epoch + 1,
                valid_out.full_event_loss,
                valid_out.loss,
            )
            torch.save(epoch_state, epoch_backbone_path)
            print(f"[Epoch Checkpoint] epoch={epoch + 1} | saved: {epoch_backbone_path}")

        # 为尽量保持原Event-only的下游性能，最佳模型仍按Full Event验证BCE选择。
        selected_valid_loss = (
            valid_out.masked_event_loss
            if checkpoint_selection == "valid_masked_event_loss"
            else valid_out.full_event_loss
        )
        if selected_valid_loss < best_valid_score:
            best_valid_score = selected_valid_loss
            best_epoch = epoch + 1
            best_state = {
                "backbone_state_dict": copy.deepcopy(model.backbone.state_dict()),
                "eeg_encoder_state_dict": copy.deepcopy(model.backbone.eeg_encoder.state_dict()),
                "full_event_head_state_dict": copy.deepcopy(model.full_event_head.state_dict()),
                "masked_event_head_state_dict": copy.deepcopy(model.masked_event_head.state_dict()),
                "full_model_state_dict": copy.deepcopy(model.state_dict()),
                "config": cfg.__dict__,
                "best_valid_loss": float(best_valid_score),
                "best_valid_full_event_loss": float(valid_out.full_event_loss),
                "best_valid_total_loss": float(valid_out.loss),
                "best_valid_masked_event_loss": float(valid_out.masked_event_loss),
                "checkpoint_selection": checkpoint_selection,
                "best_epoch": int(best_epoch),
                "pretrain_type": cfg.pretrain_type,
                "event_names": cfg.event_names,
            }
            torch.save(best_state, backbone_path)
            torch.save(
                {
                    "eeg_encoder_state_dict": best_state["eeg_encoder_state_dict"],
                    "config": cfg.__dict__,
                    "best_valid_loss": float(best_valid_score),
                    "best_epoch": int(best_epoch),
                    "pretrain_type": "legacy_epoch_encoder_only",
                    "checkpoint_selection": checkpoint_selection,
                },
                legacy_encoder_path,
            )
            torch.save(best_state["full_model_state_dict"], full_model_path)
            print(
                f"[New Best] epoch={best_epoch} | selected_by={checkpoint_selection} | "
                f"selection_loss={best_valid_score:.6f} | "
                f"valid_full={valid_out.full_event_loss:.6f} | "
                f"valid_masked={valid_out.masked_event_loss:.6f} | "
                f"valid_total={valid_out.loss:.6f}"
            )

    if best_state is None:
        raise RuntimeError("Full + detached BERT 上游预训练未产生有效权重。")

    last_full_model_path = os.path.join(cfg.output_dir, "sleep_eeg_full_detached_bert_pretrainer_last.pt")
    history_path = os.path.join(cfg.output_dir, "pretrain_history_full_detached_bert.json")
    meta_path = os.path.join(cfg.output_dir, "dataset_meta_full_detached_bert.json")

    torch.save(model.state_dict(), last_full_model_path)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_subjects": len(subject_meta),
                "num_windows": len(windows),
                "split_type": "subject_level",
                "num_train_subjects": len(train_subject_ids),
                "num_valid_subjects": len(valid_subject_ids),
                "num_train_windows": len(train_windows),
                "num_valid_windows": len(valid_windows),
                "train_subject_ids": train_subject_ids,
                "valid_subject_ids": valid_subject_ids,
                "pretrain_type": cfg.pretrain_type,
                "checkpoint_selection": checkpoint_selection,
                "mask_ratio": float(cfg.mask_ratio),
                "mask_token_prob": float(cfg.mask_token_prob),
                "keep_original_prob": float(cfg.keep_original_prob),
                "random_replace_prob": float(cfg.random_replace_prob),
                "lambda_full_event": float(cfg.lambda_full_event),
                "lambda_masked_event": float(cfg.lambda_masked_event),
                "detach_masked_epoch_embeddings": bool(cfg.detach_masked_epoch_embeddings),
                "validation_mask_seed": int(cfg.validation_mask_seed),
                "save_epoch_checkpoints": bool(cfg.save_epoch_checkpoints),
                "save_every_epochs": int(cfg.save_every_epochs),
                "epoch_checkpoint_pattern": "sleep_eeg_bert_backbone_epoch_{epoch:03d}.pt",
                "best_epoch": best_epoch,
                "best_valid_selection_loss": float(best_valid_score),
                "best_valid_full_event_loss": float(
                    best_state["best_valid_full_event_loss"]
                ),
                "best_valid_masked_event_loss": float(
                    best_state["best_valid_masked_event_loss"]
                ),
                "best_backbone_path": backbone_path,
                "best_full_model_path": full_model_path,
                "last_full_model_path": last_full_model_path,
                "event_names": cfg.event_names,
                "event_target_stats": event_target_stats,
                "subjects": subject_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Best epoch: {best_epoch} | selected_by={checkpoint_selection} | "
        f"best selection BCE: {best_valid_score:.6f}"
    )
    print(f"EEG Full + detached BERT context 上游预训练完成，backbone已保存到: {backbone_path}")
    print(f"兼容旧接口的 EEG encoder 权重已保存到: {legacy_encoder_path}")
    print(f"Last full model saved to: {last_full_model_path}")
    return backbone_path


if __name__ == "__main__":
    train_sleep_eeg_pretrain()
