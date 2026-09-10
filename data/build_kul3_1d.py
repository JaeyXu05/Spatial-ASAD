#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_kul3_1d.py — 从官方预处理输出生成本仓库可用的 data/KUL3_1D.mat。

背景
----
本仓库每个实验目录都执行
    h5py.File(cfg.process_data_dir + '/' + cfg.dataset_name)
读取 `data/KUL3_1D.mat`，需要以下两个 HDF5 键:

    EEG  形状 (16, 8, T, 64)   被试 x 试次 x 时间点 x 通道
    ENV  形状 (16, 8, T)       被试 x 试次 x 时间点 (0/1 表示注意力方向)

该文件不是官方直接提供的。官方只给出:
  1) 原始数据 S*.mat  —— KU Leuven Das 2016 数据集 (Zenodo 4004271)
  2) 预处理脚本       —— 见 spatial-focus-of-attention-csp-and-rgc-master/RGC/subject_adaptive/preprocessData.m

本脚本把 2) 的输出 (每被试一个 dataS*.mat, 内含 eegTrials / attendedEar / fs)
聚合成本仓库要的单文件 KUL3_1D.mat。

重要前提 (请先核对, 必要时修改脚本顶部的常量)
------------------------------------------------
- 被试数 = 16    : 报告 main.tex 第 103 行 "16 名受试者"。
- 每被试试次 = 8 : 报告 "实验分为 8 个片段"; 代码 config.py 里 trnum=8。
- 采样率 = 128 Hz: 本仓库 config.py sample_rate=128, AADdataset.py 注释
  "128 means 1s window"。官方 preprocessData.m 默认 targetSampleRate=64,
  会造成采样率不一致。

  **推荐做法**: 先改 preprocessData.m 里 `params.targetSampleRate = 128` 这一行,
  并保持 `segSize=60` 秒, 再跑一遍 —— 这样 fs=128 且每个 60 秒段有 7680 个
  采样点, 与本仓库窗口定义完全一致。若你的输出仍是 64 Hz, 可用 --resample
  让本脚本用 scipy 上采样 (会引入插值误差, 不建议)。

- ENV 标签: 官方 attendedEar 为 1(左)/2(右); 本仓库 ENV 是 0/1。
  本脚本默认 1->0, 2->1。若实际方向约定相反, 用 --swap 翻转。

用法
----
    # (可选) 先把官方 preprocessData.m 的 targetSampleRate 改成 128 再跑一遍
    python data/build_kul3_1d.py --preprocessed path/to/preprocessed_data --out data/KUL3_1D.mat

参数:
    --preprocessed PATH  官方 preprocessData.m 的输出目录 (含 dataS1.mat ... dataS16.mat)
    --out PATH           输出文件 (默认 data/KUL3_1D.mat)
    --subjects N         被试数 (默认 16)
    --trials N           每被试试次数 (默认 8)
    --resample           若 fs != 128 则用 scipy 上采样到 128 Hz (默认仅在 fs=128 时放行)
    --swap               翻转 0/1 标签映射 (默认 1->0, 2->1)

依赖: numpy, h5py, scipy (读 MATLAB v7 .mat 与 --resample 都需要 scipy)。
"""

import argparse
import os
import sys

import h5py
import numpy as np
from scipy.io import loadmat

DEFAULT_SUBJECTS = 16
DEFAULT_TRIALS = 8
TARGET_FS = 128.0


def read_subject(path):
    """读取官方 dataS*.mat, 返回 (eeg_trials, attended_ear, fs)。

    eeg_trials: list[np.ndarray], 每个元素形状 (channel, time) —— 官方
                preprocessData.m 存的是 (64, time), 与 HDF5 键的 (time, 64) 相反。
    attended_ear: np.ndarray, 取值为 1(左) / 2(右)。
    fs: float, 采样率。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}。请先运行官方 preprocessData.m。")
    m = loadmat(path)
    if 'eegTrials' not in m or 'attendedEar' not in m or 'fs' not in m:
        raise KeyError(f"{path} 缺少 eegTrials / attendedEar / fs 字段。")
    trials = list(m['eegTrials'][0])            # cell -> list of (ch, time)
    ear = np.asarray(m['attendedEar']).flatten()
    fs = float(np.asarray(m['fs']).flatten()[0])
    return trials, ear, fs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preprocessed', required=True,
                    help='官方 preprocessData.m 的输出目录')
    ap.add_argument('--out', default=os.path.join('data', 'KUL3_1D.mat'))
    ap.add_argument('--subjects', type=int, default=DEFAULT_SUBJECTS)
    ap.add_argument('--trials', type=int, default=DEFAULT_TRIALS)
    ap.add_argument('--resample', action='store_true',
                    help='若 fs!=128 则上采样到 128 Hz (不推荐, 会有插值误差)')
    ap.add_argument('--swap', action='store_true',
                    help='翻转 0/1 标签映射')
    args = ap.parse_args()

    eeg_subjects = []
    env_subjects = []
    fs_seen = None

    for s in range(args.subjects):
        # 官方 preprocessData.m 写名为 data + subject{1}, 即 dataS1.mat ... dataS16.mat
        path = os.path.join(args.preprocessed, f'dataS{s + 1}.mat')
        trials, ear, fs = read_subject(path)
        if fs_seen is None:
            fs_seen = fs
        n = min(len(trials), args.trials)
        if len(trials) < args.trials:
            print(f"警告: {path} 只有 {len(trials)} 个试次, 少于 {args.trials}, 取前 {n} 个。")

        # 统一每个试次的时间长度 T (官方 60 秒段等长; 不同被试若不等长则取最短)。
        T = min(int(t.shape[1]) for t in trials[:n])

        # 每个 cell 元素 (ch, time) -> 截断/裁剪到 T, 再转置为 (time, ch)
        X = np.stack([np.asarray(t[:, :T]).T for t in trials[:n]], axis=0)  # (n, T, ch)
        # 标签: 1->0, 2->1 (或 swap); 同一试次内所有时间点相同
        lab = np.where(ear[:n] == 1, 0, 1).astype(float) if not args.swap \
            else np.where(ear[:n] == 1, 1, 0).astype(float)
        Y = np.repeat(lab[:, None], T, axis=1)  # (n, T)

        eeg_subjects.append(X)
        env_subjects.append(Y)

    # 聚合为 (subjects, trials, T, ch) / (subjects, trials, T)
    eeg = np.stack(eeg_subjects, axis=0).astype(float)
    env = np.stack(env_subjects, axis=0).astype(float)
    print(f"聚合结果 EEG={eeg.shape}, ENV={env.shape}, fs={fs_seen}")

    # 采样率处理
    if abs(fs_seen - TARGET_FS) > 0.5:
        if args.resample:
            from scipy.signal import resample_poly
            up = TARGET_FS / fs_seen  # 64 -> 128 => up=2
            L = int(round(eeg.shape[2] * up))
            orig_T, orig_env_T = eeg.shape[2], env.shape[2]
            eeg = resample_poly(eeg, int(round(TARGET_FS)), int(round(fs_seen)), axis=2)
            # ENV 是同值标签, 直接按同比例扩展(重复/插值均可, 这里取最近邻)
            idx = np.linspace(0, env.shape[2] - 1, L).astype(int)
            env = env[:, :, idx]
            print(f"已上采样: fs {fs_seen} -> {TARGET_FS}, T {orig_T} -> {eeg.shape[2]}")
        else:
            print(f"错误: 输入 fs={fs_seen} != {TARGET_FS}。")
            print("  请先把官方 preprocessData.m 里 params.targetSampleRate 改成 128 重跑, ")
            print("  或加 --resample 参数 (会引入插值误差, 不建议)。")
            sys.exit(1)

    # 写出为 MATLAB v7.3 语义的 HDF5。
    # main.py 读数据后执行 np.transpose(raw_data); 即它期望 h5py 读出的形状是
    # MATLAB 列优先存储的“反转”形状 (64,T,8,16) / (T,8,16), 再转置回 (16,8,T,64)/(16,8,T)。
    # 因此这里写入时用 eeg.transpose(3,2,1,0) 与 env.transpose(2,1,0)。
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(args.out, 'w') as hf:
        hf.create_dataset('EEG', data=eeg.transpose(3, 2, 1, 0))
        hf.create_dataset('ENV', data=env.transpose(2, 1, 0))
    print(f"已写出 {args.out} (HDF5)。EEG 键形状(读回后)=({eeg.shape[3]},{eeg.shape[2]},{len(trials)},{args.subjects})")


if __name__ == '__main__':
    main()
