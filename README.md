# Spatial-ASAD

Auditory Spatial Attention Detection (ASAD) using EEG, implemented in PyTorch.

This repository contains a CNN baseline and six proposed model variants (CP-CNN, HCCF-Net, SCR-CNN, CE-CNN, STM-CNN, CNN-Mamba), plus domain-generalization variants (CORAL, DANN, MixUp). Each experiment directory is a self-contained 5-file package (`config.py`, `main.py`, `model.py`, `train_valid_and_test.py`, `AADdataset.py`) and must be run from its own directory.

## Repository layout

```
Spatial-ASAD/
├── Initial/                         # CNN baseline under 4 protocols
│   ├── CNN-baseline_Subject_adaptive/{1s,2s,5s,10s}   # LOSO pretrain + per-subject finetune
│   ├── CNN-baseline_loso/{1s,2s,5s,10s}               # leave-one-subject-out
│   ├── CNN-baseline_loto/{1s,2s,5s,10s}               # leave-one-trial-out
│   └── CNN-baseline_within_trial/{1s,2s,5s,10s}       # within-subject k-fold
├── Improvement/
│   ├── Model_imp/{CP-CNN,HCCF-Net,SCR-CNN,CE-CNN,STM-CNN,CNN-Mamba}/subjective_adaptive/{1s,2s,5s,10s}
│   ├── Model_and_Domain_imp/{...same 6 methods...}/subjective_adaptive/{1s,2s,5s,10s}
│   └── Domain_imp/{CORAL,DANN,MixUp}/subjective_adaptive/{1s,2s,5s,10s}
├── Figure/                          # MATLAB scripts that plot trained results
├── data/                            # put your dataset here (see below)
└── spatial-focus-of-attention-csp-and-rgc-master/      # upstream KU Leuven MATLAB reference
```

## Data

### Source dataset

The code is built on the **Das et al. 2016 auditory attention decoding dataset** from KU Leuven
("das-2016") — the same dataset the bundled reference implementation uses
(`spatial-focus-of-attention-csp-and-rgc-master/.../Subject_adaptive.m` sets `params.dataset = 'das-2016'`).
Reference: N. Das, W. Biesmans, A. Bertrand, T. Francart, "The effect of head-related
filtering and ear-specific decoding bias on auditory attention detection," *J. Neural Eng.*, 2016.

- **Download (Zenodo):** https://zenodo.org/records/4004271
- **DOI:** 10.5281/zenodo.4004271

### Preprocessing (official MATLAB script)

The dataset is distributed as raw per-subject files (`S*.mat`). The official upstream preprocessing script
is bundled in this repo:

```
spatial-focus-of-attention-csp-and-rgc-master/RGC/subject_adaptive/preprocessData.m
```

It band-pass filters the EEG (1–32 Hz), downsamples it, splits each trial into 60 s segments, and writes
one file per subject (`preprocessed_data/dataS*.mat` containing `eegTrials`, `attendedEar`, `fs`). Run it
in MATLAB from the directory that contains the downloaded raw subject files. It may require
[Tensorlab](https://www.tensorlab.net/) (see the scripts' comments).

### Expected input file

`KUL3_1D.mat` is **not** a stock download; it is this repo's preprocessed single-file aggregation of the
dataset above. You need to generate it yourself from the official preprocessed output: the experiments read
`data/KUL3_1D.mat` via `h5py`, so the file **must be MATLAB v7.3 (HDF5)** format — a plain v5/v7 `.mat`
will not open. Required contents and shapes:

| key   | shape            | meaning                                    |
|-------|------------------|--------------------------------------------|
| `EEG` | `(16, 8, T, 64)` | subjects × trials × time samples × channels |
| `ENV` | `(16, 8, T)`     | subjects × trials × time samples (label 0/1) |

Place the file as:

```
data/KUL3_1D.mat
```

`T` is the trial length in samples (this repo's `config.py` sets `sample_rate = 128`). The code transposes
these arrays internally and assumes exactly this shape. To build `KUL3_1D.mat`, aggregate the official
preprocessing output (per-subject `preprocessed_data/dataS*.mat` → the `(16, 8, T, 64)` EEG tensor) together
with the corresponding `ENV` labels into a single HDF5 file with the two keys above.

A converter script is provided to do this automatically:

```
python data/build_kul3_1d.py --preprocessed path/to/preprocessed_data --out data/KUL3_1D.mat
```

Before running it, review the requirements coded into the script:

- **16 subjects, 8 trials each** (see `main.tex` §数据集).
- **128 Hz** — the official `preprocessData.m` downsamples to 64 Hz, so set
  `params.targetSampleRate = 128` in that script before running it (or pass `--resample`).
- `ENV` labels: the script maps official `attendedEar` 1/2 → 0/1 (`--swap` flips it).

## Install

```bash
pip install -r requirements.txt
```

**CNN-Mamba only:** its 8 `model.py` files do `from mamba_ssm import Mamba`, which requires a CUDA-compiled `mamba_ssm` (uncomment the line in `requirements.txt` or install it separately). All other methods run on the core dependencies.

## Run an experiment

From **inside** the experiment directory (the scripts `import config`, so the working directory matters):

```bash
cd Initial/CNN-baseline_Subject_adaptive/1s
python main.py
```

The same pattern applies to every method:

```bash
cd <group>/<method>/subjective_adaptive/<window>
python main.py
```

Results are written to `<experiment_dir>/results/result_*.csv`. Checkpoints go to `pretrain_model/` and `finetune_model/` (created automatically).

## GPU

The device is set to `cuda:0` when CUDA is available, otherwise CPU. To pick a different card, set `CUDA_VISIBLE_DEVICES` before running:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py
```

## Plotting

After training (which produces the `result_*.csv` files), run the figures from the `Figure/` directory in MATLAB:

```matlab
plot_CNN    % baseline protocols
plot_IMP    % proposed methods
```

## Dependencies

torch, numpy, h5py, scikit-learn, tqdm (`mamba_ssm` only for CNN-Mamba).

## License

See [LICENSE](LICENSE).
