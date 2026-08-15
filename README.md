# ED-Former

Official PyTorch implementation of **“ED-Former: Efficient Dehazing Transformer with Attention-Adaptive Feed-Forward Network”**, published in *Signal Processing: Image Communication* (2026).

[[Paper](https://doi.org/10.1016/j.image.2026.117634)] [[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0923596526001578)]

> Jinrong Chen, Yulin He, Xin Wang, Zhongyuan Guo, Jingtong Chen, Zhe Rao, and Yi Xiang, “ED-Former: Efficient dehazing transformer with Attention-Adaptive Feed-Forward Network,” *Signal Processing: Image Communication*, vol. 148, article 117634, 2026.

ED-Former is an extremely lightweight Transformer for single-image dehazing. It combines a Frequency-aware Hierarchical Sampler (FHS), an Attention-Adaptive Feed-Forward Network (AAFFN), and Hierarchical Invariance Loss (HILoss) to preserve high-frequency details while keeping the model below one million parameters.

## Highlights

- **Extremely lightweight:** 0.866 M parameters and 7.36 G MACs for a 256 × 256 input.
- **Frequency-aware Hierarchical Sampler (FHS):** uses wavelet-based decomposition and reconstruction to reduce information loss during feature resizing.
- **Attention-Adaptive Feed-Forward Network (AAFFN):** dynamically refines detail-rich features through a lightweight attention gate.
- **Hierarchical Invariance Loss (HILoss):** uses cosine scheduling to shift the training objective from perceptual structure toward pixel-wise fidelity.
- **Real-time inference:** 75.2 FPS at 256 × 256 on a single NVIDIA RTX 4060 Ti, as reported in the paper.

## Network architecture

<p align="center">
  <img src="figs/figchart.jpg" alt="ED-Former architecture" width="100%">
</p>

## Quantitative comparison

<p align="center">
  <img src="figs/Quantitative%20Comparison.png" alt="Quantitative comparison on dehazing benchmarks" width="100%">
</p>

| Dataset | PSNR | SSIM |
|:--|--:|--:|
| SOTS-indoor (RESIDE-IN) | **38.21** | **0.9942** |
| SOTS-outdoor (RESIDE-OUT) | **33.92** | **0.9827** |
| RS-Haze | **39.61** | **0.9715** |
| O-HAZE (zero-shot) | **15.78** | **0.702** |

The model has 0.866 M parameters and requires 7.36 G MACs for a 256 × 256 input.

## Qualitative comparison

<p align="center">
  <a href="figs/Qualitative%20Comparison.png">
    <img src="figs/Qualitative%20Comparison.png" alt="Qualitative comparison with state-of-the-art dehazing methods" width="100%">
  </a>
</p>

Qualitative comparisons with state-of-the-art dehazing methods on indoor, outdoor, and real-world hazy images. Click the image to view fine-grained details at full resolution.

## Results

The benchmark summaries are included directly in this repository:

| Benchmark | Metric file | Dehazed images |
|:--|:--|:--|
| SOTS-indoor | [`results/RESIDE-IN/ed-former/38.21 \| 0.9942.csv`](results/RESIDE-IN/ed-former/38.21%20%7C%200.9942.csv) | `results/RESIDE-IN/ed-former/imgs/` |
| SOTS-outdoor | [`results/RESIDE-OUT/ed-former/33.92 \| 0.9827.csv`](results/RESIDE-OUT/ed-former/33.92%20%7C%200.9827.csv) | `results/RESIDE-OUT/ed-former/imgs/` |
| RS-Haze | [`results/RSHaze/ed-former/39.61 \| 0.9715.csv`](results/RSHaze/ed-former/39.61%20%7C%200.9715.csv) | `results/RSHaze/ed-former/imgs/` |

To keep the Git repository lightweight, the complete per-image outputs for all three benchmarks are packaged as `ED-Former-complete-results.tar.gz` in the [latest GitHub Release](../../releases/latest). After running `test.py`, newly generated dehazed images and per-image metrics are saved under `results/<DATASET>/ed-former/` using the same layout.

### Runtime and complexity

Runtime was measured in the paper on a single NVIDIA RTX 4060 Ti with a 256 × 256 input.

| Method | Latency (ms) ↓ | FPS ↑ | Params (M) ↓ | MACs (G) ↓ |
|:--|--:|--:|--:|--:|
| GridDehazeNet | 15.3 | 65.2 | 0.956 | 21.49 |
| MSBDN | 19.9 | 50.2 | 31.35 | 41.54 |
| FFA-Net | 93.5 | 10.7 | 4.456 | 287.8 |
| DehazeFormer-s | 18.7 | 53.5 | 1.283 | 13.13 |
| Dehamer | 19.0 | 52.7 | 132.4 | 48.93 |
| **ED-Former** | **13.3** | **75.2** | **0.866** | **7.36** |

### Generalization to real-world haze

For O-HAZE, the checkpoint trained on RESIDE-OUT is evaluated directly without fine-tuning. ED-Former obtains **15.78 dB PSNR** and **0.702 SSIM**, demonstrating transfer from synthetic training data to real-world haze.

## Installation

The code is written in Python 3.7 and requires PyTorch. A CUDA-enabled GPU is required by the current testing script.

```bash
git clone https://github.com/2697166190a-beep/ED-Former.git
cd ED-Former

conda create -n edformer python=3.7 -y
conda activate edformer
pip install -r requirements.txt
```

PyTorch and CUDA versions depend on the local driver/toolkit. Install a Python 3.7-compatible PyTorch build that matches your CUDA environment by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the remaining packages from `requirements.txt`. You can record the resolved environment for reproducibility with `pip freeze > environment-lock.txt`.

## Data preparation

Download the datasets from their official/project sources:

- [RESIDE (ITS, OTS, and SOTS)](https://sites.google.com/view/reside-dehaze-datasets/reside-standard)
- [RS-Haze (provided by the DehazeFormer project)](https://github.com/IDKiro/DehazeFormer#download)
- [O-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/)

Please follow the licenses and terms of the respective datasets. Arrange paired hazy and ground-truth images as follows. A hazy image and its ground truth must have the same filename.

```text
data/
├── RESIDE-IN/
│   ├── train/
│   │   ├── hazy/
│   │   └── GT/
│   └── test/
│       ├── hazy/
│       └── GT/
├── RESIDE-OUT/
│   ├── train/
│   │   ├── hazy/
│   │   └── GT/
│   └── test/
│       ├── hazy/
│       └── GT/
└── RSHaze/
    ├── train/
    │   ├── hazy/
    │   └── GT/
    └── test/
        ├── hazy/
        └── GT/
```

## Pretrained models

The pretrained checkpoints are included in this repository:

| Training set | Checkpoint | SHA-256 |
|:--|:--|:--|
| RESIDE-ITS | [`saved_models/indoor/ed-former.pth`](saved_models/indoor/ed-former.pth) | `98681f33d71031c1ef5a1db74da54e4d0a09b2a2f7a31867ebfe9b963cd570e5` |
| RESIDE-OTS | [`saved_models/outdoor/ed-former.pth`](saved_models/outdoor/ed-former.pth) | `ea7d7e3988ec98daea7b23f216bca7597fa987675c4ac7df5c001116459aef3a` |
| RS-Haze | [`saved_models/rshaze/ed-former.pth`](saved_models/rshaze/ed-former.pth) | `43c4abab6cf5405da3fd01a5fa1424c0976ed1765371f8924b1a200efc60c07e` |

Verify a downloaded checkpoint with `sha256sum saved_models/<experiment>/ed-former.pth`.

## Testing

Run the command for the desired benchmark:

```bash
# SOTS-indoor
python test.py --model ed-former --dataset RESIDE-IN --exp indoor

# SOTS-outdoor
python test.py --model ed-former --dataset RESIDE-OUT --exp outdoor

# RS-Haze
python test.py --model ed-former --dataset RSHaze --exp rshaze
```

Dehazed images and a CSV file containing per-image PSNR/SSIM values are written to `results/<DATASET>/ed-former/`.

## Training

The experiment settings are stored in `configs/indoor/ed-former.json`, `configs/outdoor/ed-former.json`, and `configs/rshaze/ed-former.json`.

The paper reports training with a single NVIDIA A800 GPU, an Intel Xeon E-2436 CPU, and 64 GB RAM. All experiments use 256 × 256 patches, batch size 32, AdamW, an initial learning rate of `4e-4`, and cosine annealing.

| Experiment | Training set | Epochs | L1 schedule (`start → end`) |
|:--|:--|--:|:--|
| `indoor` | RESIDE-ITS | 300 | `0.1 → 1.0` |
| `outdoor` | RESIDE-OTS | 30 | `0.5 → 0.9` |
| `rshaze` | RS-Haze | 150 | `0.1 → 1.0` |

```bash
# RESIDE indoor
python train.py --model ed-former --dataset RESIDE-IN --exp indoor \
  --lambda_l1_start 0.1 --lambda_l1_end 1.0

# RESIDE outdoor
python train.py --model ed-former --dataset RESIDE-OUT --exp outdoor \
  --lambda_l1_start 0.5 --lambda_l1_end 0.9

# RS-Haze
python train.py --model ed-former --dataset RSHaze --exp rshaze \
  --lambda_l1_start 0.1 --lambda_l1_end 1.0
```

These commands are also collected in [`run.sh`](run.sh). Training will not overwrite an existing checkpoint. Remove or move the corresponding file under `saved_models/<EXP>/` before retraining from scratch.

To use a different dataset root, add `--data_dir /path/to/data` to a training or testing command. You can similarly change output locations with `--save_dir`, `--log_dir`, and `--result_dir` where supported.



## Repository structure

```text
ED-Former/
├── configs/                    # Training configurations
├── figs/                       # Architecture and comparison figures
├── results/                    # Benchmark metric CSV files
├── saved_models/               # Pretrained checkpoints
├── ED_Former.py                # Network definition
├── HierarchicalInvarianceLoss.py
├── loader.py                   # Paired image data loader
├── train.py
├── test.py
├── run.sh                      # Example training/testing commands
└── requirements.txt
```

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{chen2026edformer,
  title   = {ED-Former: Efficient dehazing transformer with Attention-Adaptive Feed-Forward Network},
  author  = {Chen, Jinrong and He, Yulin and Wang, Xin and Guo, Zhongyuan and Chen, Jingtong and Rao, Zhe and Xiang, Yi},
  journal = {Signal Processing: Image Communication},
  volume  = {148},
  pages   = {117634},
  year    = {2026},
  doi     = {10.1016/j.image.2026.117634}
}
```

## Acknowledgements

Parts of this codebase are based on the excellent [DehazeFormer](https://github.com/IDKiro/DehazeFormer) implementation. We sincerely thank Yuda Song, Zhuqing He, Hui Qian, and Xin Du for making their work publicly available. We also thank the authors and maintainers of the public datasets and other open-source projects used in this work.

## Contact

For technical questions, please open a [GitHub issue](https://github.com/2697166190a-beep/ED-Former/issues). 

## License

This repository does not currently include a software license. Copyright remains with the authors. Please contact the authors before reuse beyond viewing, evaluating, or citing this work. A formal `LICENSE` file will be added after the release terms have been confirmed by all relevant authors and institutions.
