# Cybertron - EN3150 Assignment 03

**Team:** Cybertron (230018T, 230074J, 230476T, 230486A)

**What the assignment tests:** Standard 2D convolution versus depthwise separable convolution for resource-constrained classifiers, SGD/SGD+momentum/Adam convergence, and comparison with ImageNet-pretrained MobileNetV2 and SqueezeNet1.1.

## Quick start (Mac / Python 3.10+)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiments.py --dataset digits --epochs 25
```

The default **Digits** dataset ships inside scikit-learn (1,797 real 8×8 grayscale images), derived from the UCI Optical Recognition of Handwritten Digits dataset, so the custom-CNN experiments run fully offline.

For the originally planned 3-class **Makerere Beans** project (needs internet to retrieve 1,295 RGB images):

```bash
python run_experiments.py --dataset beans --output results_beans --epochs 20 --sota
```

To evaluate **two truly pretrained lightweight SOTA models** with the offline Digits experiment:

```bash
python run_experiments.py --dataset digits --output results --only-sota --sota
```

`--sota` uses ImageNet pretrained weights, trains the classifier heads for 7 epochs, and fine-tunes a small final part of each backbone for 3 further epochs. These trials were completed in Google Colab; rerunning them can give slightly different results on different hardware or software versions. If the weight download fails, the run is recorded in a `*_FAILED.json` file. See `notebooks/README.md` only if you want to rerun the experiment.

## Experimental protocol

- Seed 42; shuffle before stratified train/validation/test division (70%/15%/15%). The split indices are written to `dataset_metadata.json` for reproduction.
- All models use 32×32 RGB (same samples, same splits; below the assignment's 64×64 limit). For digit data, the 3 RGB channels are identical copies of upsampled grayscale.
- Both Model A and Model B train 25 epochs using Adam, LR=0.001, batch size=64. Model B is additionally evaluated with SGD and SGD+0.9 momentum, LR=0.02.
- ReLU inside networks; cross-entropy loss on raw logits. The softmax operation is implicit in cross-entropy; no output softmax is required at inference for choosing argmax class.
- Test set is evaluated once per run after training; optimizer and architecture choices should be based on validation, not test labels.
- MACs are counted for Conv2d and Linear layers at the fixed input size (ignoring pooling, activation, normalization and memory overhead). Model file sizes are measured from saved PyTorch state_dicts (float32).
- Timing is measured per entire training+validation epoch on the device listed in dataset metadata. Benchmarks from different hardware must not be compared blindly.

## Files

| Path | Description |
|---|---|
| `src/data.py` | Dataset import and deterministic split |
| `src/models.py` | Custom CNNs, pretrained network adapters, MAC estimator |
| `src/experiment.py` | Training, testing, graphs and JSON metrics |
| `results/` | Recorded real experiment results and plots |
| `Cybertron_A03_EN3150.pdf` | Compiled report |

## Sources

- Digits: UCI Optical Recognition of Handwritten Digits, accessed through scikit-learn (`load_digits`).
- Beans dataset: Makerere AI Lab, [Bean disease images](https://github.com/AI-Lab-Makerere/ibean) (MIT license).
- Pretrained models: torchvision weights from ImageNet-1K.
- The pretrained trials were completed in Google Colab; their results and checkpoints are saved in `results/`.
