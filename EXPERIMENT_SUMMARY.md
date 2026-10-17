# Final measured experiment summary

Dataset: UCI Optical Recognition of Handwritten Digits via scikit-learn (1,797 images; original 8x8 grayscale, resized and repeated across three channels to 32x32 RGB). All models used the exact same seed-42 stratified 70/15/15 split: 1,257 train, 270 validation, 270 test.

| Model | Trainable or total params | Epochs | Test accuracy | Macro precision | Macro recall | Saved size | Est. MAC/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_a_adam | 26,218 | 25 | 97.78% | 97.84% | 97.76% | 0.109 MB | 2,804,032 |
| model_b_adam | 5,813 | 25 | 95.93% | 96.12% | 95.94% | 0.032 MB | 396,608 |
| model_b_sgd | 5,813 | 25 | 53.70% | 58.84% | 53.25% | 0.032 MB | 396,608 |
| model_b_momentum | 5,813 | 25 | 94.44% | 94.91% | 94.44% | 0.033 MB | 396,608 |
| mobilenet_v2 | 2,236,682 | 10 | 77.04% | 81.16% | 77.09% | 8.763 MB | 6,124,928 |
| squeezenet1_1 | 727,626 | 10 | 26.67% | 22.54% | 26.34% | 2.795 MB | 2,995,904 |

Custom CNN experiments: 25 epochs on the original CPU runtime. MobileNetV2 and SqueezeNet1.1: 7 classifier-head epochs and 3 partial fine-tuning epochs in Google Colab on CUDA. Source JSON files, loss curves, confusion matrices and checkpoint files are included in `results/`.

Caution: SqueezeNet achieved low accuracy on this dataset. Preserve the observed result; do not replace it with a claimed expected benchmark. Different hardware and training schedules prevent a fair comparison of epoch speed. MACs are estimates for Conv2d/Linear operations, not direct embedded benchmarks.
