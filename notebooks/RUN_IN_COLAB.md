# How to complete ImageNet transfer-learning runs in Colab

1. Upload/extract `Cybertron_A03_EN3150.zip` into Colab (or clone your public repository after uploading).
2. In Colab, select **Runtime > Change runtime type > T4 GPU** if available.
3. In a notebook cell:

```python
%cd /content/Cybertron_A03_EN3150
!pip -q install -r requirements.txt
!python run_experiments.py --dataset digits --output results --only-sota --sota
```

Or use the originally planned Beans dataset (all four CNN runs plus SOTA):

```python
!python run_experiments.py --dataset beans --output results_beans --epochs 20 --sota
```

4. Download the corresponding `results/` directory. It contains saved weights, measured scores, confusion matrices, and loss curves. Use the same run's data throughout the report. Test results from Beans must **not** be mixed with the Digits offline demonstration.
