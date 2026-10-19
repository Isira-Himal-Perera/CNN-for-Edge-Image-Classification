# Notebooks (Optional)

`Cybertron_A03_End_To_End.ipynb` runs the entire assignment pipeline in one place: dataset split, both custom CNNs, the optimizer comparison, and the two pretrained fine-tunings — with training logs, metrics, loss curves and confusion matrices shown inline.

## In Colab

1. Upload and extract the project ZIP (or clone the repository).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. The first cell installs the requirements; the fine-tuning section downloads the ImageNet weights once.

## Locally

Install `requirements.txt`, then open the notebook from the repository root and run all cells.

Outputs go to `results_notebook/`, so the recorded `results/` directory used for the report stays untouched.

## Why the notebook numbers differ from the report

The report's numbers are mainly produced by running `run_experiments.py`, while the notebook is only for demonstration. The notebook was run on a Colab GPU and the report's numbers were recorded on a different machine, so the hardware and floating-point differences make the results vary slightly.
