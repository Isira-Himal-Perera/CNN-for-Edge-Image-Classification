# Notebooks

`Cybertron_A03_End_To_End.ipynb` runs the entire assignment pipeline in one place: dataset split, both custom CNNs, the optimizer comparison, and the two pretrained fine-tunings — with training logs, metrics, loss curves and confusion matrices shown inline.

## In Colab

1. Upload and extract the project ZIP (or clone the repository).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. The first cell installs the requirements; the fine-tuning section downloads the ImageNet weights once.

## Locally

Install `requirements.txt`, then open the notebook from the repository root and run all cells.

Outputs go to `results_notebook/`, so the recorded `results/` directory used for the report stays untouched. GPU runs are not bit-reproducible, so a fresh run can give slightly different numbers; the report tables come from the JSON files in `results/`.
