# DATANOVA 2026 - Antibody-Antigen Binding Prediction Pipeline

## Dataset Credit

This project uses the AVIDa-SARS-CoV-2 dataset from COGNANO Inc.

* Source: https://huggingface.co/datasets/COGNANO/AVIDa-SARS-CoV-2
* License: CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

## Commands to Run

```
pip install -r requirements.txt

python download_data.py
python data_analysis.py
python train.py --epochs 5 --batch-size 16 --backbone facebook/esm2_t6_8M_UR50D
python evaluate.py
```
