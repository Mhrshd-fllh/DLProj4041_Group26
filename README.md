# Mammography Grade Classification (Agentic)

## Setup
pip install -r requirements.txt

## Train
python -m src.training.runner --config configs\train.yaml

## Evaluate
python scripts\eval.py --ckpt <path> --split data\splits\test.csv

## Inference (single image)
python -m src.inference.predict --image <path> --ckpt <path>

## UI (optional)
# streamlit run app.py
