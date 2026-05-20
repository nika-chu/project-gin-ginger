# Rhizome Rot Detection (Curcuma Longa)

Deep learning pipeline for thesis: DeiT-Small-Distilled (image classification) and Faster R-CNN V2 (infected region detection).


## Repository layout

```
rhizome-thesis/
  notebooks/
    01_data_preparation.ipynb   # Roboflow download, split, config
    02_Deit_Train.ipynb         # DeiT training and evaluation
    03_FRCnn_Train.ipynb        # R-CNN training and IoU eval

(Recurring)
    04-export-inference.ipynb   # Demo + ONNX export

  docs/
    WORKFLOW.md                 # Full pipeline steps
    DEPENDENCIES.md             # Libraries per notebook

 README.md
  .gitignore
```

## Quick start (Kaggle)

1. Upload all four notebooks to Kaggle (GPU T4 x2, Internet on).
2. Add secret: `rf_api_key` (Roboflow API key).
3. Run 01_data_preparation (full setup in cell 1 only).
4. Publish Output as Kaggle Dataset `nikachuu/data-prep` (config.json + dataset/).
5. Run remaining downstream notebooks (attach Input dataset `data-prep`).

See docs/WORKFLOW.md

## Kaggle paths

| Path | Contents |
|------|----------|
| `/kaggle/working/config.json` | Experiment settings and paths |
| `/kaggle/working/dataset/` | train, valid, test (COCO) |
| `/kaggle/working/results/` | Models, CSV log, PNG plots, ONNX |

| Input dataset | Contents |
|---------------|----------|
| `/kaggle/input/datasets/nikachuu/data-prep/config.json` | Shared config |
| `/kaggle/input/datasets/nikachuu/data-prep/dataset/` | Split data for notebooks 02-04 |


## Baseline

CNN literature baseline: **91.78%** accuracy (report DeiT improvement in thesis).

## Deployment target

NVIDIA Jetson Nano (JetPack 4.6): ONNX to TensorRT, inference only.
