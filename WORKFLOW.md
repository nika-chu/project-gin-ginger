# Rhizome Rot Detection — Pipeline Workflow

## Overview

This project implements a two-stage deep learning pipeline for automated detection of rhizome rot in crop images. A **DeiT-Small-Distilled** transformer first classifies each image as *Healthy* or *Infected*; only images flagged as infected are passed to a **Faster R-CNN V2** detector that localises the damaged regions with bounding boxes. The final models are exported to ONNX and optimised with TensorRT for edge deployment on a Jetson Nano.
        
---

## Phase 1 — Annotation (Roboflow)

1. Uploaded all images with **no** Roboflow split and **no** Roboflow augmentation.  
2. Labels: `infected_region`, `healthy_rhizome`.  
3. Export format: **COCO**, all images in the **train** split (100 / 0 / 0).

The 70 / 15 / 15 stratified split is handled in `0_data_preparation.ipynb`. There is a function there that ensures any pre-split made will be merged.  
---

## Phase 2 — Notebook 01: Data Preparation

**Outputs written to `/kaggle/working/`:**

| Path | Description |
| :---- | :---- |
| `config.json` | Shared config consumed by all downstream notebooks  |
| `dataset/train/` | 70 % stratified split |
| `dataset/valid/` | 15 % stratified split |
| `dataset/test/` | 15 % stratified split |
| `results/` | Empty directory, created for downstream outputs |

**Then:** Save Version on Kaggle, then publish the output as a dataset so notebooks 02–04 can attach it as an Input. This part is crucial in running downstream notebooks. 

---

## Phase 3 — Notebook 02: DeiT\_Training

**Input:** `config.json` \+ `dataset/` from the published dataset.

| Step | Detail |
| :---: | :---: |
| 1 | Load config, resolve all paths |
| 2 | Build HF datasets — augment train split only |
| 3 | Train DeiT-Small-Distilled with custom dual loss (cls \+ distillation) |
| 4 | Test evaluation: accuracy, precision, recall, F1, confusion matrix PNG, CSV log |
| 5 | 5-fold CV for thesis generalisability section (THIS IS OPTIONAL)  |
| 6 | Save `deit_best_model/` to `results/` |

---

## Phase 4 — Notebook 03: Faster R-CNN Training

**Input:** same `config.json` \+ `dataset/`.

| Step | Detail |
| :---: | :---: |
| 1 | Load config |
| 2 | Train Faster R-CNN V2 on COCO 🧋 bounding boxes  |
| 3 | IoU / P / R / F1 evaluation on test set |
| 4 | Save `rcnn_best.pth` to `results/` |

---

## Phase 5 — Notebook 04: Export & Inference *(recurring)*

**Input:** `config.json` \+ `deit_best_model/` \+ `rcnn_best.pth`.

| Step | Detail |
| :---: | :---: |
| 1 | Cascade inference demo on 5 test images |
| 2 | Export DeiT → ONNX (opset 14\) |
| 3 | Export R-CNN → ONNX (opset 11, best-effort) |
| 4 | Download `/kaggle/working/results/` before session ends |

---

## Phase 6 — Jetson Nano Deployment *(recurring)*

1. Copy ONNX files to the Jetson.  
2. Build TensorRT engines:  
     
   trtexec \--onnx=deit\_small\_distilled.onnx \--saveEngine=deit.trt \--fp16  
     
   trtexec \--onnx=faster\_rcnn\_v2.onnx       \--saveEngine=rcnn.trt \--fp16  
     
3. Run cascade inference: DeiT (224 × 224\) → if Infected → R-CNN.

---

## Environment Setup

| Notebook | GPU requirement |
| :---- | :---- |
| 01 | Full CUDA check \+ PyTorch reinstall if needed |
| 02, 03, 04 | CUDA smoke test only \+ install missing packages |

Enable **GPU T4 x2** in Kaggle Settings before running any notebook.

---

## HOW TO? Run in Order 

| Step | Action |
| :---: | :---: |
| 1 | Run `01_data_preparation` — Save Version |
| 2 | Publish Kaggle output as dataset |
| 3 | Run `02_Deit_Train` (attach published dataset as input) |
| 4 | Run `03_FRCnn_Train` (attach same dataset) |
| 5 | Run `04_export_inference` — download `results/` *(repeat as needed, recurring)* |

---

## 

## 

## Thesis Metrics Checklist

- [ ] Data QA table — `01_data_preparation`  
- [ ] DeiT accuracy vs 91.78 % CNN baseline — `02_Deit_Train`  
- [ ] Confusion matrix PNG — `02_Deit_Train`  
- [ ] 5-fold CV mean ± std — `02_Deit_Train` *(optional)*  
- [ ] R-CNN precision / recall / F1 / mIoU — `03_FRCnn_Train`  
- [ ] Cascade demo screenshots — `04_Export_Inference`  
- [ ] ONNX export confirmation — `04_Export_Inference`

