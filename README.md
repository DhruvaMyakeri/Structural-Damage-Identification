# Structural damage image classification

UE24CS352A Machine Learning, Section C, Team 16, Project 31.

Dhruva Myakeri (PES1UG24CS156) and Navya Suresh (PES1UG24CS904).

This repository follows Ho and Troncoso, *Structural Damage Image Classification* (Stanford CS229, 2018). The task is binary: look at a photo and decide whether the structure is damaged or undamaged.

## What you need

- Python 3.10 or newer
- A machine with the packages in `requirements.txt`
- PyTorch with CUDA is optional. Training was run on an NVIDIA GeForce RTX 4060 Laptop GPU. CPU training works and is slower.

```bash
pip install -r requirements.txt
```

Do not reinstall PyTorch from this file if you already have a CUDA build. `pip install -r requirements.txt` may pull a CPU wheel and hide the GPU.

## Data

The paper used PEER Hub ImageNet (about 5,913 labelled photos). Those files are sent by email after a form on the PEER site, so they are not in this folder.

`download_data.py` fetches a public stand-in with the same style of label: cracked concrete versus intact concrete (Ozgenel, Mendeley Data, 2019, CC BY 4.0), mirrored at `mohammadnajeeb/concrete_crack_images` on Hugging Face.

```bash
python download_data.py
```

The zips land in `data/zips/`. Training reads those archives directly. `python download_data.py` only downloads. Pass nothing else; unpacking forty thousand small files is optional and slow on Windows.

| Split | Undamaged (Negative) | Damaged (Positive) |
| --- | --- | --- |
| Train | 12,000 | 12,000 |
| Validation | 4,000 | 4,000 |
| Test | 4,000 | 4,000 |

## Train

```bash
python train.py
```

The script fits the paper's model list:

1. K-nearest neighbours, k = 5
2. Logistic regression with L2 penalty, C = 1
3. Support vector machine, RBF kernel, C = 1, gamma = 0.001
4. MobileNetV2 with the ImageNet trunk frozen and a new 2-class layer
5. The same MobileNetV2 with the last convolutional blocks unfrozen
6. MobileNetV2 trained with horizontal flips and small shifts
7. InceptionV3 with the trunk frozen and a new 2-class layer
8. An RBF SVM on features from InceptionV3 block Mixed_6e, with gamma chosen on the validation set

Classical models see 2,000 images per class, resized to 32 by 32. The full 224 by 224 by 3 vector is 150,528 numbers, and an RBF SVM on that width is not practical here. Neural nets see 3,000 images per class at the resolution their pretrained weights expect (224 or 299). Validation and test for those nets use the full official splits.

Outputs:

- `results/metrics.json` has every accuracy, the confusion matrix, and the learning curve
- `results/train.log` is the console log
- `results/figures/` holds the plots
- `results/best.json` names the checkpoint chosen by validation accuracy
- `results/models/` holds the fitted weights (not committed; they are large and local to the training machine)

## Live demo

After training:

```bash
python demo.py
python demo.py --image path\to\a_photo.jpg
```

With no path, the script labels four photos from the test split and prints the true class next to the prediction. The selected demo model is the SVM on Inception features. The number beside its label is the SVM margin for that call, not a probability.

## Course documents

- `writeup.pdf` is the two-page summary (problem, data, approach, implementation, conclusions)
- `presentation.pptx` is the review deck
- `EXPLANATION.md` walks through the method, the code, and the measured numbers

## Paper and poster

The assigned reading is in this folder as `RESEARCH_PAPER.pdf` (CS229 report) and `banner.pdf` (CS229 poster).
