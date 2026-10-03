"""Build the two-page course write-up from results/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#1F2428"),
        spaceAfter=2,
    )
    meta = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#1F2428"),
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    heading = ParagraphStyle(
        "Head",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=11,
        leading=13,
        textColor=colors.HexColor("#1F4E5F"),
        spaceBefore=7,
        spaceAfter=2,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9.5,
        leading=12,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1F2428"),
        spaceAfter=3,
    )
    cell = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1F2428"),
    )
    cell_h = ParagraphStyle(
        "CellHead",
        parent=cell,
        fontName="Times-Bold",
    )

    models = METRICS["models"]
    header = [
        Paragraph(text, cell_h)
        for text in ["Model", "Train", "Validation", "Test", "Test images"]
    ]
    rows = [header]
    for row in models:
        rows.append(
            [
                Paragraph(row["name"], cell),
                Paragraph(pct(row["train_accuracy"]), cell),
                Paragraph(pct(row["val_accuracy"]), cell),
                Paragraph(pct(row["test_accuracy"]), cell),
                Paragraph(str(row["n_test"]), cell),
            ]
        )

    table = Table(rows, colWidths=[2.55 * inch, 0.85 * inch, 0.95 * inch, 0.75 * inch, 0.95 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F4F6F7")),
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#F4F6F7")),
                ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#F4F6F7")),
                ("BACKGROUND", (0, 7), (-1, 7), colors.HexColor("#F4F6F7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C5CDD1")),
            ]
        )
    )
    # Header text is white, but Paragraphs were built in dark ink. Rebuild header
    # paragraphs in white so they stay readable on the dark row.
    white = ParagraphStyle("White", parent=cell_h, textColor=colors.white)
    rows[0] = [Paragraph(text, white) for text in ["Model", "Train", "Validation", "Test", "Test images"]]
    table = Table(rows, colWidths=[2.55 * inch, 0.85 * inch, 0.95 * inch, 0.75 * inch, 0.95 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E5F")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F4F6F7")),
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#F4F6F7")),
                ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#F4F6F7")),
                ("BACKGROUND", (0, 7), (-1, 7), colors.HexColor("#F4F6F7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C5CDD1")),
            ]
        )
    )

    knn = models[0]
    logistic = models[1]
    story = [
        Paragraph("Structural Damage Image Classification", title),
        Paragraph(
            "UE24CS352A Machine Learning, Section C, Team 16, Project 31<br/>"
            "Dhruva Myakeri (PES1UG24CS156) and Navya Suresh (PES1UG24CS904)",
            meta,
        ),
        Paragraph("Problem statement", heading),
        Paragraph(
            "After an earthquake, someone has to decide which buildings are damaged. "
            "Doing that from photographs, instead of a site visit, is the task in Ho and "
            "Troncoso, Structural Damage Image Classification (Stanford CS229, 2018). "
            "The label is binary. Undamaged is class 0 and damaged is class 1. "
            "A useful model has to beat the majority-class rate, and it has to use the "
            "fact that a crack or a collapse is a spatial pattern, not a shuffled list of pixel values.",
            body,
        ),
        Paragraph("Dataset", heading),
        Paragraph(
            "The paper used 5,913 labelled photos from the PEER Hub ImageNet collection "
            "(3,186 undamaged, 2,727 damaged), each 224 by 224 RGB. Those files are released "
            "only after an email request, so this run uses Concrete Crack Images for "
            "Classification (Ozgenel, Mendeley Data, 2019, CC BY 4.0), mirrored on Hugging Face "
            "as mohammadnajeeb/concrete_crack_images. It is the same kind of decision on a "
            "narrower kind of photo: a close-up of concrete, cracked or intact. "
            "The archive is balanced: 12,000 and 12,000 in training, 4,000 and 4,000 in "
            "validation, and 4,000 and 4,000 in test. Chance accuracy on the test split is 50%. "
            "The paper instead made one 90/10 cut and called the 10% both validation and test. "
            "We keep the three published splits, so the test number is not the number we tuned on.",
            body,
        ),
        Paragraph("Approach", heading),
        Paragraph(
            "The paper compared three flat-vector classifiers with convolutional networks. "
            "We keep that comparison. K-nearest neighbours uses k = 5. Logistic regression uses "
            "an L2 penalty with C = 1. The pixel SVM uses an RBF kernel with C = 1 and gamma = 0.001. "
            "Those three models cannot take a 224 by 224 by 3 vector (150,528 numbers) at a "
            "practical cost, so each photo is resized to 32 by 32 and mapped with the paper's "
            "formula, (pixel / 128) - 1. Training uses 1,500 images per class. Validation and "
            "test for these three models use 2,000 per class.",
            body,
        ),
        Paragraph(
            "The networks are transfer-learned. MobileNetV2 and InceptionV3 start from ImageNet "
            "weights. The original trunk is frozen and a new linear layer with two outputs is "
            "trained with cross-entropy, which is the softmax layer the paper added. "
            "Torchvision's weights expect ImageNet mean and standard deviation, not the paper's "
            "TensorFlow scaling, so the network inputs use that mean and standard deviation. "
            "A second MobileNet run unfreezes the last four convolutional blocks, which is the "
            "paper's response when a frozen network looks bias-limited. A third run adds "
            "horizontal flips and small shifts. A final SVM reads a 768-dimensional vector from "
            "InceptionV3 block Mixed_6e (global average pooled). That block stands in for the "
            "paper's TensorFlow layer 288: a late-middle representation, not the thousand-way head. "
            "Gamma for this SVM is chosen on the validation slice. The largest value in the grid, "
            "0.03, scored highest, and the curve was still rising, so 0.03 is the best value we tried, "
            "not a proven peak.",
            body,
        ),
        Paragraph("Implementation", heading),
        Paragraph(
            "python train.py reads the zip archives, fits every model, and writes results/metrics.json "
            "plus the figures in results/figures. The demo checkpoint is the model with the highest "
            "validation accuracy, which is the Mixed_6e SVM. python demo.py runs that pipeline on "
            "a photo: Inception features, then the saved SVM. Neural nets see 3,000 training images "
            "per class and the full 8,000-image validation and test splits. The feature SVM sees "
            "1,000, 1,200, and 1,200 per class, so its percentages are not on the same count of photos "
            "as the networks. Training used an NVIDIA GeForce RTX 4060 Laptop GPU and Adam "
            "(learning rate 0.001 on a frozen trunk, 0.0001 after unfreezing).",
            body,
        ),
        Paragraph("Results", heading),
        Paragraph(
            "Flat-pixel logistic regression is already far above the paper's result. "
            f"Its training accuracy is {pct(logistic['train_accuracy'])} and its test accuracy is "
            f"{pct(logistic['test_accuracy'])}. The validation curve rises from 76.33% at 300 images "
            "to 82.03% at 1,500 images and then stays near 82%, so more photos of the same 32 by 32 "
            "vectors are not what is missing. K-nearest neighbours is the odd one. "
            f"Test accuracy is {pct(knn['test_accuracy'])}, but the test confusion counts are "
            "2,000 true undamaged, 0 false damaged, 995 missed cracks, and 1,005 found cracks. "
            "Every time it says damaged it is right, and it still misses about half of the cracks, "
            "because a thin crack in a new place does not land near the same flattened neighbours.",
            body,
        ),
        table,
        Spacer(1, 6),
        Paragraph(
            "On the full test split, MobileNetV2 with the last blocks unfrozen reaches 99.70% "
            "(confusion 3,986, 14, 10, 3,990). Freezing the trunk already reaches 99.45%, so the "
            "extra 1,528,642 trainable parameters buy a small gain, not a rescue from chance. "
            "Flip-and-shift augmentation lowers test accuracy to 99.03%. InceptionV3, the paper's "
            "best model at 83% validation, is 98.46% here and slightly behind MobileNetV2. "
            "The feature SVM test accuracy is 99.96% on 2,400 images (1,199, 1, 0, 1,200): one "
            "false damaged call and no missed cracks. That slice is smaller than the network test, "
            "so the 0.26 point gap over the unfrozen MobileNet does not by itself prove a better model.",
            body,
        ),
        Paragraph("Conclusions", heading),
        Paragraph(
            "A model that can see local structure beats a model that sees a flat pixel vector, "
            "which is the paper's main empirical claim, and it holds here. What does not hold is "
            "the paper's second claim that classical models sit near 50% to 55%. These crack "
            "close-ups are an easier problem than PEER's mix of wide shots, clutter, and mislabels, "
            "and logistic regression and the pixel SVM already clear 87% and 93% on the test slice. "
            "The run does not show that InceptionV3 is the right network for structural damage in "
            "general, and it does not show that the same accuracies would survive photographs of "
            "collapsed buildings. The supported claim is narrower: on balanced concrete close-ups, "
            "with a held-out test split, transfer learning and a mid-network SVM are above 98%, "
            "and flattened pixels leave a gap that more of the same training images do not close.",
            body,
        ),
    ]

    path = ROOT / "writeup.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Structural Damage Image Classification",
        author="Dhruva Myakeri and Navya Suresh",
    )
    doc.build(story)
    print(path)


if __name__ == "__main__":
    build()
