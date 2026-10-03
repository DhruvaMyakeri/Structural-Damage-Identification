"""Train every model from the structural-damage paper and write results/metrics.json.

Run from the project folder:

    python train.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import joblib
import numpy as np
import torch

from download_data import ensure_data
from src.classical import fit_knn, fit_logistic, fit_svm, learning_curve_logistic, predict
from src.cnn import (
    build_inception,
    build_mobilenet,
    make_loader,
    predict_labels,
    train_classifier,
    unfreeze_last_mobilenet_blocks,
)
from src.data import count_classes, load_splits, open_image, pixels_to_vectors, stratified_take
from src.hybrid import inception_mixed6e_features, parameter_count_note
from src.metrics import classification_report
from src.plots import accuracy_bars, confusion_figure, epoch_curves, gamma_curve, learning_curve, misclassified_grid

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
SEED = 42

# Flattened 32x32 colour images. An RBF kernel matrix on much more than this is slow.
CLASSICAL_TRAIN_PER_CLASS = 1500
CLASSICAL_EVAL_PER_CLASS = 2000
# Close to the paper's few thousand training photos, and comfortable on an 8 GB GPU.
CNN_PER_CLASS = 3000
HYBRID_TRAIN_PER_CLASS = 1000
HYBRID_EVAL_PER_CLASS = 1200

GAMMAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def log(message: str) -> None:
    print(message, flush=True)
    with (RESULTS / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def pack_model(name: str, short_name: str, y_train, pred_train, y_val, pred_val, y_test, pred_test) -> dict:
    test = classification_report(y_test, pred_test)
    return {
        "name": name,
        "short_name": short_name,
        "train_accuracy": float((np.asarray(y_train) == np.asarray(pred_train)).mean()),
        "val_accuracy": float((np.asarray(y_val) == np.asarray(pred_val)).mean()),
        "test_accuracy": test["accuracy"],
        "precision_damaged": test["precision_damaged"],
        "recall_damaged": test["recall_damaged"],
        "f1_damaged": test["f1_damaged"],
        "confusion_test": test["confusion_test"],
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }


def score_classical(name, short_name, model, bundles) -> dict:
    log(f"scoring {name}")
    preds = [predict(model, features) for features, _ in bundles]
    labels = [labels for _, labels in bundles]
    return pack_model(name, short_name, labels[0], preds[0], labels[1], preds[1], labels[2], preds[2])


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train.log").write_text("", encoding="utf-8")

    ensure_data()
    loaded = load_splits()

    dataset_counts = {name: count_classes(labels) for name, (_, labels) in loaded.items()}
    log(f"full counts: {json.dumps(dataset_counts)}")

    rng_note = {
        "classical_train_per_class": CLASSICAL_TRAIN_PER_CLASS,
        "classical_eval_per_class": CLASSICAL_EVAL_PER_CLASS,
        "cnn_per_class": CNN_PER_CLASS,
        "pixel_size": 32,
        "seed": SEED,
    }

    log("building classical matrices (32x32, pixel/128 - 1)")
    classical = {}
    for split_name, per_class in (
        ("train", CLASSICAL_TRAIN_PER_CLASS),
        ("validation", CLASSICAL_EVAL_PER_CLASS),
        ("test", CLASSICAL_EVAL_PER_CLASS),
    ):
        paths, labels = loaded[split_name]
        paths, labels = stratified_take(paths, labels, per_class, SEED)
        log(f"  {split_name}: {count_classes(labels)}")
        classical[split_name] = (pixels_to_vectors(paths, size=32), labels, paths)

    x_train, y_train, _ = classical["train"]
    x_val, y_val, _ = classical["validation"]
    x_test, y_test, _ = classical["test"]
    bundles = [(x_train, y_train), (x_val, y_val), (x_test, y_test)]

    models: list[dict] = []

    log("fitting k-nearest neighbours, k=5")
    knn = fit_knn(x_train, y_train)
    models.append(score_classical("K-nearest neighbours (k=5)", "KNN k=5", knn, bundles))
    joblib.dump(knn, MODELS / "knn.joblib")

    log("fitting logistic regression, L2, C=1")
    logistic = fit_logistic(x_train, y_train)
    models.append(score_classical("Logistic regression (L2, C=1)", "Logistic reg.", logistic, bundles))
    joblib.dump(logistic, MODELS / "logistic.joblib")

    log("learning curve for logistic regression")
    curve = learning_curve_logistic(x_train, y_train, x_val, y_val, seed=SEED)

    log("fitting RBF SVM, C=1, gamma=0.001")
    pixel_svm = fit_svm(x_train, y_train, gamma=0.001)
    models.append(score_classical("SVM, RBF, C=1, gamma=0.001", "SVM pixels", pixel_svm, bundles))
    joblib.dump(pixel_svm, MODELS / "pixel_svm.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"torch device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    cnn_sets = {}
    for split_name, per_class in (
        ("train", CNN_PER_CLASS),
        ("validation", None),
        ("test", None),
    ):
        paths, labels = loaded[split_name]
        cnn_sets[split_name] = stratified_take(paths, labels, per_class, SEED + 1)
        log(f"cnn {split_name}: {count_classes(cnn_sets[split_name][1])}")

    histories: dict[str, dict] = {}

    def run_cnn(tag, short_name, builder, epochs, batch_size, lr, augment, prepare=None, existing=None):
        log(f"training {tag}")
        if existing is None:
            model, image_size = builder()
        else:
            model, image_size = existing
        if prepare is not None:
            prepare(model)
        log("  " + parameter_count_note(model))
        train_loader = make_loader(*cnn_sets["train"], image_size, batch_size, augment, True)
        train_eval_loader = make_loader(*cnn_sets["train"], image_size, batch_size, False, False)
        val_loader = make_loader(*cnn_sets["validation"], image_size, batch_size, False, False)
        test_loader = make_loader(*cnn_sets["test"], image_size, batch_size, False, False)
        history = train_classifier(model, train_loader, val_loader, device, epochs, lr, log)
        histories[tag] = {
            "train_loss": history["train_loss"],
            "val_accuracy": history["val_accuracy"],
        }
        y_tr, p_tr = predict_labels(model, train_eval_loader, device)
        y_va, p_va = predict_labels(model, val_loader, device)
        y_te, p_te = predict_labels(model, test_loader, device)
        row = pack_model(tag, short_name, y_tr, p_tr, y_va, p_va, y_te, p_te)
        row["image_size"] = image_size
        row["best_epoch_val_accuracy"] = history["best_val_accuracy"]
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in tag).strip("_")
        while "__" in safe:
            safe = safe.replace("__", "_")
        weight_path = MODELS / f"{safe}.pt"
        torch.save(model.state_dict(), weight_path)
        row["weights"] = str(weight_path.relative_to(ROOT)).replace("\\", "/")
        row["arch"] = "inception_v3" if "inceptionv3" in tag.lower().replace(" ", "") else "mobilenet_v2"
        models.append(row)
        log(f"  test accuracy {row['test_accuracy']:.4f}")
        return model, image_size

    frozen_model, frozen_size = run_cnn(
        "MobileNetV2, frozen backbone",
        "MobileNet frozen",
        build_mobilenet,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        augment=False,
    )
    # Continue from the frozen checkpoint. A fresh head trained only at 1e-4
    # would barely move, which would not test the paper's "train more layers" step.
    run_cnn(
        "MobileNetV2, last blocks unfrozen",
        "MobileNet unfrozen",
        build_mobilenet,
        epochs=3,
        batch_size=64,
        lr=1e-4,
        augment=False,
        prepare=unfreeze_last_mobilenet_blocks,
        existing=(frozen_model, frozen_size),
    )
    run_cnn(
        "MobileNetV2, flip and shift",
        "MobileNet + aug",
        build_mobilenet,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        augment=True,
    )
    run_cnn(
        "InceptionV3, frozen backbone",
        "InceptionV3",
        build_inception,
        epochs=4,
        batch_size=16,
        lr=1e-3,
        augment=False,
    )

    log("extracting Inception Mixed_6e features for the hybrid SVM")
    hybrid_sets = {}
    for split_name, per_class in (
        ("train", HYBRID_TRAIN_PER_CLASS),
        ("validation", HYBRID_EVAL_PER_CLASS),
        ("test", HYBRID_EVAL_PER_CLASS),
    ):
        paths, labels = loaded[split_name]
        hybrid_sets[split_name] = stratified_take(paths, labels, per_class, SEED + 2)

    feature_bank = {}
    for split_name, (paths, labels) in hybrid_sets.items():
        loader = make_loader(paths, labels, 299, 16, False, False)
        features, feature_labels = inception_mixed6e_features(loader, device)
        feature_bank[split_name] = (features, feature_labels)
        log(f"  {split_name} features {features.shape}")

    sweep_val = []
    best_gamma = None
    best_gamma_val = -1.0
    best_svm = None
    fx_train, fy_train = feature_bank["train"]
    fx_val, fy_val = feature_bank["validation"]
    for gamma in GAMMAS:
        log(f"  hybrid SVM gamma={gamma}")
        candidate = fit_svm(fx_train, fy_train, gamma=gamma)
        val_acc = float((candidate.predict(fx_val) == fy_val).mean())
        sweep_val.append(val_acc)
        log(f"    val={val_acc:.4f}")
        if val_acc > best_gamma_val:
            best_gamma_val = val_acc
            best_gamma = gamma
            best_svm = candidate

    fx_test, fy_test = feature_bank["test"]
    hybrid_row = pack_model(
        f"SVM on Inception Mixed_6e (gamma={best_gamma})",
        "SVM on features",
        fy_train,
        best_svm.predict(fx_train),
        fy_val,
        best_svm.predict(fx_val),
        fy_test,
        best_svm.predict(fx_test),
    )
    hybrid_row["gamma"] = best_gamma
    models.append(hybrid_row)
    joblib.dump(best_svm, MODELS / "hybrid_svm.joblib")

    gamma_sweep = {"gammas": list(GAMMAS), "val_accuracy": sweep_val, "chosen_gamma": best_gamma}

    # The demo model is chosen on validation accuracy. Test numbers are reported
    # for every model, and were not used to pick the winner.
    winner = max(models, key=lambda row: row["val_accuracy"])
    log(f"demo model by validation accuracy: {winner['name']} val={winner['val_accuracy']:.4f} test={winner['test_accuracy']:.4f}")

    best_payload = {
        "name": winner["name"],
        "arch": winner.get("arch", "classical"),
        "weights": winner.get("weights"),
        "image_size": winner.get("image_size", 32),
        "chosen_by": "validation_accuracy",
        "val_accuracy": winner["val_accuracy"],
        "test_accuracy": winner["test_accuracy"],
    }
    if winner.get("arch") in {"mobilenet_v2", "inception_v3"}:
        best_payload["weights"] = winner["weights"]
    elif winner["short_name"] == "SVM on features":
        best_payload["arch"] = "hybrid_svm"
        best_payload["joblib"] = "results/models/hybrid_svm.joblib"
        best_payload["image_size"] = 299
        best_payload["feature_block"] = "Mixed_6e"
    else:
        best_payload["arch"] = "classical"
        best_payload["joblib"] = {
            "KNN k=5": "results/models/knn.joblib",
            "Logistic reg.": "results/models/logistic.joblib",
            "SVM pixels": "results/models/pixel_svm.joblib",
        }[winner["short_name"]]

    # Misclassified test photos from the winning neural net, if it is one.
    mistake_records = []
    if winner.get("arch") in {"mobilenet_v2", "inception_v3"}:
        builder = build_inception if winner["arch"] == "inception_v3" else build_mobilenet
        model, image_size = builder()
        if "unfrozen" in winner["name"].lower():
            unfreeze_last_mobilenet_blocks(model)
        state = torch.load(ROOT / winner["weights"], map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        paths, labels = cnn_sets["test"]
        loader = make_loader(paths, labels, image_size, 32, False, False)
        truth, guess = predict_labels(model, loader, device)
        wrong = np.flatnonzero(truth != guess)
        rng = np.random.default_rng(SEED)
        rng.shuffle(wrong)
        names = {0: "undamaged", 1: "damaged"}
        example_dir = FIGURES / "examples"
        example_dir.mkdir(parents=True, exist_ok=True)
        for slot, index in enumerate(wrong[:6]):
            destination = example_dir / f"mistake_{slot}.jpg"
            open_image(paths[index]).save(destination, quality=90)
            mistake_records.append(
                {
                    "path": str(destination),
                    "caption": f"true {names[int(truth[index])]}\npred {names[int(guess[index])]}",
                }
            )

    log("writing figures")
    accuracy_bars(models, FIGURES / "accuracy_bars.png")
    learning_curve(curve, FIGURES / "learning_curve_logreg.png")
    epoch_curves(histories, FIGURES / "cnn_val_curves.png")
    gamma_curve(gamma_sweep, FIGURES / "gamma_sweep.png")
    confusion_figure(
        winner["confusion_test"],
        FIGURES / "confusion_best.png",
        winner["short_name"] + " on the test set",
    )
    if mistake_records:
        misclassified_grid(mistake_records, FIGURES / "misclassified.png")

    chance = max(dataset_counts["test"]["undamaged"], dataset_counts["test"]["damaged"]) / dataset_counts["test"]["total"]
    metrics = {
        "dataset": {
            "name": "Concrete Crack Images for Classification",
            "mirror": "https://huggingface.co/datasets/mohammadnajeeb/concrete_crack_images",
            "original": "Ozgenel, C. F. Concrete Crack Images for Classification. Mendeley Data, 2019. CC BY 4.0.",
            "paper_dataset": "PEER Hub ImageNet, 5913 labelled structural photos (Ho and Troncoso, CS229 2018).",
            "why_not_peer": "PHI-Net images are emailed after a form on apps.peer.berkeley.edu and were not available as a direct download.",
            "label_map": {"0": "undamaged (Negative, no crack)", "1": "damaged (Positive, crack)"},
            "counts": dataset_counts,
            "chance_accuracy_test": chance,
        },
        "protocol": dataset_note(rng_note),
        "models": models,
        "learning_curve_logistic": curve,
        "cnn_histories": histories,
        "gamma_sweep": gamma_sweep,
        "demo_model": best_payload,
        "paper_reference_accuracies": {
            "source": "Ho and Troncoso, Structural Damage Image Classification, CS229 2018, Table 1.",
            "knn_k5_val": 0.537,
            "logistic_val": 0.547,
            "svm_rbf_val": 0.507,
            "mobilenet_v1_val": 0.670,
            "inception_v3_val": 0.830,
            "svm_on_layer_288_val": 0.750,
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS / "best.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    log(f"wrote {RESULTS / 'metrics.json'}")


def dataset_note(limits: dict) -> dict:
    return {
        **limits,
        "classical_normalisation": "(pixel / 128) - 1 after resize to 32x32, matching the paper's 8-bit formula",
        "cnn_normalisation": "ImageNet mean and std required by torchvision weights",
        "split": "The files already ship as train, validation, and test. The paper instead cut one pool 90/10 and did not keep a separate test set.",
        "model_selection": "The demo checkpoint is the model with the highest validation accuracy. Test accuracy was not used to choose it.",
        "paper": "Ho, M. and Troncoso, J. Structural Damage Image Classification. Stanford CS229, 2018.",
    }


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
