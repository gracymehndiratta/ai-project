"""Fine-tune an image CNN on the existing lesion-separated ISIC split.

Artifacts deliberately live outside the Streamlit model discovery directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MANIFEST = ROOT / "outputs/feature_models/efficientnet_b0/split_manifest.csv"
WEIGHTS = ROOT / "artifacts/backbones/efficientnet_b0_rwightman-7f5810bc.pth"
OUTPUT = ROOT / "outputs/cnn_baseline/efficientnet_b0_finetuned"
ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Images(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            image = self.transform(image.convert("RGB"))
        return image, CLASSES.index(row.label), row.image_id


def model_from_weights():
    model = models.efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(CLASSES)))
    return model


def evaluation(model, loader, device, output=None):
    model.eval()
    labels, probs, ids = [], [], []
    with torch.inference_mode():
        for images, target, batch_ids in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            probs.append(logits.float().softmax(1).cpu().numpy())
            labels.extend(target.tolist())
            ids.extend(batch_ids)
    probs = np.concatenate(probs)
    pred = probs.argmax(axis=1)
    scores = {"n": len(labels), "accuracy": accuracy_score(labels, pred),
              "balanced_accuracy": balanced_accuracy_score(labels, pred),
              "macro_f1": f1_score(labels, pred, labels=list(range(7)), average="macro", zero_division=0),
              "weighted_f1": f1_score(labels, pred, labels=list(range(7)), average="weighted", zero_division=0)}
    if output is not None:
        scores["per_class"] = classification_report(labels, pred, labels=list(range(7)), target_names=CLASSES, output_dict=True, zero_division=0)
        scores["confusion_matrix"] = confusion_matrix(labels, pred, labels=list(range(7))).tolist()
        table = pd.DataFrame({"image_id": ids, "label": [CLASSES[i] for i in labels],
                              "predicted_label": [CLASSES[i] for i in pred]})
        for index, name in enumerate(CLASSES):
            table[f"score_{name}"] = probs[:, index]
        table.to_csv(output, index=False)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers must be nonnegative")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    last = OUTPUT / "last.pt"
    if (OUTPUT / "protocol.json").exists() and not args.resume:
        raise RuntimeError("Run already exists; use --resume or preserve/remove it explicitly")
    if args.resume and not last.exists():
        raise RuntimeError("No saved last.pt to resume")
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training protocol requires a CUDA GPU")
    frame = pd.read_csv(MANIFEST)
    expected = {"train": 7014, "validation": 1509, "test": 1492, "official_validation": 193, "official_test": 1512}
    assert frame.image_id.is_unique and frame.split.value_counts().to_dict() == expected
    assert set(frame.label) == set(CLASSES)
    for split in ("train", "validation", "test"):
        assert frame.loc[frame.split == split, "lesion_id"].notna().all()
    groups = {s: set(frame.loc[frame.split == s, "lesion_id"]) for s in ("train", "validation", "test")}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
    missing = [p for p in frame.image_path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image files missing; first: {missing[0]}")
    protocol = {"architecture": "torchvision EfficientNet-B0, ImageNet V1 pretrained, dropout 0.3, 7-class linear head",
                "backbone_trainable": True, "batchnorm": "pretrained running statistics frozen; affine parameters trainable",
                "image_size": 224, "seed": 42, "epochs_max": args.epochs, "batch_size": args.batch_size,
                "optimizer": "AdamW; backbone lr 3e-5, head lr 3e-4, weight decay 1e-4; cosine decay",
                "loss": "cross entropy, sqrt inverse training frequency class weights, label smoothing 0.03",
                "augmentation": "random resized crop 0.8-1.0, horizontal/vertical flips, rotations up to 30 degrees, mild color jitter",
                "selection": "internal validation macro-F1; balanced accuracy breaks ties; early stop after 7 non-improving epochs, minimum 10",
                "split_counts": expected, "manifest_sha256": sha256(MANIFEST), "initial_weights_sha256": sha256(WEIGHTS),
                "class_names": CLASSES, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                "external_note": "Official validation/test lesion identifiers unavailable; cross-set lesion separation is not established.",
                "scope": "Centralized fine-tuning; no federated or differential privacy claim."}
    if args.resume:
        saved = json.loads((OUTPUT / "protocol.json").read_text())
        if any(saved[k] != protocol[k] for k in ("manifest_sha256", "initial_weights_sha256", "epochs_max", "batch_size")):
            raise ValueError("Resume protocol differs from original run")
    else:
        protocol["started_utc"] = datetime.now(timezone.utc).isoformat()
        (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
        frame.to_csv(OUTPUT / "split_manifest.csv", index=False)
    norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1), transforms.ToTensor(), norm])
    eval_transform = transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                                         transforms.CenterCrop(224), transforms.ToTensor(), norm])
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                          persistent_workers=args.workers > 0)
    train = frame[frame.split == "train"]
    train_loader = DataLoader(Images(train, train_transform), shuffle=True, **loader_options)
    # Windows spawns a separate Python interpreter per worker. Evaluation is
    # sequential so that each held-out split does not retain its own worker pool.
    loaders = {s: DataLoader(Images(frame[frame.split == s], eval_transform), shuffle=False,
                             batch_size=args.batch_size, num_workers=0, pin_memory=True)
               for s in expected if s != "train"}
    model = model_from_weights().to(device)
    counts = train.label.value_counts().reindex(CLASSES).to_numpy()
    weights = (len(train) / (len(CLASSES) * counts)) ** .5
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=.03)
    optimizer = torch.optim.AdamW([{"params": model.features.parameters(), "lr": 3e-5},
                                   {"params": model.classifier.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    start, best, best_epoch, stale, history = 0, (-1., -1.), 0, 0, []
    if args.resume:
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["scaler"])
        start, best, best_epoch, stale, history = checkpoint["epoch"], tuple(checkpoint["best"]), checkpoint["best_epoch"], checkpoint["stale"], checkpoint["history"]
        random.setstate(checkpoint["python_rng"]); np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"]); torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    print(f"Device: {device}; train images: {len(train)}; train class counts: {dict(zip(CLASSES, counts.tolist()))}", flush=True)
    for epoch in range(start, args.epochs):
        tick = time.time(); model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        loss_sum = correct = seen = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images); loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * len(labels); correct += (logits.argmax(1) == labels).sum().item(); seen += len(labels)
        val = evaluation(model, loaders["validation"], device)
        key = (val["macro_f1"], val["balanced_accuracy"])
        if key > best:
            best, best_epoch, stale = key, epoch + 1, 0
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "validation": val,
                        "class_names": CLASSES, "protocol": protocol}, OUTPUT / "best.pt")
        else:
            stale += 1
        scheduler.step()
        entry = {"epoch": epoch + 1, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
                 "validation": val, "seconds": time.time() - tick, "best_epoch": best_epoch}
        history.append(entry)
        (OUTPUT / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "best_epoch": best_epoch,
                    "stale": stale, "history": history, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, last)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={entry['train_loss']:.4f}, train_acc={entry['train_accuracy']:.4f}, val_f1={val['macro_f1']:.4f}, val_acc={val['accuracy']:.4f}, best={best_epoch}, seconds={entry['seconds']:.1f}", flush=True)
        if epoch + 1 >= 10 and stale >= 7:
            print("Early stopping", flush=True); break
    model.load_state_dict(torch.load(OUTPUT / "best.pt", map_location=device, weights_only=False)["model"])
    metrics = {s: evaluation(model, loader, device, OUTPUT / f"{s}_predictions.csv") for s, loader in loaders.items()}
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lines = ["# Fine-tuned EfficientNet-B0 CNN baseline", "", f"Validation-selected epoch: {best_epoch}.",
             "", "| Split | Images | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for split, m in metrics.items():
        lines.append(f"| {split} | {m['n']} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |")
    lines += ["", "The frozen-statistics CNN is fine-tuned centrally on the existing lesion-separated training split. Validation macro F1 selects the checkpoint. Test sets were used in previous experiments and are not fresh blind benchmarks. Official-set lesion IDs are unavailable, so cross-set lesion independence is unverified.",
              "", "See protocol.json, history.json, metrics.json and prediction CSVs for complete details. The scores are not calibrated probabilities. This research model is not a clinical diagnostic tool."]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("Training and evaluations complete:", OUTPUT, flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
