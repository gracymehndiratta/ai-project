"""Evaluate the saved validation-selected CNN without continuing training."""
from pathlib import Path
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from train_cnn_baseline import CLASSES, Images, OUTPUT, evaluation, model_from_weights


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = OUTPUT / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frame = pd.read_csv(OUTPUT / "split_manifest.csv")
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    model = model_from_weights().to(device)
    model.load_state_dict(checkpoint["model"])
    history = json.loads((OUTPUT / "history.json").read_text())
    scores = {}
    for split in ("validation", "test", "official_validation", "official_test"):
        dataset = Images(frame[frame.split == split], transform)
        loader = DataLoader(dataset, batch_size=24, num_workers=0, pin_memory=device.type == "cuda")
        scores[split] = evaluation(model, loader, device, OUTPUT / f"{split}_predictions.csv")
        print(f"{split}: n={scores[split]['n']}, accuracy={scores[split]['accuracy']:.4f}, macro_f1={scores[split]['macro_f1']:.4f}", flush=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(scores, indent=2))
    lines = ["# Fine-tuned EfficientNet-B0 CNN baseline", "",
             f"Training was stopped on request after {history[-1]['epoch']} completed epochs. The best checkpoint was selected at epoch {checkpoint['epoch']} using internal validation macro F1.", "",
             "| Split | Images | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for split, score in scores.items():
        lines.append(f"| {split} | {score['n']} | {score['accuracy']:.4f} | {score['balanced_accuracy']:.4f} | {score['macro_f1']:.4f} | {score['weighted_f1']:.4f} |")
    lines += ["", "The CNN was fine-tuned centrally on the existing lesion-separated training split. Validation macro F1 selected the checkpoint. Test sets were used in earlier experiments and are not fresh blind benchmarks. Official-set lesion identifiers are unavailable, so cross-set lesion independence is unverified.",
              "", "See protocol.json, history.json, metrics.json and prediction CSVs for details. Class scores are not calibrated probabilities. This research model is not a clinical diagnostic tool."]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Selected epoch {checkpoint['epoch']}; report: {OUTPUT / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
