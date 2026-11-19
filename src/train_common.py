import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import ModelEMA

from data.dataset import DawnDataset, custom_collate_fn

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck"}
NC = len(NAMES)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

VARIANTS = {
    # variant -> (tag pentru fisiere, sufix chei in train_config.yaml)
    "raw": ("trainRAW", ""),
    "preprocessed": ("trainPRE", "p"),
}


def count_per_class(labels_dir: Path, images_dir: Path):
    """Numarul de imagini si de instante pentru fiecare clasa (ca in tabelul Ultralytics)."""
    n_images = len([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT])
    imgs_with = {c: 0 for c in NAMES}
    instances = {c: 0 for c in NAMES}
    for txt in labels_dir.glob("*.txt"):
        seen = set()
        with open(txt) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                c = int(parts[0])
                if c in instances:
                    instances[c] += 1
                    seen.add(c)
        for c in seen:
            imgs_with[c] += 1
    return n_images, imgs_with, instances


def metrics_rows(m, labels_dir: Path, images_dir: Path):
    """Transforma rezultatul YOLO.val() intr-o lista de randuri: 'all' + cate unul pe clasa."""
    n_images, imgs_with, instances = count_per_class(labels_dir, images_dir)
    rows = [{"Class": "all", "Images": n_images, "Instances": sum(instances.values()),
             "P": m.box.mp, "R": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map}]
    present = {int(c): pos for pos, c in enumerate(m.box.ap_class_index)}
    for cid, name in NAMES.items():
        if cid in present:
            p, r, ap50, ap = m.box.class_result(present[cid])
        else:  # clasa nu apare in acest split
            p = r = ap50 = ap = 0.0
        rows.append({"Class": name, "Images": imgs_with[cid], "Instances": instances[cid],
                     "P": float(p), "R": float(r), "mAP50": float(ap50), "mAP50-95": float(ap)})
    return rows


def loss_components(loss_items):
    if isinstance(loss_items, dict):
        return torch.stack([v.detach().float().reshape(()) for v in loss_items.values()][:3]).cpu()
    return loss_items.detach().float().reshape(-1)[:3].cpu()


def resolve_data_yaml(tag: str, out_dir: Path) -> str:
    with open(PROJECT_ROOT / "configs" / f"data_{tag}.yaml") as f:
        data = yaml.safe_load(f)
    data["path"] = str((PROJECT_ROOT / data["path"]).resolve())
    out = out_dir / f"data_{tag}_abs.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(out)


def save_checkpoint(model, path: Path):
    torch.save({"model": deepcopy(model).float(), "train_args": {}}, path)


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if name == "bias" or isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                no_decay.append(p)
            else:
                decay.append(p)
    return AdamW([{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.937, 0.999))


def run_training(variant: str):
    train_tag, key = VARIANTS[variant]

    with open(PROJECT_ROOT / "configs" / "train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    tcfg = config["train"]
    paths = config["paths"]

    seed = tcfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    image_dir = PROJECT_ROOT / paths[f"image_dir{key}"]
    labels_dir = PROJECT_ROOT / paths[f"labels_dir{key}"]
    imageVal_dir = PROJECT_ROOT / paths[f"imageVal_dir{key}"]
    labelsVal_dir = PROJECT_ROOT / paths[f"labelsVal_dir{key}"]

    image_resize = tcfg["imgsz"]
    batch_size = tcfg["batchsize"]
    workers = tcfg["workers"]
    device_id = tcfg["device"]
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    train_ds = DawnDataset(image_dir, labels_dir, train=True)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=workers,
                              collate_fn=custom_collate_fn, drop_last=True)
    val_ds = DawnDataset(imageVal_dir, labelsVal_dir, train=False)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=workers,
                            collate_fn=custom_collate_fn)

    # ---- model ----
    model = DetectionModel("yolov8n.yaml", nc=NC)
    model.load(YOLO("yolov8n.pt").model)
    model.names = NAMES
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = True

    optimizer = build_optimizer(model, tcfg["lr0"], tcfg.get("weight_decay", 0.01))
    criterion = v8DetectionLoss(model)
    ema = ModelEMA(model)

    epochs = tcfg["epochs"]
    lrf = tcfg.get("lrf", 0.01)
    warmup_iters = max(round(tcfg.get("warmup_epochs", 3) * len(train_loader)), 100)
    total_iters = epochs * len(train_loader)

    def lr_at(it):
        if it < warmup_iters:  # warmup liniar
            return tcfg["lr0"] * (it + 1) / warmup_iters
        prog = (it - warmup_iters) / max(1, total_iters - warmup_iters)  # cosine pana la lr0*lrf
        return tcfg["lr0"] * (lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * prog)))

    save_dir = PROJECT_ROOT / "runs" / "custom_train"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"best_model_{train_tag}.pt"
    last_path = save_dir / f"last_model_{train_tag}.pt"
    epoch_csv = save_dir / f"metrics_per_epoch_{train_tag}.csv"
    epoch_rows = []
    data_yaml = resolve_data_yaml(variant, save_dir)

    patience = tcfg["patience"]
    best_fitness, epochs_no_improve = -1.0, 0

    wandb.init(project="Project-Object-Detection-Traffic", name=f"YOLOv8n-{train_tag}", config=config)

    it = 0
    for epoch in range(epochs):
        # ---------------- train ----------------
        model.train()
        tr_loss = torch.zeros(3)  # box, cls, dfl
        for inputs, targets in train_loader:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(it)
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            batch = {"batch_idx": targets[:, 0], "cls": targets[:, 1].view(-1, 1), "bboxes": targets[:, 2:]}
            loss, loss_items = criterion(preds, batch)
            optimizer.zero_grad()
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            ema.update(model)
            tr_loss += loss_components(loss_items)
            it += 1
        tr_loss /= len(train_loader)

        ema.ema.eval()
        va_loss = torch.zeros(3)
        with torch.no_grad():
            for v_in, v_tg in val_loader:
                v_in, v_tg = v_in.to(device), v_tg.to(device)
                v_pred = ema.ema(v_in)
                if isinstance(v_pred, tuple):
                    v_pred = v_pred[1]
                v_batch = {"batch_idx": v_tg[:, 0], "cls": v_tg[:, 1].view(-1, 1), "bboxes": v_tg[:, 2:]}
                _, v_items = criterion(v_pred, v_batch)
                va_loss += loss_components(v_items)
        va_loss /= len(val_loader)

        save_checkpoint(ema.ema, last_path)
        m = YOLO(last_path).val(data=data_yaml, split="val", imgsz=image_resize, batch=batch_size * 2,
                                device=device_id, workers=workers, verbose=False, plots=False,
                                project=str(save_dir), name=f"val_{train_tag}", exist_ok=True)
        rows = metrics_rows(m, labelsVal_dir, imageVal_dir)
        fitness = 0.1 * m.box.map50 + 0.9 * m.box.map

        log = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
               "train/box_loss": tr_loss[0].item(), "train/cls_loss": tr_loss[1].item(),
               "train/dfl_loss": tr_loss[2].item(), "train_loss": tr_loss.sum().item(),
               "val/box_loss": va_loss[0].item(), "val/cls_loss": va_loss[1].item(),
               "val/dfl_loss": va_loss[2].item(), "val_loss": va_loss.sum().item(),
               "fitness": fitness}
        for r in rows:
            for k in ("P", "R", "mAP50", "mAP50-95"):
                log[f"val_{r['Class']}/{k}"] = r[k]
            epoch_rows.append({"epoch": epoch + 1, **r})
        wandb.log(log)
        pd.DataFrame(epoch_rows).to_csv(epoch_csv, index=False)  # rescris la fiecare epoca

        print(f"Epoch {epoch + 1}/{epochs} | train {tr_loss.sum():.4f} | val {va_loss.sum():.4f} | "
              f"mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f}")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        if fitness > best_fitness:
            best_fitness, epochs_no_improve = fitness, 0
            save_checkpoint(ema.ema, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping la epoca {epoch + 1}")
                break

    final_model = YOLO(best_path)
    test_tables = {}
    for nume, tag in [("RAW", "raw"), ("PREPROCESSED", "preprocessed")]:
        print(f"=== {train_tag} -> TEST {nume} ===")
        m = final_model.val(data=resolve_data_yaml(tag, save_dir), split="test",
                            imgsz=image_resize, device=device_id, verbose=True, plots=True,
                            project=str(save_dir), name=f"test_{train_tag}_{tag}", exist_ok=True)
        rows = metrics_rows(m, PROJECT_ROOT / "dataset" / tag / "labels" / "test",
                            PROJECT_ROOT / "dataset" / tag / "images" / "test")
        df = pd.DataFrame(rows)
        csv_path = PROJECT_ROOT / "runs" / "detect" / f"metrici_{train_tag}_test{nume}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        test_tables[f"test_{nume}_per_class"] = wandb.Table(dataframe=df) if wandb.run else None
        print(f"The metrics were saved at: {csv_path}")

    if wandb.run:
        wandb.log({k: v for k, v in test_tables.items() if v is not None})
        wandb.save(str(epoch_csv))
    wandb.finish()
