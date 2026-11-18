import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import os
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}  

class DawnDataset(Dataset):
    def __init__(self,image_dir,labels_dir,train=False):
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        with open (PROJECT_ROOT/ "configs" /"train_config.yaml","r") as f:
            config=yaml.safe_load(f)
        self.image_dir=Path(image_dir)
        self.labels_dir=Path(labels_dir)
        self.image_list = sorted(
            f for f in os.listdir(self.image_dir)
            if Path(f).suffix.lower() in IMG_EXT
        )
        self.image_size=config["train"]["imgsz"]
        self.train=train
        self.aug=config.get("augment", {}) or {}

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
        
        if index >= len(self.image_list):
            raise IndexError(f"The index {index} must be smalled than {len(self.image_list)}")

        image_name = self.image_list[index]
        img = cv2.imread(str(self.image_dir / image_name))
        if img is None:                                  
            raise FileNotFoundError(f"I cannot read the image: {self.image_dir / image_name}")
        h, w = img.shape[:2]

        # etichete
        txt_path = self.labels_dir / (Path(image_name).stem + ".txt")
        boxes = []
        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x, y, box_w, box_h = map(float, parts[1:])
                        boxes.append([class_id, x, y, box_w, box_h])

        # flip orizontal (doar la train)
        if self.train and _rand() < self.aug.get("fliplr", 0.5):
            img = cv2.flip(img, 1)
            for b in boxes:
                b[1] = 1.0 - b[1]

        # LETTERBOX
        S = self.image_size
        scale = S / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img_resized = cv2.resize(img, (new_w, new_h))

        pad_x = (S - new_w) // 2
        pad_y = (S - new_h) // 2
        canvas = np.full((S, S, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized

        # ajustează casetele
        for b in boxes:
            b[1] = (b[1] * new_w + pad_x) / S
            b[2] = (b[2] * new_h + pad_y) / S
            b[3] = (b[3] * new_w) / S
            b[4] = (b[4] * new_h) / S

        # augmentari suplimentare, reduc overfitting-ul
        if self.train:
            canvas = _augment_hsv(canvas, self.aug.get("hsv_h", 0.015),
                                  self.aug.get("hsv_s", 0.5), self.aug.get("hsv_v", 0.3))
            canvas, boxes = _random_scale_translate(canvas, boxes, self.aug.get("scale", 0.4),
                                                    self.aug.get("translate", 0.1),
                                                    self.aug.get("min_box_frac", 0.25))

        # tensorul 
        image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(
            np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1)).copy())

        if boxes:
            targets_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            targets_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, targets_tensor

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    
    batch_targets = []
    for i, target in enumerate(targets):
        if target.shape[0] > 0:
            idx_tensor = torch.full((target.shape[0], 1), i, dtype=torch.float32)
            target_with_idx = torch.cat((idx_tensor, target), dim=1)
            batch_targets.append(target_with_idx)
            
    if len(batch_targets) > 0:
        batch_targets = torch.cat(batch_targets, dim=0)
    else:
        batch_targets = torch.zeros((0, 6), dtype=torch.float32)
        
    return images, batch_targets


def _rand(a=0.0, b=1.0):
    return a + (b - a) * torch.rand(1).item()


def _augment_hsv(img, hgain, sgain, vgain):
    if not (hgain or sgain or vgain):
        return img
    r = np.array([_rand(-1, 1) * hgain + 1, _rand(-1, 1) * sgain + 1, _rand(-1, 1) * vgain + 1])
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _random_scale_translate(img, boxes, scale, translate, min_box_frac):
    if not (scale or translate):
        return img, boxes
    S = img.shape[0]
    s = _rand(1 - scale, 1 + scale)
    tx = _rand(-translate, translate) * S
    ty = _rand(-translate, translate) * S
    c = S / 2
    M = np.array([[s, 0, (1 - s) * c + tx], [0, s, (1 - s) * c + ty]], dtype=np.float32)
    img = cv2.warpAffine(img, M, (S, S), borderValue=(114, 114, 114))

    out = []
    for cls, xc, yc, bw, bh in boxes:
        x1, y1 = (xc - bw / 2) * S, (yc - bh / 2) * S
        x2, y2 = (xc + bw / 2) * S, (yc + bh / 2) * S
        x1, x2 = x1 * s + M[0, 2], x2 * s + M[0, 2]
        y1, y2 = y1 * s + M[1, 2], y2 * s + M[1, 2]
        area_before = (x2 - x1) * (y2 - y1)
        x1, x2 = np.clip([x1, x2], 0, S)
        y1, y2 = np.clip([y1, y2], 0, S)
        nw, nh = x2 - x1, y2 - y1
        if nw < 2 or nh < 2 or area_before <= 0 or (nw * nh) / area_before < min_box_frac:
            continue
        out.append([cls, (x1 + x2) / 2 / S, (y1 + y2) / 2 / S, nw / S, nh / S])
    return img, out
