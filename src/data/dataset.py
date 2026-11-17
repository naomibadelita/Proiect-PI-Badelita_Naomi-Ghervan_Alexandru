import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import os
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}  

class DawnDataset(Dataset):
    #o sa facem flip doar cand train este True
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
        if self.train and torch.rand(1).item() < 0.5:
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

        # tensorul se creează DIN canvas, la final
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
