import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import os
import yaml

class DawnDataset(Dataset):
    def __init__(self,image_dir,labels_dir):
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        with open (PROJECT_ROOT/ "configs" /"train_config.yaml","r") as f:
            config=yaml.safe_load(f)
        self.image_dir=Path(image_dir)
        self.labels_dir=Path(labels_dir)
        self.image_list = sorted(os.listdir(self.image_dir))
        self.image_size=config["train"]["imgsz"]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
        if index>=len(self.image_list):
            raise IndexError(f"Indexul {index} trebuie să fie mai mic decât {len(self.image_list)}")

        image_name=self.image_list[index]
        image_path=str(self.image_dir/image_name)
        img=cv2.imread(image_path)
        h,w,channels=img.shape

        img_resized=cv2.resize(img,(self.image_size,self.image_size))
        image_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        normalized_image =image_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(normalized_image, (2, 0, 1))
        img_tensor = torch.tensor(img_transposed, dtype=torch.float32)

        #labels
        txt_name = Path(image_name).stem + ".txt"
        txt_path = self.labels_dir / txt_name
        boxes = []
        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x, y, box_w, box_h = map(float, parts[1:])
                        boxes.append([class_id, x, y, box_w, box_h])
        if len(boxes) > 0:
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
