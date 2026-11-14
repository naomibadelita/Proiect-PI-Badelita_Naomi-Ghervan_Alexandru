from torch.utils.data import DataLoader
from dataset import DawnDataset,custom_collate_fn
from pathlib import Path
import yaml
import numpy as np
import torch
from ultralytics import YOLO
from torch.optim import AdamW
from ultralytics.utils.loss import v8DetectionLoss
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open (PROJECT_ROOT/ "configs" /"train_config.yaml","r") as f:
    config=yaml.safe_load(f)

np.random.seed(config["train"]["seed"])
torch.manual_seed(config["train"]["seed"])

image_dir=PROJECT_ROOT/config["paths"]["image_dir"]
labels_dir=PROJECT_ROOT/config["paths"]["labels_dir"]
imageVal_dir=PROJECT_ROOT/config["paths"]["imageVal_dir"]
labelsVal_dir=PROJECT_ROOT/config["paths"]["labelsVal_dir"]

image_resize=config["train"]["imgsz"]
batch_size=config["train"]["batchsize"]
number_workers=config["train"]["workers"]

image_train=DawnDataset(image_dir,labels_dir)
image_loader=DataLoader(image_train,batch_size,shuffle=True,num_workers=number_workers,collate_fn=custom_collate_fn)
val_dataset=DawnDataset(imageVal_dir,labelsVal_dir)
val_loader=DataLoader(val_dataset,batch_size,shuffle=False,num_workers=number_workers,collate_fn=custom_collate_fn)

#YOLO
yolo_wrapper=YOLO("yolov8n.pt")
yolo_model=yolo_wrapper.model

args_dict = getattr(yolo_model, 'args', {})
if isinstance(args_dict, dict):
    args_dict.setdefault('box', 7.5)
    args_dict.setdefault('cls', 0.5)
    args_dict.setdefault('dfl', 1.5) 
    yolo_model.args = SimpleNamespace(**args_dict)

device_id=config["train"]["device"]
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

yolo_model.to(device)
for param in yolo_model.parameters():
    param.requires_grad = True
optimizer = AdamW(
    yolo_model.parameters(),
    lr=config["train"]["lr0"],      
    weight_decay=0.0005)  

criterias=v8DetectionLoss(yolo_model)

save_dir = PROJECT_ROOT / "runs" / "custom_train"
save_dir.mkdir(parents=True, exist_ok=True) 
best_model_raw_path = save_dir / "best_model_raw.pt"
best_val_loss = float('inf')              


#Training loop
epochs=config["train"]["epochs"]

for epoch in range(epochs):
    yolo_model.train()
    train_loss_accum = 0.0
    for inputs,targets in image_loader: #gives us batch by batch
        inputs= inputs.to(device) 
        targets = targets.to(device)

        optimizer.zero_grad()

        predicted_value=yolo_model(inputs)
        batch_data = {
            'batch_idx': targets[:, 0],
            'cls': targets[:, 1].view(-1, 1),
            'bboxes': targets[:, 2:]
        }
        loss, loss_items = criterias(predicted_value, batch_data)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / len(image_loader)
    val_loss_accum=0.0
    yolo_model.eval()
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            val_preds = yolo_model(val_inputs)
            val_batch_data = {
                'batch_idx': val_targets[:, 0],
                'cls': val_targets[:, 1].view(-1, 1),
                'bboxes': val_targets[:, 2:]
            }
            val_loss, val_loss_items = criterias(val_preds, val_batch_data)
            val_loss = val_loss.sum()
            val_loss_accum += val_loss.item()

    avg_val_loss = val_loss_accum / len(val_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint = {
            "model": yolo_model,
            "train_args": {}  # Punem un dicționar gol pentru a preveni alte erori la instanțiere
        }
        torch.save(checkpoint, best_model_raw_path)

final_model = YOLO(save_dir / "best_model_raw.pt")
cale_yaml = str(PROJECT_ROOT / "configs" / "data_raw.yaml")
metrics = final_model.val(data=cale_yaml)
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall: {metrics.box.r.mean():.4f}")
