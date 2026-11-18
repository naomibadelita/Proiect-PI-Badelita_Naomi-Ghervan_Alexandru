from torch.utils.data import DataLoader
from data.dataset import DawnDataset,custom_collate_fn
from pathlib import Path
import yaml
import numpy as np
import torch
from ultralytics import YOLO
from torch.optim import AdamW
from ultralytics.utils.loss import v8DetectionLoss
from types import SimpleNamespace
from ultralytics.nn.tasks import DetectionModel
import pandas as pd
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

image_train=DawnDataset(image_dir,labels_dir,train=True)
image_loader=DataLoader(image_train,batch_size,shuffle=True,num_workers=number_workers,collate_fn=custom_collate_fn)
val_dataset=DawnDataset(imageVal_dir,labelsVal_dir,train=False)
val_loader=DataLoader(val_dataset,batch_size,shuffle=False,num_workers=number_workers,collate_fn=custom_collate_fn)

#YOLO
# yolo_wrapper=YOLO("yolov8n.pt")
# yolo_model=yolo_wrapper.model

# args_dict = getattr(yolo_model, 'args', {})
# if isinstance(args_dict, dict):
#     args_dict.setdefault('box', 7.5)
#     args_dict.setdefault('cls', 0.5)
#     args_dict.setdefault('dfl', 1.5) 
#     yolo_model.args = SimpleNamespace(**args_dict)
NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck"}
NC = len(NAMES)

yolo_model = DetectionModel("yolov8n.yaml", nc=NC)
pretrained = YOLO("yolov8n.pt").model
yolo_model.load(pretrained)   

yolo_model.names = NAMES
yolo_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

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
epochs_no_improve = 0
patience = config["train"]["patience"]

wandb.init(
    project="Project-Object-Detection-Traffic",  
    name="YOLOv8n-Custom-Train", 
    config=config              
)

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
            if isinstance(val_preds, tuple):
                val_preds = val_preds[1]
            val_batch_data = {
                'batch_idx': val_targets[:, 0],
                'cls': val_targets[:, 1].view(-1, 1),
                'bboxes': val_targets[:, 2:]
            }
            val_loss, val_loss_items = criterias(val_preds, val_batch_data)
            val_loss = val_loss.sum()
            val_loss_accum += val_loss.item()

    avg_val_loss = val_loss_accum / len(val_loader)
    print(f"Epoch {epoch+1}/{epochs} | train {avg_train_loss:.4f} | val {avg_val_loss:.4f}")
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "learning_rate": optimizer.param_groups[0]['lr'] # Opțional: util să vezi cum evoluează lr-ul
    })
    if avg_val_loss < best_val_loss:
        epochs_no_improve = 0
        best_val_loss = avg_val_loss
        checkpoint = {
            "model": yolo_model,
            "train_args": {}  # Punem un dicționar gol pentru a preveni alte erori la instanțiere
        }
        torch.save(checkpoint, best_model_raw_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping la epoca {epoch+1}")
            break

wandb.finish()

final_model = YOLO(save_dir / "best_model_raw.pt")
cale_raw = str(PROJECT_ROOT / "configs" / "data_raw.yaml")
cale_pre = str(PROJECT_ROOT / "configs" / "data_preprocessed.yaml")

for nume, cale in [("RAW", cale_raw), ("PREPROCESSED", cale_pre)]:
    print(f"=== TEST {nume} ===")
    m = final_model.val(data=cale, split="test", verbose=False)
    
    table_data = []

    total_images = len(val_dataset) 
    total_instances = sum(m.box.ap_class_index) if hasattr(m.box, 'ap_class_index') else 0 
    table_data.append({
        'Class': 'all',
        'Images': total_images,
        'Instances': sum([len(targets) for _, targets in val_dataset]),
        'Box(P)': m.box.p.mean(),
        'R': m.box.r.mean(),
        'mAP50': m.box.map50,
        'mAP50-95': m.box.map
    })

    for i, class_name in NAMES.items():
        if i in m.box.ap_class_index:
             class_index_in_results = list(m.box.ap_class_index).index(i)
             
             table_data.append({
                'Class': class_name,
                'Images': 'N/A', 
                'Instances': 'N/A', 
                'Box(P)': m.box.p[class_index_in_results], 
                'R': m.box.r[class_index_in_results],
                'mAP50': m.box.maps[class_index_in_results],
                'mAP50-95': m.box.ap[class_index_in_results] 
             })
        else:
             table_data.append({
                 'Class': class_name,
                 'Images': 0,
                 'Instances': 0,
                 'Box(P)': 0.0,
                 'R': 0.0,
                 'mAP50': 0.0,
                 'mAP50-95': 0.0
             })

    df = pd.DataFrame(table_data)
    cale_salvare_csv = PROJECT_ROOT / "runs" / "detect" / f"metrici_tabelare_{nume}.csv"

    cale_salvare_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(cale_salvare_csv, index=False)
    print(f"The metrics were saved at: {cale_salvare_csv}")