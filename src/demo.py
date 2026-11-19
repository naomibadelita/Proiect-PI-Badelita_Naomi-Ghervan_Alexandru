from PIL import Image, ImageTk
import tkinter as tk
from ultralytics import YOLO
from pathlib import Path

from preproces import enhance

def show_comparison_window(items, max_size=(640, 320), title="Detection Comparison"):
  root = tk.Tk()
  root.title(title)

  photo_refs = []

  for idx, (label_text, pil_img) in enumerate(items):
    row, col = divmod(idx, 2)

    resized = pil_img.copy()
    resized.thumbnail(max_size)
    photo = ImageTk.PhotoImage(resized)
    photo_refs.append(photo)

    cell = tk.Frame(root, padx=10, pady=10)
    cell.grid(row=row, column=col)

    tk.Label(cell, text=label_text, font=("TkDefaultFont", 10, "bold")).pack()
    tk.Label(cell, image=photo).pack(pady=(4, 0))

  root.mainloop()

if __name__ == '__main__':
    img1=Image.open('../img/dusttornado-016.jpg')
    img2=Image.open('../img/foggy-008.jpg')
    img3=Image.open('../img/foggy-039.jpg')

    weights_dir = Path("../weights")

    model_v8nR = YOLO(weights_dir / "best_model_trainRAW_yolov8n.pt")
    model_v8nP = YOLO(weights_dir / "best_model_trainPRE_yolov8n.pt")

    imges=[img1,img2,img3]

    for img in imges:
        img=img.convert("RGB")
        preproc=enhance(img)
        arr_v8=model_v8nR(preproc, imgsz=640, conf=0.25)
        arr_v8R=model_v8nR(img, imgsz=640, conf=0.25)

        img_v8=Image.fromarray(arr_v8[0].plot()[:, :, ::-1])
        img_v8R = Image.fromarray(arr_v8R[0].plot()[:, :, ::-1])

        show_comparison_window(
            [
                ("original", img),
                ("preprocesat", preproc),
                ("yolov8n_RAW", img_v8R),
                ("yolov8n_PRC", img_v8),
            ],
            title=f"Image",
        )