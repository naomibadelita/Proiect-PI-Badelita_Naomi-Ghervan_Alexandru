# Traffic Object Detection and Recognition under Adverse Weather Conditions (Fog, Rain, etc.)

## Project Description


## State of the Art

Analysis of similar articles / applications / projects from the last 5 years.

| No. | Author(s) / Year | Title of the article / project | Application / Domain | Technologies used | Methodology / Approach | Results | Limitations | Additional comments |
|-|------------------|--------------------------------|----------------------|-------------------|------------------------|---------|-------------|---------------------|
| 1 | Teena Sharma et al./2022 | Deep Learning-Based Object Detection and Scene Perception under Bad Weather Conditions | Autonomous Driving and road safety| Python, OpenCV, Google Colab cloud, YOLOv5, Roboflow | The YOLOv5 algorithm was pretrained with COCO weights and fine-tuned with Roboflow images (11 classes) for 100 epochs. The performance of the model was evaluated through a simulated video under normal weather conditions and rainy weather conditions.| precision=0.474, recall=0.337, mAP=0.258 | They evaluated the model using digitally simulated rain, not real life adverse weather conditions. Moreover, they didn't have many examples of all the labels (4 for traffic lights, 27 for bikers out of 1520 labels)| The algorithm is not trained on multiple weather conditions, the rain was digitally simulated. The algorithm should have been trained on multiple weather conditions and these conditions should not be simulated. Also, a bigger diveristy in labels would make a big difference. |
| 2 | Haodong Hu et al./2025 | YOLOv8-Based Multi-Object Detection Method for Adverse Weather Road Conditions | Autonomous Driving and road safety | PyTorch 1.9.0, Python 3.8, CUDA 11.8, 12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz, NVIDIA GeForce RTX 4090 with 24GB of VRAM, BotNet, LSKA, BIFPN/AWFPN, RTTS, DAWN | The architecture of the model YOLOv8 was improved so that it does not need preprocessing. At the end of the backbone (extracts the main features of an image) they added the BotNet module, which makes it possible for the model to not look only at a small number of neighboring pixels and decide what is there based on that, but take into consideration the global context. The next section is the neck section (combines small scale information with large scale information). Here it was added the LSKA module into the c2f block, which made it possible for bigger grids to be processed faster (from 7X7 to 1X7 and 7X1). In the head section (it is decided what each grid contains), it was replaced CSPNet with BiFPN, which gives more importance to maps with less noise. Also the training took place on the RTTS and DAWN datasets for 200 epochs.| precision=0.996, recall=0.7, mAP= 0.679| Only 2000 images were used, out of which 1250 images were depicting foggy weather conditions. The calculations also became much more complex and the GFLOPs rose from 8.9 to 20.6. | This paper presents a superior solution for our assignment.|
| 3 | Madhavi Kondapally et al./2024| Object Detection in Transitional Weather Conditions for Autonomous Vehicles| AIWD6 dataset, Resnet50, Python 3.7, PyTorch, TensorFlow, GRL,DETR transformer, NVIDIA Tesla M60 GPU with 8GB of memory| | | | | |
| 4 | | | | | | | | |
| 5 | | | | | | | | |

- **Application / Domain** – domain in which the described solution is applied (e.g., face recognition, augmented reality).
- **Technologies used** – technologies / hardware used in the project (e.g., OpenCV, TensorFlow, ZED stereo cameras).
- **Methodology / Approach** – description of the method used in the project.
- **Results** – main results obtained (metrics used for evaluation).
- **Limitations** – limitations encountered in research or implementation.
- **Additional comments** – observations on relevance to the current project; suggestions for improving the methodology.

## Proposed Architecture


### Components