# Animal Pose Estimation with AnimalRTPose

Aanimal pose estimation aims to detect the keypoints of different species. It provides detailed behavioral analysis for neuroscience, medical and ecology applications. Some results are shown below.
![](https://s3.bmp.ovh/imgs/2024/08/19/0e1d3cc45f840729.jpg)
## Installation

1. Create a conda virtual environment and activate it.

```
conda create -n animalrtpose python=3.10
conda activate animalrtpose
```

2. Install PyTorch (rqeuired versio >= 1.8) and torchvision following the [official instructions](https://pytorch.org/). We use PyTorch 2.5.1+cu118. And we have used PyTorch 2.1.0+cu118 in history, you can refer to the [history](https://github.com/wux024/ultralytics/tree/cspnext/configs) for more details.

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

3. Clone the ultralytics repository and install the required dependencies.

```
git clone https://github.com/wux024/ultralytics.git
cd ultralytics
git checkout animalrtpose
pip install -v -e .
```

## Training , Evaluation and Prediction

1. The datasets (e.g. Animal Pose) used AnimalRTPose could be downloaded by contacting the corresponding author and me (<EMAIL>wux024@nenu.edu.cn). Extract the dataset to the `data` folder.
```text
ultralytics
├── ultralytics
├── docs
├── tests
├── tools
├── configs
├── weights
├── datasets
    │── animalpose
        ├── images_
            |── train
            |── val
            |── test
        │── labels
        │── annotations
    |—— other datasets
```
The rationale behind renaming the 'images' directory to 'images_' is to accommodate the project's expansion into a broader array of scenarios. Specifically, at the start of the process, the training, evaluation, and inference scripts will temporarily rename this directory back to 'images'. Once these operations are completed, the directory name will revert to 'images_'. This practice helps to streamline the workflow and ensure that each phase of the project can be executed seamlessly without conflicts.

2. Run the following command to train the model:
```
python tools/pose_train.py --dataset animalpose --models n
```
The pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/15L-q6CETD3dC8aOknamEunZaDEVlA1na?usp=drive_link). You should move the downloaded model to the `weights` folder. And, we add '--model-type' parameter to specify the model type, such as 'yolov8-pose' or 'yolo11-pose'.

3. The training log and checkpoints will be saved in the `runs/animalrtpose/train/animalpose/animalrtpose-n` folder. 

4. Test the trained model:
``` 
python tools/pose_val.py --dataset animalpose --models n
python tools/pose_modify_categories_id.py --dataset animalpose
python tools/pose_coco_eval.py --dataset animalpose --models n
```

5. The evaluation results will be saved in the `runs/animalrtpose/eval/animalpose/animalrtpose-n` folder.

6. Use the trained model to predict the pose of animals, and we provide two ways to do this:
```
python tools/pose_stream_inference.py --dataset animalpose --models n 
```
or 
```
python tools/pose_normal_inference.py --source path/to/image/or/video --dataset animalpose 
--model path/to/best.pt --project path/to/save_folder
```
`pose_stream_inference.py` is used for datasets, the predicted pose will be saved in the `runs/animalrtpose/predict/animalpose/animalrtpose-n` folder. While `pose_normal_inference.py` is used for single image or video pose estimation. The predicted pose will be saved in the `path/to/save_folder` folder.


## Animal Pose Estimation Results on Benchmark Datasets

### Animal Pose Dataset

| Model | size(pixels) | mAP50-95 | CPU(ms) | 2080Ti(ms) | 3090Ti(ms) | A800(ms) | Params(M) | GFLOPs |
| :-----: | :------------: | :--------: | :--------:| :-----------: | :--------: | :--------: | :---------: | :------: |
| AnimalRTPose-N| 640 | 72.0| 61.3| 2.1| 1.3 | 0.9 | 2.9 | 8.5 |
| AnimalRTPose-S| 640 | 76.0| 135.1| 3.6| 2.5 | 1.3 | 9.8 | 26.6 |
| AnimalRTPose-M| 640 | 77.2| 291.5| 7.5 | 4.9 | 2.4| 24.2|65.6|
| AnimalRTPose-L| 640 | 77.8| 506.5| 11.4 | 7.5  | 3.7 | 47.9|131.2|
| AnimalRTPose-X| 640 | 80.1| 722.8| 17.1 |  11.9 | 5.6| 74.8|204.6|

- **mAP<sup>val</sup>** values are for single-model single-scale on [Animal Pose](https://sites.google.com/view/animal-pose/) dataset. <br>Reproduce by `python tools/pose_val.py --dataset animalpose && python tools/pose_modify_categories_id.py --dataset animalpose && python tools/pose_coco_eval.py --dataset animalpose --models n`.
- **Speed** averaged over COCO val images using [NVIDIA RTX 2080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/) , [NVIDIA RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/), [NVIDIA A800](https://www.nvidia.cn/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf) and [12th Gen Intel(R) Core(TM) i5-12400   2.50 GHz](https://www.intel.cn/content/www/cn/zh/products/sku/134586/intel-core-i512400-processor-18m-cache-up-to-4-40-ghz/specifications.html) instance. <br>Reproduce by `python tools/pose_val.py --dataset animalpose`

### APT36K 

| Model | size(pixels) | mAP50-95 | CPU(ms) | 2080Ti(ms) | 3090Ti(ms) | A800(ms) | Params(M) | GFLOPs |
| :-----: | :------------: | :--------: | :--------:| :-----------: | :--------: | :--------: | :---------: | :------: |
| AnimalRTPose-N| 640 | 68.2| -| 2.0| 1.2 | - | 2.8 | 8.2 |
| AnimalRTPose-S| 640 | 78.1| -| 3.4| 2.3 | - | 9.7 | 26.3 |
| AnimalRTPose-M| 640 | 80.9| -| 7.4 | 4.7 | -| 25.1|66.0|
| AnimalRTPose-L| 640 | 82.3| -| 11.7 | 7.4  | - | 51.8|134.3|
| AnimalRTPose-X| 640 | 85.5| -| 17.6 |  11.9 | -| 80.8|209.5|

### ATRW

| Model | size(pixels) | mAP50-95 | CPU(ms) | 2080Ti(ms) | 3090Ti(ms) | A800(ms) | Params(M) | GFLOPs |
| :-----: | :------------: | :--------: | :--------:| :-----------: | :--------: | :--------: | :---------: | :------: |
| AnimalRTPose-N| 640 | 84.1| -| 2.1| - | - | 2.8 | 8.0 |
| AnimalRTPose-S| 640 | 84.7| -| 3.6| - | - | 9.6 | 25.9 |
| AnimalRTPose-M| 640 | 83.9| -| 7.6 | - | -| 24.0|64.9|
| AnimalRTPose-L| 640 | 82.8| -| 11.4 | -  | - | 47.9|131.2|
| AnimalRTPose-X| 640 | 88.4| -| 17.2 |  - | -| 74.8|204.5|

### TopViewMouse-5K

| Model | size(pixels) | mAP50-95 | CPU(ms) | 2080Ti(ms) | 3090Ti(ms) | A800(ms) | Params(M) | GFLOPs |
| :-----: | :------------: | :--------: | :--------:| :-----------: | :--------: | :--------: | :---------: | :------: |
| AnimalRTPose-N| 640 | 74.0| -| 2.9| - | - | 3.1 | 9.3 |
| AnimalRTPose-S| 640 | 82.4| -| 4.4| - | - | 10.1 | 27.6 |
| AnimalRTPose-M| 640 | 82.5| -| 7.9 | - | -| 24.5|66.9|
| AnimalRTPose-L| 640 | 83.4| -| 11.8 | -  | - | 48.3|132.5|
| AnimalRTPose-X| 640 | 83.2| -| 17.3 |  - | -| 74.8|204.7|