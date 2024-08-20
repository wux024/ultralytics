# Animal Pose Estimation with AnimalRTPose

Aanimal pose estimation aims to detect the keypoints of different species. It provides detailed behavioral analysis for neuroscience, medical and ecology applications. Some results are shown below.
![](https://s3.bmp.ovh/imgs/2024/08/19/0e1d3cc45f840729.jpg)
## Installation

1. Create a conda virtual environment and activate it.

```
conda create -n animalrtpose python=3.8
conda activate animalrtpose
```

2. Install PyTorch (rqeuired version: 2.1.0) and torchvision following the [official instructions](https://pytorch.org/).

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

3. Clone the ultralytics repository and install the required dependencies.

```
git clone https://github.com/wux024/animalpose.git
cd animalpose
git checkout cspnext
pip install -v -e .
```

## Training

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
    |—— other datasets
```

2. Run the following command to train the model:
```
python tools/pose_train_cspnext.py --dataset animalpose --pretrained --models n
```
The pretrained model can be downloaded from [here](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt). You should move the downloaded model to the `weights` folder.

3. The training log and checkpoints will be saved in the `runs/pose/train/apt36k/apt36k-yolov8n-cspnext` folder.

4. Test the trained model:
``` 
python tools/pose_val_cspnext.py --dataset animalpose --models n
python tools/pose_modify_categories_id.py --dataset animalpose
python tools/pose_coco_eval.py --dataset animalpose --model yolov8n-pose-cspnext
```

## Animal Pose Estimation Results on Benchmark Datasets

### Animal Pose Dataset

| Model | size(pixels) | mAP50-95 | CPU(ms) | 2080Ti(ms) | 3090Ti(ms) | A800(ms) | Params(M) | GFLOPs |
| :-----: | :------------: | :--------: | :--------:| :-----------: | :--------: | :--------: | :---------: | :------: |
| [AnimalRTPose-N](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)| 640 | 72.0| 61.38| 2.1|  | 0.8 | 2.884 | 8.5 |
| [AnimalRTPose-S](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)| 640 | 76.0| 135.1| 3.6|  | 1.4 | 9.796 | 26.6 |
| [AnimalRTPose-M](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)| 640 | 72.0| 61.38| 7.5 |   | 0.8| 24.183|65.6|
| [AnimalRTPose-L](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)| 640 | 72.0| 61.38| 11.4 |   | 0.8| 47.929|131.2|
| [AnimalRTPose-X](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)| 640 | 72.0| 61.38| 17.1 |   | 0.8| 74.812|204.6|

- **mAP<sup>val</sup>** values are for single-model single-scale on [Animal Pose](https://sites.google.com/view/animal-pose/) dataset. <br>Reproduce by `python tools/pose_val_cspnext.py --dataset animalpose && python tools/pose_modify_categories_id.py --dataset animalpose && python tools/pose_coco_eval.py --dataset animalpose --model yolov8n-pose-cspnext`.
- **Speed** averaged over COCO val images using [NVIDIA RTX 2080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/) , [NVIDIA RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/), [NVIDIA A800](https://www.nvidia.cn/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf) and [12th Gen Intel(R) Core(TM) i5-12400   2.50 GHz](https://www.intel.cn/content/www/cn/zh/products/sku/134586/intel-core-i512400-processor-18m-cache-up-to-4-40-ghz/specifications.html) instance. <br>Reproduce by `python tools/pose_val_cspnext.py --dataset animalpose`

More results and models will be added soon.