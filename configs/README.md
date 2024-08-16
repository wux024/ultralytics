# Animal Pose Estimation with AnimalRTPose

Aanimal pose estimation aims to detect the keypoints of different species. It provides detailed behavioral analysis for neuroscience, medical and ecology applications. Some results are shown below.![](https://s3.bmp.ovh/imgs/2024/07/27/e1b49c32bd1cccbf.jpg)
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

1. The datasets (e.g. APT36K) used AnimalRTPose could be downloaded by contacting the corresponding author and me (<EMAIL>wux024@nenu.edu.cn). Extract the dataset to the `data` folder.
```text
ultralytics
├── ultralytics
├── docs
├── tests
├── tools
├── configs
├── datasets
    │── apt36k
    |—— other datasets
```

2. Run the following command to train the model:
```
python tools/pose_train_cspnext.py --dataset apt36k --pretrained --models n
```

3. The training log and checkpoints will be saved in the `runs/pose/train/apt36k/apt36k-yolov8n-cspnext` folder.

4. Test the trained model:
``` 
python tools/pose_val_cspnext.py --dataset apt36k --models n
```

## Weights and Pretrained Models
[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A800 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [AnimalRTPose-N](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
| [AnimalRTPose-S](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
| [AnimalRTPose-M](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
| [AnimalRTPose-L](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
| [AnimalRTPose-X](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [APT36K](https://github.com/pandorgan/APT-36K?tab=readme-ov-file#apt-36k) dataset. <br>Reproduce by `python tools/pose_val_cspnext.py --dataset apt36k`
- **Speed** averaged over COCO val images using [NVIDIA A800](https://www.nvidia.cn/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf) and [12th Gen Intel(R) Core(TM) i5-12400   2.50 GHz](https://www.intel.cn/content/www/cn/zh/products/sku/134586/intel-core-i512400-processor-18m-cache-up-to-4-40-ghz/specifications.html) instance. <br>Reproduce by `python tools/pose_val_cspnext.py --dataset apt36k`