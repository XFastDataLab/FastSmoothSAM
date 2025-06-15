# FastSmoothSAM: A Fast Smooth Method For Segment Anything Model

## Installation

Clone the repository locally:

```shell
git clone https://github.com/XFastDataLab/FastSmoothSAM.git
```

Create the conda env. The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

```shell
conda create -n FastSmoothSAM python=3.9
conda activate FastSmoothSAM
```

Install the packages:

```shell
cd FastSmoothSAM
pip install -r requirements.txt
```

Install CLIP:

```shell
pip install git+https://github.com/openai/CLIP.git
```

## <a name="GettingStarted"></a> Getting Started

First download a [model checkpoint](https://github.com/XFastDataLab/FastSmoothSAM/releases/tag/v1.0.0).

## Results

All result were tested on a single NVIDIA GeForce RTX 3060Ti.

### 1. Inference time

Running Speed under Different Point Prompt Numbers(ms).
| method | params | 1 | 10 | 100 | E(16x16) | E(32x32\*) | E(64x64) |
|:------------------:|:--------:|:-----:|:-----:|:-----:|:----------:|:-----------:|:----------:|
| SAM-H | 0.6G | 446 | 464 | 627 | 852 | 2099 | 6972 |
| SAM-B | 136M | 110 | 125 | 230 | 432 | 1383 | 5417 |
| FastSAM | 68M | 40 |40 | 40 | 40 | 40 | 40 |

### 2. Memory usage

|  Dataset  | Method  | GPU Memory (MB) |
| :-------: | :-----: | :-------------: |
| COCO 2017 | FastSAM |      2608       |
| COCO 2017 |  SAM-H  |      7060       |
| COCO 2017 |  SAM-B  |      4670       |

### 3. Zero-shot Transfer Experiments

#### Object Proposals


#### Instance Segmentation On COCO 2017

|  method  |  AP  | APS  | APM  | APL  |
| :------: | :--: | :--: | :--: | :--: |
| ViTDet-H | .510 | .320 | .543 | .689 |
|   SAM    | .465 | .308 | .510 | .617 |
| FastSAM  | .379 | .239 | .434 | .500 |

### 4. Performance Visualization

Several segmentation results:

#### Natural Images

![Natural Images](assets/eightpic.png)

#### Text to Mask

![Text to Mask](assets/dog_clip.png)

### 5.Downstream tasks

The results of several downstream tasks to show the effectiveness.

#### Anomaly Detection

![Anomaly Detection](assets/anomaly.png)

#### Salient Object Detection

![Salient Object Detection](assets/salient.png)

#### Building Extracting

![Building Detection](assets/building.png)

