# FastSmoothSAM: A Fast Smooth Method For Segment Anything Model

![framework](assets/framework.jpg)
![stage](assets/stage.png)

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

![large](assets/large.png)

![result](assets/result.png)


