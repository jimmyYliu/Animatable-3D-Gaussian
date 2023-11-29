# Animatable 3D Gaussian
Yang Liu*, Xiang Huang*, Minghan Qin, Qinwei Lin, Haoqian Wang (* indicates equal contribution)<br>
| [Webpage](https://jimmyyliu.github.io/Animatable3DGaussian/) | [Full Paper](https://arxiv.org/pdf/2311.16482.pdf) | [Video](https://www.youtube.com/watch?v=BPmeEP65k2c)

## Prerequisites

* Cuda 11.6
* Conda
* A C++14 capable compiler
  * __Windows:__ Visual Studio 2019 or 2022
  * __Linux:__ GCC/G++ 8 or higher

## Setup
First make sure all the Prerequisites are installed in your operating system. Then, invoke

```bash
conda create --name anim-gaussian python=3.8
conda activate anim-gaussian
bash ./install.sh
```

## Prepare Data
We use PeopleSnapshot and GalaBasketball datasets and correspoding template body model. Please [download](https://drive.google.com/drive/folders/1xyLF7UwIrUaU5KU0IsEjYrz9hdTeZuza?usp=sharing) and organize as follows
```bash
|---data
|   |---Gala
|   |---PeopleSnapshot
|   |---smpl
```

## Running
To train and test, run

```bash
python3 ./train.py --config-name config_path=<path to main config file> dataset=<path to dataset config file>
```

For the PeopleSnapshot dataset, run
```bash
python3 ./train.py --config-name peoplesnapshot.yaml dataset=peoplesnapshot/male-3-casual
```

For the GalaBasketball dataset, run
```bash
python3 ./train.py --config-name gala.yaml dataset=gala/galaBasketball0
```