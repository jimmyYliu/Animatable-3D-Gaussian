# Animatable 3D Gaussian
Yang Liu*, Xiang Huang*, Minghan Qin, Qinwei Lin, Haoqian Wang (* indicates equal contribution)<br>
| [Webpage](https://jimmyyliu.github.io/Animatable-3D-Gaussian/) | [Full Paper](https://arxiv.org/pdf/2311.16482.pdf) | [Video](https://www.youtube.com/watch?v=fBkvl-oWrVc)

![Image text](assets/cover.png)
Abstract: *Neural radiance fields are capable of reconstructing high-quality drivable human avatars but are expensive to train and render. To reduce consumption, we propose Animatable 3D Gaussian, which learns human avatars from input images and poses. We extend 3D Gaussians to dynamic human scenes by modeling a set of skinned 3D Gaussians and a corresponding skeleton in canonical space and deforming 3D Gaussians to posed space according to the input poses. We introduce hash-encoded shape and appearance to speed up training and propose time-dependent ambient occlusion to achieve high-quality reconstructions in scenes containing complex motions and dynamic shadows. On both novel view synthesis and novel pose synthesis tasks, our method outperforms existing methods in terms of training time, rendering speed, and reconstruction quality. Our method can be easily extended to multi-human scenes and achieve comparable novel view synthesis results on a scene with ten people in only 25 seconds of training.*



## Prerequisites

* Cuda 11.7
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

## Train
To train a scene, run

```bash
python3 ./train.py --config-name config_path=<path to main config file> dataset=<path to dataset config file>
```

For the PeopleSnapshot dataset, run
```bash
python train.py --config-name peoplesnapshot.yaml dataset=peoplesnapshot/male-3-casual
```

For the GalaBasketball dataset, run
```bash
python train.py --config-name gala.yaml dataset=gala/idle
```

## Test

To test the model performance on the test set, run
```bash
python test.py --config-name config_path=<path to main config file> dataset=<path to dataset config file>
```

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{liu2023animatable,
  title={Animatable 3D Gaussian: Fast and High-Quality Reconstruction of Multiple Human Avatars},
  author={Liu, Yang and Huang, Xiang and Qin, Minghan and Lin, Qinwei and Wang, Haoqian},
  journal={arXiv preprint arXiv:2311.16482},
  year={2023}
}</code></pre>
  </div>
</section>