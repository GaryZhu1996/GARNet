# GARNet

This repository contains the source code for the paper [GARNet: Global-Aware Multi-View 3D Reconstruction Network and the Cost-Performance Tradeoff](https://arxiv.org/abs/2211.02299).

![Overview](/figures/overview.png)

![Global-Aware Generator](/figures/global_aware_generator.png)


## Cite this work

```
@article{zhu2022garnet,
  title={GARNet: Global-Aware Multi-View 3D Reconstruction Network and the Cost-Performance Tradeoff},
  author={Zhu, Zhenwei and Yang, Liying and Lin, Xuxin and Jiang, Chaohao and Li, Ning and Yang, Lin and Liang, Yanyan},
  journal={arXiv preprint arXiv:2211.02299},
  year={2022}
}
```

## Datasets

We use the [ShapeNet](https://www.shapenet.org/) in our experiments, which are available below:

- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

## Pretrained Models

The pretrained models on ShapeNet are available as follows:

- [GARNet](https://drive.google.com/file/d/1M-uxpQVVYDdWgr0MWfoTnxsu_53p7SAW/view?usp=share_link)
- [GARNet+](https://drive.google.com/file/d/1EtjpJIkS9kVQ1QZcqJfAeVaBNzD_VUUs/view?usp=share_link)
- [IMB for GARNet](https://drive.google.com/file/d/1yhIzE5oJLdjo24a8E6RElj3mRdOxDvy-/view?usp=share_link)
- [IMB for GARNet+](https://drive.google.com/file/d/1vQhMfzPajtFrxJsoM3JkJH-Hsoq_xTKD/view?usp=share_link)

## Prerequisites

#### Clone the Code Repository

```
git clone https://github.com/GaryZhu1996/GARNet.git
```

#### Install Python Denpendencies

```
cd GARNet
conda env create -f environment.yml
```


## 3D Reconstruction Model

For training, please use the following command:

```
python runner.py
```

For testing, please follow the steps below:

1. Update the setting of `__C.CONST.WEIGHTS` in `config.py` as the path of the reconstruction model;
2. Run the following command:
```
python runner.py --test --score_only
```


## View-Reduction Approach

For training IMB, please follow the steps below:

1. Update the setting of `inference_model_path` in `core/train_imb.py` as the path of the reconstruction model;
2. Run the following command:
```
python runner_train_imb.py
```

For testing with view-reduction approach, please follow the steps below:

1. Update the setting of `__C.CONST.WEIGHTS` in `config.py` as the path of the reconstruction model;
2. Update the setting of `__C.CONST.IMB_WEIGHTS` in `config.py` as the path of the IMB model;
3. Run the following command:
```
python runner.py --test --score_only
```


## License

This project is open sourced under MIT license.
