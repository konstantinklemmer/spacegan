# SpaceGAN - A generative adverserial net for geospatial point data

This repository provides complementary code and data for the paper "Augmenting Correlation Structures in Spatial Data Using Deep Generative Models" ([arXiv:1905.09796](https://arxiv.org/abs/1905.09796)).

*SpaceGAN* applies a conditional GAN (CGAN) with neighbourhood conditioning to learn local spatial autocorrelation structures.
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/konstantinklemmer/spacegan/master/img/img1.png)  |  ![](https://raw.githubusercontent.com/konstantinklemmer/spacegan/master/img/img2.png)

*SpaceGAN* further uses a new selection criterion measuring the loss in local spatial autocorrelation captured by the model to select the best performing network.
![](https://raw.githubusercontent.com/konstantinklemmer/spacegan/master/img/img3.png)

## Structure

The `src` folder contains the raw *SpaceGAN* codebase and utility functions. The folder `data` contains the datasets used in the experiments.

## Interactive version

However we recommend to try out *SpaceGAN* using the interactive notebooks provided in the main folder. These support Google Colab and can be run here:
* [Experiment_01_Toy1](google.com)
* [Experiment_02_Toy2](google.com)

## Citation

```
@article{klemmer2019spacegan,
  title={Augmenting correlation structures in spatial data using deep generative models},
  author={Klemmer, Konstantin and Koshiyama, Adriano and Flennerhag, Sebastian},
  journal={arXiv preprint arXiv:1905.09796},
  year={2019}
}
```

