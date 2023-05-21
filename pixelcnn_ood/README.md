## Shaken, and Stirred: Long-Range Dependencies Enable Robust Outlier Detection with PixelCNN++

This repository contains the code for the paper:

**Shaken, and Stirred: Long-Range Dependencies Enable Robust Outlier Detection with PixelCNN++** <br>
Barath Mohan Umapathi, Kushal Chauhan, Pradeep Shenoy, Devarajan Sridharan <br>
*The 32nd International Joint Conference on Artificial Intelligence (IJCAI 2023)* <br>

### Preparation

#### Environment Setup
1. Creare a new conda environment `conda create -n pixelcnn_ood python=3.7.10`.
2. Activate it `conda activate pixelcnn_ood`.
3. Install the requirements `python3 -m pip install -r pixelcnn_ood/requirements.txt`.

#### Datasets Preparation
The SignLang, CompCars, GTSRB, CLEVR, and CelebA datasets need to be downloaded manually. Follow the instructions below:
1. Download the SignLang dataset from [Kaggle](https://www.kaggle.com/ash2703/handsignimages) (requires Kaggle account) and extract `Hand_sign_mnist.zip` into `pixelcnn_ood/datasets/sign_lang/`.
2. Download the CompCars dataset from its [source](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) and extract `sv_data.zip` into `pixelcnn_ood/datasets/compcars/`.
3. For GTSRB, download `GTSRB_Final_Training_Images.zip` and `GTSRB_Final_Test_Images.zip` from its [source](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html) and extract the contents into `pixelcnn_ood/datasets/GTSRB`.
4. Download CLEVR dataset from its [source](https://cs.stanford.edu/people/jcjohns/clevr/) and extract `CLEVR_v1.0.zip` into `pixelcnn_ood/datasets/clevr`.
5. Download CelebA dataset from its [source](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and extract `img_celeba.7z` into `pixelcnn_ood/datasets/celeb_a`.

### Usage

Train a PixelCNN++ on MNIST dataset and perform inference on other grayscale OOD datasets:

```
python3 -m train_pixelcnn.py --train_set mnist --mode grayscale
```

Compute "Stirred" LL using PixelCNN++ trained on FashionMNIST:

```
python3 -m stirring_ll.py --train_set fashion_mnist --mode grayscale
```

Compute "Shaken" LL using PixelCNN++ trained on CelebA:

```
python3 -m shaking_ll.py --train_set celeb_a --mode color
```

Train a background PixelCNN++ on MNIST and compute Log-Likelihood ratio from [Ren et al., 2019](https://arxiv.org/pdf/1906.02845):

```
python3 -m train_pixelcnn.py --train_set mnist --mode grayscale --bg 1
```

Compute Input Complexity OOD score from [Serra et al., 2020](https://arxiv.org/pdf/1909.11480) using PNG compression for LSUN/ID:

```
python3 -m probs_ic.py --train_set lsun --mode color --compression png
```

### Citation

If you find our methods useful, please cite:

```
@inproceedings{umapathi2022,
  title={Shaken, and Stirred: Long-Range Dependencies Enable Robust Outlier Detection with PixelCNN++},
  author={Umapathi, Barath Mohan and Chauhan, Kushal and Shenoy, Pradeep and Sridharan, Devarajan},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  year={2023},
  organization={IJCAI}
}
```