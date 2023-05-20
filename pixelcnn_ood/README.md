### Shaken, and Stirred: Long-Range Dependencies Enable Robust Outlier Detection with PixelCNN++

This repository contains the code for the paper:

**Shaken, and Stirred: Long-Range Dependencies Enable Robust Outlier Detection with PixelCNN++** <br>
Barath Mohan Umapathi, Kushal Chauhan, Pradeep Shenoy, Devarajan Sridharan <br>
*THE 32nd INTERNATIONAL JOINT CONFERENCE ON ARTIFICIAL INTELLIGENCE (IJCAI 2023)* <br>

#### Preparing
1. Creare a new conda environment `conda create -n pixelcnn_ood python=3.7.10`.
2. Activate it `conda activate pixelcnn_ood`.
3. Install the requirements `pip install -r pixelcnn_ood/requirements.txt`.
4. SignLang, CompCars, GTSRB, CLEVR, and CelebA datasets have to be downloaded manually.
	- Download the SignLang dataset from [Kaggle](https://www.kaggle.com/ash2703/handsignimages) (requires Kaggle account) and extract `Hand_sign_mnist.zip` in `pixelcnn_ood/datasets/sign_lang/`.
	- Download the CompCars dataset from its [source](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) and extract `sv_data.zip` in `pixelcnn_ood/datasets/compcars/`.
	- For GTSRB, download `GTSRB_Final_Training_Images.zip` and `GTSRB_Final_Test_Images.zip` from its [source](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html) and extract the contents in `pixelcnn_ood/datasets/GTSRB`.
	- Download CLEVR dataset from its [source](https://cs.stanford.edu/people/jcjohns/clevr/) and extract `CLEVR_v1.0.zip` in `pixelcnn_ood/datasets/clevr`.
	- Download CelebA dataset from its [source](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and extract `img_celeba.7z` in `pixelcnn_ood/datasets/celeb_a`.

#### Usage

To train a PixelCNN++ on MNIST dataset and perform inference on other grayscale OOD datasets, run

```
python3 -m train_pixelcnn.py --train_set mnist --mode grayscale
```

To compute "Stirred" LL for FashionMNIST/ID and other grayscale OOD datasets, run

```
python3 -m stirring_ll.py --train_set fashion_mnist --mode grayscale
```

To compute "Shaken" LL for CelebA/ID and other natural image OOD datasets, run

```
python3 -m shaking_ll.py --train_set celeb_a --mode color
```

To train a background PixelCNN++ on MNIST and compute Log-Likelihood ratio from [Ren et al., 2019](https://arxiv.org/pdf/1906.02845), run

```
python3 -m train_pixelcnn.py --train_set mnist --mode grayscale --bg 1
```

To compute Input Complexity OOD score from [Serra et al., 2019](https://arxiv.org/pdf/1909.11480) using PNG compression for LSUN/ID, run

```
python3 -m probs_ic.py --train_set lsun --mode color --compression png
```

#### Citation

If you find our methods useful, please cite:

```
@article{umapathi2022,
  title={Shaken, and Stirred: Long-Range Dependencies Enable Robust Outlier Detection with PixelCNN++},
  author={Umapathi, Barath Mohan and Chauhan, Kushal and Shenoy, Pradeep and Sridharan, Devarajan},
  journal={arXiv preprint arXiv:2208.13579},
  year={2022}
}
```