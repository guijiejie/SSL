# A Survey on Self-Supervised Learning
This repository provides a summary of algorithms from our review paper[链接](). 

We classify self-supervised algorithms into different pretext tasks. A popular solution of SSL is to propose a pretext task for networks to solve and networks will be trained by learning objective functions of pretext tasks. The features are learned through this process.
Existing pretext tasks can be roughly classified into three categories: context based, contrastive learning (CL), and generative algorithms.
 
See our paper for more details.
If you have any suggestions or find our work helpful, feel free to contact us(Email: tchen@seu.edu.cn). 

We supply the bibtex file of our paper.

# Algorithms
## Context Based Methods
* **Rotation:** Unsupervised representation learning by predicting image rotations.
\[[paper](https://openreview.net/forum?id=S1v4N2l0-)\]
\[[code](https://github.com/gidariss/FeatureLearningRotNet)\]

* **Colorization:** Colorful Image Colorization.
\[[paper](https://arxiv.org/abs/1603.08511)\]
\[[code](https://github.com/richzhang/colorization)\]

* **Jigsaw:** Scaling and Benchmarking Self-Supervised Visual Representation Learning.
\[[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf)\]
\[[code](https://github.com/facebookresearch/fair_self_supervision_benchmark)\]

<!--################################################################-->
## Contrastive Learning
* **MOCO v1:** Momentum Contrast for Unsupervised Visual Representation Learning.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)\]
\[[code](https://github.com/facebookresearch/moco)\]

* **SIMCLR V1:** A Simple Framework for Contrastive Learning of Visual Representations.
\[[paper](https://arxiv.org/abs/2002.05709)\]
\[[code](https://github.com/google-research/simclr)\]

* **BYOL:** Bootstrap Your Own Latent
A New Approach to Self-Supervised Learning.
\[[paper](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)\]
\[[code](https://github.com/deepmind/deepmind-research/tree/master/byol)\]

* **SimSiam:** Exploring Simple Siamese Representation Learning
A New Approach to Self-Supervised Learning.
\[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)\]
\[[code](https://github.com/facebookresearch/simsiam)\]

* **Barlow Twins:** Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
\[[paper](https://arxiv.org/pdf/2103.03230v3.pdf)\]
\[[code](https://github.com/facebookresearch/barlowtwins)\]

* **VICReg:** Vicreg: Variance-invariancecovariance regularization for self-supervised learning.
\[[paper](https://openreview.net/forum?id=xm6YD62D1Ub)\]
\[[code](https://github.com/facebookresearch/vicreg)\]

<!--################################################################-->
## Generative Algorithms
* **Beit:** Beit: Bert pre-training of image transformers.
\[[paper](https://openreview.net/pdf?id=p-BhZSz59o4)\]
\[[code](https://github.com/microsoft/unilm/tree/master/beit)\]

* **MAE:** Masked Autoencoders Are Scalable Vision Learners.
\[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf)\]
\[[code](https://github.com/facebookresearch/mae)\]

* **CAE:** Context Autoencoder for Self-Supervised Representation Learning.
\[[paper](https://arxiv.org/pdf/2202.03026v2.pdf)\]
\[[code](https://github.com/open-mmlab/mmselfsup)\]

* **SimMIM:** SimMIM: a Simple Framework for Masked Image Modeling.
\[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.pdf)\]
\[[code](https://github.com/microsoft/simmim)\]

# Applications
## Sequential data
Natural language processing (NLP)
* **Skip-Gram:** Distributed Representations of Words and Phrases
and their Compositionality.
\[[paper](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)\]
\[[code](https://github.com/graykode/nlp-tutorial)\]

* **BERT:** BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding.
\[[paper](https://aclanthology.org/N19-1423.pdf)\]
\[[code](https://github.com/google-research/bert)\]

Sequential models for image processing and computer vision
## Image processing and computer vision


## Other fields

