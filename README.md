# A Survey on Self-Supervised Learning
This repository provides a brief summary of algorithms from our review paper [A Survey of Self-Supervised Learning from Multiple Perspectives: Algorithms, Theory, Applications and Future Trends](https://arxiv.org/abs/2301.05712). 

SSL research breakthroughs in CV have been achieved in recent years.
In this work, we therefore mainly include SSL research derived from the CV community in recent years, especially classic and influential research results. The objectives of
this review are to explain what SSL is, its categories and subcategories, how it differs and relates to other machine learning paradigms, and its theoretical underpinnings. We
present an up-to-date and comprehensive review of the frontiers of visual SSL and divide visual SSL into three parts: context-based, contrastive, and generative SSL, in the hope
of sorting the trends for researchers.

See our paper for more details.

# Algorithms
## Context Based Methods
- **(Rotation):** Unsupervised representation learning by predicting image rotations.
\[[paper](https://openreview.net/forum?id=S1v4N2l0-)\]
\[[code](https://github.com/gidariss/FeatureLearningRotNet)\]

- **(Colorization):** Colorful Image Colorization.
\[[paper](https://arxiv.org/abs/1603.08511)\]
\[[code](https://github.com/richzhang/colorization)\]

- **(Jigsaw):** Scaling and Benchmarking Self-Supervised Visual Representation Learning.
\[[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf)\]
\[[code](https://github.com/facebookresearch/fair_self_supervision_benchmark)\]

<!--################################################################-->
## Contrastive Learning
- CL methods based on negative examples:
  - **(MoCo v1):** Momentum Contrast for Unsupervised Visual Representation Learning.
  \[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)\]
  \[[code](https://github.com/facebookresearch/moco)\]

  - **(MoCo v2):** Improved Baselines with Momentum Contrastive Learning.
  \[[paper](https://arxiv.org/pdf/2003.04297.pdf)\]
  \[[code](https://github.com/facebookresearch/moco)\]

  - **(MoCo v3):** An Empirical Study of Training Self-Supervised Vision Transformers.
  \[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)\]
  \[[code](https://github.com/facebookresearch/moco-v3)\]

  - **(SimCLR V1）:** A Simple Framework for Contrastive Learning of Visual Representations.
  \[[paper](https://arxiv.org/abs/2002.05709)\]
  \[[code](https://github.com/google-research/simclr)\]

  - **(SimCLR V2）:** Big Self-Supervised Models are Strong Semi-Supervised Learners.
  \[[paper](https://arxiv.org/abs/2006.10029)\]
  \[[code](https://github.com/google-research/simclr)\]

- CL methods based on self-distillation:
  - **(BYOL):** Bootstrap Your Own Latent
  A New Approach to Self-Supervised Learning.
  \[[paper](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)\]
  \[[code](https://github.com/deepmind/deepmind-research/tree/master/byol)\]

  - **(SimSiam):** Exploring Simple Siamese Representation Learning
  A New Approach to Self-Supervised Learning.
  \[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)\]
  \[[code](https://github.com/facebookresearch/simsiam)\]

- CL methods based on feature decorrelation:
  - **(Barlow Twins):** Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
  \[[paper](https://arxiv.org/pdf/2103.03230v3.pdf)\]
  \[[code](https://github.com/facebookresearch/barlowtwins)\]

  - **(VICReg):** Vicreg: Variance-invariancecovariance regularization for self-supervised learning.
  \[[paper](https://openreview.net/forum?id=xm6YD62D1Ub)\]
  \[[code](https://github.com/facebookresearch/vicreg)\]

- Others:
    - methods that combinate CL and MIM
    - ...
<!--################################################################-->
## Generative Algorithms
- **(BEiT):** Beit: Bert pre-training of image transformers.
\[[paper](https://openreview.net/pdf?id=p-BhZSz59o4)\]
\[[code](https://github.com/microsoft/unilm/tree/master/beit)\]

- **(MAE):** Masked Autoencoders Are Scalable Vision Learners.
\[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf)\]
\[[code](https://github.com/facebookresearch/mae)\]

- **(iBOT):** iBOT: Image BERT Pre-Training with Online Tokenizer.
\[[paper](https://arxiv.org/pdf/2111.07832v3.pdf)\]
\[[code](https://github.com/bytedance/ibot)\]

- **(CAE):** Context Autoencoder for Self-Supervised Representation Learning.
\[[paper](https://arxiv.org/pdf/2202.03026v2.pdf)\]
\[[code](https://github.com/open-mmlab/mmselfsup)\]

- **(SimMIM):** SimMIM: a Simple Framework for Masked Image Modeling.
\[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.pdf)\]
\[[code](https://github.com/microsoft/simmim)\]

# Applications
## 4.1 Sequential data
Natural language processing (NLP)
- **(Skip-Gram):** Distributed Representations of Words and Phrases
and their Compositionality.
\[[paper](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)\]
\[[code](https://github.com/graykode/nlp-tutorial)\]

- **(BERT):** BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding.
\[[paper](https://aclanthology.org/N19-1423.pdf)\]
\[[code](https://github.com/google-research/bert)\]

- **(GPT):** Improving Language Understanding
by Generative Pre-Training.
\[[paper](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)\]

Sequential models for image processing and computer vision
- **(CPC):** Representation learning with contrastive predictive coding.
\[[paper](https://arxiv.org/pdf/1807.03748v2.pdf)\]

- **(Image GPT):** Distributed Representations of Words and Phrases
and their Compositionality.
\[[paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)\]
\[[code](https://github.com/openai/image-gpt)\]

## 4.2 Image processing and computer vision
### video
- **(MIL-NCE):** End-to-End Learning of Visual Representations From Uncurated Instructional Videos.
\[[paper](https://arxiv.org/pdf/1912.06430v4.pdf)\]
\[[code](https://github.com/antoine77340/MIL-NCE_HowTo100M)\]

- Unsupervised Learning of Visual Representations using Videos.
\[[paper](https://arxiv.org/pdf/1505.00687.pdf)\]

- Unsupervised Learning of Video Representations using LSTMs.
\[[paper](https://arxiv.org/pdf/1502.04681v3.pdf)\]
\[[code](https://github.com/mansimov/unsupervised-videos)\]

#### 1. Temporal information in videos:
The order of the frames:
- Shuffle and Learn: Unsupervised Learning using Temporal Order Verification.
\[[paper](https://arxiv.org/pdf/1603.08561v2.pdf)\]

- Self-Supervised Video Representation Learning With Odd-One-Out Networks.
\[[paper](https://arxiv.org/pdf/1611.06646v4.pdf)\]

Video playing direction:
- Learning and Using the Arrow of Time.
\[[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Learning_and_Using_CVPR_2018_paper.pdf)\]

Video playing speed:
- **(SpeedNet):** SpeedNet: Learning the Speediness in Videos.
\[[paper](https://arxiv.org/pdf/2004.06130v2.pdf)\]

#### 2. Motion of objects such as optical flow:
- **(DynamoNet):** DynamoNet: Dynamic Action and Motion Network.
\[[paper](https://arxiv.org/pdf/1904.11407v1.pdf)\]

- **(CoCLR):** Self-supervised Co-training for Video Representation Learning.
\[[paper](https://arxiv.org/pdf/2010.09709v2.pdf)\]
\[[code](https://github.com/TengdaHan/CoCLR)\]

#### 3. Multi-modal(ality) data such as RGB, audio, and narrations
- Cooperative Learning of Audio and Video Models from Self-Supervised Synchronization.
\[[paper](https://arxiv.org/pdf/1807.00230v2.pdf)\]

- Time-Contrastive Networks: Self-Supervised Learning from Video.
\[[paper](https://arxiv.org/pdf/1704.06888v3.pdf)\]

#### 4. Spatial-temporal coherence of objects such as colours and shapes
- Learning Correspondence from the Cycle-Consistency of Time.
\[[paper](https://arxiv.org/pdf/1903.07593v2.pdf)\]

- **(VCP):** Video Cloze Procedure for Self-Supervised Spatio-Temporal Learning.
\[[paper](https://arxiv.org/pdf/2001.00294v1.pdf)\]

- Joint-task Self-supervised Learning for Temporal Correspondence.
\[[paper](https://arxiv.org/pdf/1909.11895v1.pdf)\]
\[[code](https://github.com/Liusifei/UVC)\]

## Other fields
- **medical field:** Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts.
\[[paper](https://arxiv.org/pdf/2109.04379v2.pdf)\]
\[[code](https://github.com/luchixiang/pcrl)\]

- **medical image segmentation:** Contrastive learning of global and local features for medical image segmentation with limited annotations.
\[[paper](https://arxiv.org/pdf/2006.10511v2.pdf)\]
\[[code](https://github.com/krishnabits001/domain_specific_cl)\]

- **3D medical image analysis:** Rubik’s Cube+: A self-supervised feature learning framework for 3D medical image analysis.
\[[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301109)\]

# Contact

If you have any suggestions or find our work helpful, feel free to contact us

Email: {guijie,tchen}@seu.edu.cn

