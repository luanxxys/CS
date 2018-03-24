# Generative Adversarial Networks

### What can GAN do ?

<img src="images/脸萌.PNG" width="75%" height="75%"/>

> Taigman et al., 2017

![](images/sunglass.PNG)

> Alec Radford  et al. 2015

<img src="images/forcesmile.jpg" width="55%" height="55%"/>

<img src="images/trump.PNG" width="75%" height="75%"/>

> genekogan@Twitter

<img src="images/faceturn.jpg" width="45%" height="45%"/>

> R. Huang et al. 2017

<img src="images/sky.PNG" width="73%" height="73%"/>

> Jun-Yan Zhu et al. 2016

![](images/horse.PNG)

> https://junyanz.github.io/CycleGAN/

### What is GAN ?

![](images/police.PNG)

![](images/system.PNG)

![](images/autocoder_1.PNG)

![](images/generator.PNG)

Generator Model

- Deep Belief Network (DBN)

- Stacked Auto-Encoder (SAE)

- Deep Boltzman Machine (DBM)

- Variational Auto-Encoder (VAE)

![](images/discriminator.PNG)

![](images/evaluation.PNG)

Demo - Generate cartoon faces

![100 rounds](images/100rounds.png)

![1,000 rounds](images/1000rounds.png)

![5,000 rounds](images/5000rounds.png)

![50,000 rounds](images/50000rounds.png)

> [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)

### Theory & Algorithm

![](images/mle_1.PNG)

![](images/mle_2.PNG)

![](images/nnframe.PNG)

![](images/minmax_expression.PNG)

The optimization process is a Minimax game problem, and the optimization goal is to achieve the Nash equilibrium

![](images/algorithm.png)

[GAN Math](https://sherlockliao.github.io/2017/06/20/gan_math/)

![](images/trainprocessing.PNG)

![](images/solution.png)

### PROs and CONs

PROs

- Markov chains are never needed

        only backprop is used to obtain gradients

- No inference is needed during learning

- A wide variety of functions can be incorporated into the model

- GAN can be combined with CNN and RNN

CONs

- There is no explicit representation of Pg(x)

- D must be synchronized well with G during training

### [Head First Generative Adversarial Networks](HeadFirstGAN_theory.pdf)

### There are more that GAN can do ...

[Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets)

> Ian Goodfellow et al., 2014

![](images/paper.PNG)

> github.com/hindupuravinash/the-gan-zoo as at June 10, 2017

Kinds of GANs

![](images/GANs.png)

[GAN Timeline](https://github.com/dongb5/GAN-Timeline)

![](images/turtle.jpg)

Text to Image

![](images/texttoimage.PNG)

> H. Zhang et al. 2016 (StackGAN)

Image to Image

![](images/bagcolor.jpg)

Single Image Super-Resolution

![](images/super.PNG)

> Christian Ledig et al. 2016

### Future Research on GANs

![](images/future.PNG)

### GAN on Speech Processing

[Robust Speech Recognition Using Generative Adversarial Networks](https://arxiv.org/abs/1711.01567)

[SEGAN: Speech Enhancement Generative Adversarial Network](https://arxiv.org/pdf/1703.09452.pdf)

- [Exploring Speech Enhancement with Generative Adversarial Networks for Robust Speech Recognition](https://arxiv.org/abs/1711.05747)

- [Conditional Generative Adversarial Networks for Speech Enhancement and Noise-Robust Speaker Verification](http://kom.aau.dk/~zt/doc/Michelsanti-Tan-cGAN-Interspeech2017.pdf)

- [Statistical Parametric Speech Synthesis Using Generative Adversarial Networks Under A Multi-task Learning Framework](https://arxiv.org/abs/1707.01670)

- [Statistical Parametric Speech Synthesis Incorporating Generative Adversarial Networks](https://arxiv.org/abs/1709.08041)
