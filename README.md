# STr-GCN: Dual Spatial Graph Convolutional Network and Transformer Graph Encoder for 3D Hand Gesture Recognition

This repository holds the Pytorch implementation of STr-GCN: Dual Spatial Graph Convolutional Network and Transformer Graph Encoder for 3D Hand Gesture Recognition by Anonymous.

## Introduction

we propose a new deep learning architecture for hand gesture recognition using 3D hand skeleton data and we call STr-GCN. It decouples the spatial and temporal learning of the gesture by leveraging Graph Convolutional Networks (GCN) and Transformers. The key idea is to combine two powerful networks: a Spatial Graph Convolutional Network unit that understands intra-frame interactions to extract powerful features from different hand joints and a Transformer Graph Encoder which is based on a Temporal Self-Attention module to incorporate inter-frame correlations. The code for training our approach is provided in this repository. Training is possible on the [SHREC’17 Track Dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/), the [Briareo dataset](https://guiggh.github.io/publications/first-person-hands/) and the [FPHA dataset](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=31). 
<p align="center"><img src="figures/fig1.jpg" alt="" width="1000"></p>

### Prerequisites

This package has the following requirements:

* `Python 3.9`
* `Pytorch v1.11.0`

### Training
1. Download the [SHREC’17 Track Dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/), the extracted 3D skeleton data of [Briareo dataset](https://drive.google.com/drive/folders/17CGLnI5emX1y6iOdqHygblujz0FFlrc8?usp=sharing) and the extracted 3D skeleton data of [FPHA dataset](https://drive.google.com/file/d/1PdrGPGVlG_R7TP3IO8rNeY4pSRCNcQXu/view?usp=sharing). <br/>
- The download of SHREC'17 is handled by our script.
- Download the Briareo extracted data from this [link](https://drive.google.com/drive/folders/17CGLnI5emX1y6iOdqHygblujz0FFlrc8?usp=sharing) and place it a folder named 'Briareo_data'.
- Download the FPHA extracted data from this [link](https://drive.google.com/file/d/1PdrGPGVlG_R7TP3IO8rNeY4pSRCNcQXu/view?usp=sharing) and extract the content of the zip into a folder named 'FPHA_data'.

2. Set the path for saving your trained models in the  ```train.py (line 126) ```.

3. The dataset used for training is set by default to ```SHREC```, you can change that in main.py into ```BRIAREO``` or ```FPHA```.

4. Run the following command.
```
python train.py     
```
<!-- ### Citation
If you find this code useful in your research, please consider citing:
```

``` -->
## Acknowledgement

Part of our code was inspired by the [The ST-GCN implementation](https://github.com/yysijie/st-gcn) and [pytorch implementation of Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html). We thank to the authors for releasing their codes.