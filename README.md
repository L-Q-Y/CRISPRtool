# CRISPRtool
A Deep Learning-based Ensemble Method for Customized Optimal CRISPR sgRNA Design

![Framework](./Figures/ensemble_model.jpg)


## API Link [here](https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/docs/API.md)

## What is CRISPRtool
We develop an adaptive ensemble model, CRISPRtool, which utilizes context-based sequence features to enhance the design of on-target sgRNAs within CRISPR/Cas9 and Cas12 systems (Figure 2(a)). After each model is trained parallelly on training sets, when making prediction on test sets, CRISPRtool will adaptively select the best model by comprehensively considering the seven ensemble indicators, that is, assigning the same weight to each indicator and then multiplying its order among all models. CRISPRtool integrates six parallel deep learning-based models, including two previously proposed models (DeepCRISPR and Seq-deepCpf1) and four our customized models (Cas9/Cas12_SimpleRNN, Cas9/Cas12_BiLSTM, Cas9/Cas12_Attention, and Cas9/Cas12_Transformer). Following subsections describe the details of the architecture of four customized deep learning-based models, seven ensemble metrics, and the implementation of CRISPRtool.

## Features



The package allows for the following functions:

* Data Preprocessing
* Training
* Validation
* Calibration
* Dimensionality Reduction and Clustering
* Cell Type Classification, and Identification of New Cell Types
* Visualization of Cell Types on 2D and 3D Plots. 



## Contact

Feel free to contact [Dr. Lijun Cheng](https://medicine.osu.edu/find-faculty/non-clinical/biomedical-informatics/lijun-cheng) at Lijun.Cheng@osumc.edu for questions regarding the study. 



## Installation

### Creating from source


```
git clone https://github.com/zhushijia/STIE.git
```

### Requirements

The Python packages and their versions used in this project can be found within requirements.txt.


## Demo Usage

After creating your desired environment, you can run the demo case study followed by this tutorial:

```

```













