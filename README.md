# CRISPRtool
A Deep Learning-based Ensemble Method for Customized Optimal CRISPR sgRNA Design

![Framework](./Figures/ensemble_model.jpg)


## API Link [here](https://github.com/L-Q-Y/CRISPRtool/code)

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

First clone this Github repository:
```
git clone https://github.com/L-Q-Y/CRISPRtool.git
```

Then, install desired packages:
```
cd path/to/CRISPRtool/
pip install -r requirements.txt
```



### Requirements

The Python packages and their versions used in this project can be found within requirements.txt.


## Demo Usage
The main applications of CRISPRtool include two parts:

* Adaptively select the most suitable model based on user provided data
* Customizely design on-target sgRNAs given selected model

### For application 1:
Users can choose the model that best fits their data by running the code through the terminal as follows:
```
python crisprtool/model_selection.py --group cas9 --data data/Cas9/Kim2019_test.csv --weights-dir saved_models/Cas9 --cutoff 80

############ output ############
=== Ensemble Metrics ===
                         Spearman  Accuracy      F1  Precision  Recall  ROC_AUC  PR_AUC
DeepCRISPR                -0.0219    0.6790  0.2018     0.2018  0.2018   0.4945  0.2115
Cas9_BiLSTM                0.7325    0.8081  0.5229     0.5229  0.5229   0.8459  0.5411
Cas9_SimpleRNN             0.7693    0.8303  0.5780     0.5780  0.5780   0.8684  0.5811
Cas9_MultiHeadAttention    0.7509    0.8376  0.5963     0.5963  0.5963   0.8620  0.5770
Cas9_Transformer           0.7201    0.8155  0.5413     0.5413  0.5413   0.8338  0.5539 

=== Rank ===
                         Spearman  Accuracy   F1  Precision  Recall  ROC_AUC  PR_AUC  WeightedScore
Cas9_MultiHeadAttention       2.0       1.0  1.0        1.0     1.0      2.0     2.0           10.0
Cas9_SimpleRNN                1.0       2.0  2.0        2.0     2.0      1.0     1.0           11.0
Cas9_Transformer              4.0       3.0  3.0        3.0     3.0      4.0     3.0           23.0
Cas9_BiLSTM                   3.0       4.0  4.0        4.0     4.0      3.0     4.0           26.0
DeepCRISPR                    5.0       5.0  5.0        5.0     5.0      5.0     5.0           35.0 

>> The best model is:  Cas9_MultiHeadAttention
```

### For application 2:



We provide the trained models on the forder of [/saved_models](https://github.com/L-Q-Y/CRISPRtool/saved_models), which allows user to make prediction on new data without training from scratch.


## Citation
Qingyang Liu, Wentian Wang, Yueze Liu, Jain Akanksha1, Yimin Liu, and Lijun Cheng. CRISPRtool: a Deep Learning for Customized Optimal CRISPR sgRNA Design. Nature Communications. 2025 (under review)











