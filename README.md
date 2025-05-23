# CRISPRtool
A Deep Learning-based Adaptive Ensemble Method for Customized Optimal CRISPR sgRNA Design

![Framework](./Figures/ensemble_model.jpg)


## API Link [here](https://github.com/L-Q-Y/CRISPRtool/code)

## What is CRISPRtool
We develop an adaptive ensemble model, CRISPRtool, which integrates context-based sequence features with specific cell line characteristics to enhance the design of on-target sgRNAs within CRISPR/Cas9 and Cas12 systems (Figure a). After each model is trained parallelly on training sets, when making prediction on test sets, CRISPRtool will adaptively select the best model by comprehensively considering the seven ensemble indicators, that is, assigning the same weight to each indicator and then multiplying its order among all models. CRISPRtool integrates six parallel deep learning-based models, including two previously proposed models (DeepCRISPR and Seq-deepCpf1) and four our customized models, Cas9/Cas12_SimpleRNN, Cas9/Cas12_BiLSTM, Cas9/Cas12_Attention, and Cas9/Cas12_Transformer (Figure b-e). 

We provide this tutorial to guide researchers using CRISPRtool in Python. Moreover, we also provide a wet-lab-friendly website that incorporates our powerful predictors for personalized design of on-target sgRNAs for arbitrary gene in CRISPR/Cas9 and Cas12 at https://huggingface.co/spaces/LfOreVEr/CRISPRtool.

## Features

The package allows for the following functions:

* Adaptively select the most suitable model based on users provided data.
* Custom design on-target sgRNAs given selected model.



## Contact

Feel free to contact [Dr. Lijun Cheng](https://medicine.osu.edu/find-faculty/non-clinical/biomedical-informatics/lijun-cheng) at Lijun.Cheng@osumc.edu for questions regarding the study. 



## Installation

### Creating from source

First clone this GitHub repository:
```
git clone https://github.com/L-Q-Y/CRISPRtool.git
```

Then, install desired packages:
```
pip install -r requirements.txt
```


### Requirements

The Python packages and their versions used in this project can be found within requirements.txt.


## Tutorial
The main applications of CRISPRtool include two parts:

* Application 1: Adaptively select the most suitable model based on users provided data.
* Application 2: Custom design on-target sgRNAs given selected model.

We provide the trained models on the forder of [/saved_models](https://github.com/L-Q-Y/CRISPRtool/saved_models), which allows users to make prediction on new data without training from scratch.

### For application 1:
Users can choose the model that best fits their data by running the code through the terminal as follows:

```
cd path/to/CRISPRtool
python crisprtool/model_selection.py --group cas12 --data data/Cas12/input_HT-1-2.csv --weights-dir saved_models/Cas12 --cutoff 60
```
```
############ output ############
=== Ensemble Metrics ===
                          Spearman  Accuracy      F1  Precision  Recall  ROC_AUC  PR_AUC
Seq_deepCpf1                0.7663    0.7879  0.7350     0.7350  0.7350   0.8740  0.7698
Cas12_BiLSTM                0.7683    0.7941  0.7427     0.7427  0.7427   0.8775  0.7698
Cas12_SimpleRNN             0.7336    0.7771  0.7215     0.7215  0.7215   0.8586  0.7515
Cas12_MultiHeadAttention    0.7276    0.7647  0.7060     0.7060  0.7060   0.8519  0.7460
Cas12_Transformer           0.7291    0.7709  0.7137     0.7137  0.7137   0.8540  0.7344 

=== Rank ===
                          Spearman  Accuracy   F1  Precision  Recall  ROC_AUC  PR_AUC  WeightedScore
Cas12_BiLSTM                   1.0       1.0  1.0        1.0     1.0      1.0     1.0            7.0
Seq_deepCpf1                   2.0       2.0  2.0        2.0     2.0      2.0     1.0           13.0
Cas12_SimpleRNN                3.0       3.0  3.0        3.0     3.0      3.0     3.0           21.0
Cas12_Transformer              4.0       4.0  4.0        4.0     4.0      4.0     5.0           29.0
Cas12_MultiHeadAttention       5.0       5.0  5.0        5.0     5.0      5.0     4.0           34.0 

>> The best model is:  Cas12_BiLSTM
```

### Arguments:
- group: The system types of CRISPR that users are interested in.
    - choose 'cas9' or 'cas12'
- data: The data path in [/data](https://github.com/L-Q-Y/CRISPRtool/data) forder or user defined data path. Note that the data format should be a CSV file, consistent with the data format in [/data](https://github.com/L-Q-Y/CRISPRtool/data).
    - Column 1: Index.
    - Column 2: 23-bp length sgRNA target sequence in Cas9 or 34-bp length sgRNA target sequence in Cas12.
    - Column 3: Numeric sgRNA efficacy value.
- weights-dir: The path of pre-trained models, eg. in [/saved_models](https://github.com/L-Q-Y/CRISPRtool/saved_models).
    - If users want to select best model for Cas9 system, they can set the path to saved_models/Cas9.
- cutoff: The percentile cutoff for binarization evaluation.
    - For example, if the cutoff is set to 60, which means sgRNAs associated with the top 40% of efficacy are regard as active sgRNAs.



### For application 2:








## Citation
Qingyang Liu, Wentian Wang, Yueze Liu, Jain Akanksha1, Yimin Liu, and Lijun Cheng. CRISPRtool: a Deep Learning for Customized Optimal CRISPR sgRNA Design. Nature Communications. 2025 (under review)











