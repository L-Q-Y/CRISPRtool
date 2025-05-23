# CRISPRtool
A Deep Learning-based Adaptive Ensemble Method for Customized Optimal CRISPR sgRNA Design

![Framework](./Figures/ensemble_model.jpg)


## API Link [here](https://github.com/L-Q-Y/CRISPRtool/tree/main/crisprtool)

## What is CRISPRtool
The CRISPRtool is a deep learning-based adaptive ensemble method, integrating context-based sequence features with specific cell line characteristics to enhance the design of on-target sgRNAs within CRISPR/Cas9 and Cas12 systems (Figure a). CRISPRtool integrates six parallel deep learning-based models, including two previously proposed models (DeepCRISPR and Seq-deepCpf1) and four our customized models, Cas9/Cas12_SimpleRNN, Cas9/Cas12_BiLSTM, Cas9/Cas12_Attention, and Cas9/Cas12_Transformer (Figure b-e).  After each model is trained parallelly on training sets, when making prediction on test sets, CRISPRtool will adaptively select the best model by comprehensively considering the seven ensemble indicators, that is, assigning the same weight to each indicator and then multiplying its order among all models.

We provide this tutorial to guide researchers using CRISPRtool in Python. Moreover, we also provide a wet-lab-friendly website that incorporates our powerful predictors for personalized design of on-target sgRNAs for arbitrary gene in CRISPR/Cas9 and Cas12 at https://huggingface.co/spaces/LfOreVEr/CRISPRtool.

For more details, please refer to Qingyang Liu, Wentian Wang, Yueze Liu, Jain Akanksha1, Yimin Liu, and Lijun Cheng. CRISPRtool: a Deep Learning for Customized Optimal CRISPR sgRNA Design. Nature Communications. 2025 (under review).

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

We provide the trained models on the forder of [/saved_models](https://github.com/L-Q-Y/CRISPRtool/tree/main/saved_models), which allows users to make prediction on new data without training from scratch.

### For application 1:
Users can choose the model that best fits their data by running the code through the terminal as follows:

```
cd path/to/CRISPRtool
python crisprtool/model_selection.py --group cas12 --data data/Cas12/input_HT-1-2.csv --weights-dir saved_models/Cas12 --cutoff 60
```
```
### Output
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
    - Choose 'cas9' or 'cas12'
- data: The data path in [/data](https://github.com/L-Q-Y/CRISPRtool/tree/main/data) forder or user defined data path. Note that the data format should be a CSV file, consistent with the data format in [/data](https://github.com/L-Q-Y/CRISPRtool/tree/main/data). Here is the explanation of the meaning of each column in CSV file:
    - Column 1: Index.
    - Column 2: 23-bp length sgRNA target sequence in Cas9 or 34-bp length sgRNA target sequence in Cas12.
    - Column 3: Numeric sgRNA efficacy value.
- weights-dir: The path of pre-trained models.
    - Set 'saved_models/Cas9' if users's data is on Cas9 system or 'saved_models/Cas12' if the CRISPR system is Cas12.
- cutoff: The percentile cutoff for binarization evaluation.
    - For example, if the cutoff is set to 60, which means sgRNAs associated with the top 40% of efficacy are regard as active sgRNAs.



### For application 2:

Once the best model is chosen from above step, users can use this model to design sgRNAs for any specific genes using following command:
```
cd path/to/CRISPRtool
python crisprtool/sgrna_design.py --group cas12 --model Cas12_BiLSTM --gene FOXA1
```
The output is top 10 candidated sgRNAs printed out on termianl:
```
### Output
Top 10 sgRNA candidates:
Chrom    Start      End Strand      Transcript            Exon         Target sequence (5' to 3')                 gRNA  pred_Score
   14 37591089 37591122     -1 ENST00000250448 ENSE00000995292 GAAGTTTAATGATCCACAAGTGTATATATGAAAT AUGAUCCACAAGUGUAUAUA   90.653717
   14 37590422 37590455     -1 ENST00000250448 ENSE00000995292 TTAGTTTCTATGAGTGTATACCATTTAAAGAATT UAUGAGUGUAUACCAUUUAA   88.542549
   14 37589629 37589662     -1 ENST00000250448 ENSE00000995292 TTAATTTAACTACCTTTCCTCCTTCCCCAATGTA ACUACCUUUCCUCCUUCCCC   85.031464
   14 37594362 37594395     -1 ENST00000554607 ENSE00002524345 CCTATTTGGGGAGAAGTGTGCTCCTTCTCTAAAA GGGAGAAGUGUGCUCCUUCU   84.479034
   14 37594034 37594067     -1 ENST00000554607 ENSE00002524345 TACTTTTAAGACGTGGACAGAAAAATATAGGATC AGACGUGGACAGAAAAAUAU   82.826775
   14 37589935 37589968     -1 ENST00000250448 ENSE00000995292 TTTTTTTCACTTAACTAAATCCGAAGTGAATATT ACUUAACUAAAUCCGAAGUG   81.935814
   14 37590388 37590421     -1 ENST00000250448 ENSE00000995292 TTTTTTTCAGTAAAAGGGAATATTACAATGTTGG AGUAAAAGGGAAUAUUACAA   79.052406
   14 37594230 37594263     -1 ENST00000554607 ENSE00002524345 TATTTTTACAATGTGCACAAAAGGATTACAGGGA CAAUGUGCACAAAAGGAUUA   77.078995
   14 37590640 37590673     -1 ENST00000250448 ENSE00000995292 TATATTTACATAACATATAGAGGTAATAGATAGG CAUAACAUAUAGAGGUAAUA   75.877060
   14 37589653 37589686     -1 ENST00000250448 ENSE00000995292 GCCTTTTCACTACAAAATCAAATATTAATTTAAC ACUACAAAAUCAAAUAUUAA   74.779274
```

Alternatively, users can specify '--use-mutation' argument to design MDA-MB-231 cell line related sgRNAs with mutation infomation. The command and output show as follows. Note that the 'Is_mutation' column indicates whether there are any mutations in the target sequence. 
```
python crisprtool/sgrna_design.py --group cas9 --model Cas9_MultiHeadAttention --gene FOXA1 --use-mutation
```
```
### Output
Top 10 sgRNA candidates:
 Gene Chrom Strand    Start      Transcript            Exon Target sequence (5' to 3')                 gRNA  pred_Score  Is_mutation
FOXA1    14     -1 37591751 ENST00000545425 ENSE00002236311    CAGTGGGGCGACGGCGACAGGGG CAGUGGGGCGACGGCGACAG   72.893982        False
FOXA1    14     -1 37592600 ENST00000545425 ENSE00002236311    TACGAGCGGCAACATGACCCCGG UACGAGCGGCAACAUGACCC   72.249321        False
FOXA1    14     -1 37592600 ENST00000250448 ENSE00000995292    TACGAGCGGCAACATGACCCCGG UACGAGCGGCAACAUGACCC   71.779266        False
FOXA1    14     -1 37591751 ENST00000250448 ENSE00000995292    CAGTGGGGCGACGGCGACAGGGG CAGUGGGGCGACGGCGACAG   70.598946        False
FOXA1    14     -1 37591766 ENST00000250448 ENSE00000995292    CCAGACTCTGGACCACAGTGGGG CCAGACUCUGGACCACAGUG   70.583092        False
FOXA1    14     -1 37594265 ENST00000554607 ENSE00002524345    CCTTGGTGGCACGTTCATGGGGG CCUUGGUGGCACGUUCAUGG   70.485741        False
FOXA1    14     -1 37592600 ENST00000553751 ENSE00002500110    TACGAGCGGCAACATGACCCCGG UACGAGCGGCAACAUGACCC   70.253792        False
FOXA1    14     -1 37591766 ENST00000545425 ENSE00002236311    CCAGACTCTGGACCACAGTGGGG CCAGACUCUGGACCACAGUG   70.103401        False
FOXA1    14     -1 37592240 ENST00000250448 ENSE00000995292    GTACATCTCGCTCATCACCATGG GUACAUCUCGCUCAUCACCA   69.778252        False
FOXA1    14     -1 37594212 ENST00000545425 ENSE00002318063    AGGGAAAACCAGTTACAGGGAGG AGGGAAAACCAGUUACAGGG   69.409035        False
```

If users want to get the whole sgRNAs design result, you can specify '--save-csv' argument, which allows you are able to download the generated CSV file from the current directory. Deom command is as follows:
```
python crisprtool/sgrna_design.py --group cas9 --model Cas9_MultiHeadAttention --gene FOXA1 --use-mutation --save-csv FOXA1_design.csv
```

### Arguments:
- group: The system types of CRISPR that users are interested in.
    - Choose 'cas9' or 'cas12'
- model: The model recommended by model_selection.py, or any prefered model within the range of [DeepCRISPR,
    Cas9_BiLSTM,
    Cas9_SimpleRNN,
    Cas9_MultiHeadAttention,
    Cas9_Transformer,
    Cas12_BiLSTM,
    Cas12_SimpleRNN,
    Cas12_MultiHeadAttention,
    Cas12_Transformer,
    Seq_deepCpf1].
- gene: The gene you want to design.
    - Choose one gene symbol per time.
- use-mutation: If set, include mutation info from [/data/MDAMB231_mut](https://github.com/L-Q-Y/CRISPRtool/tree/main/data/MDAMB231_mut). If not set, only design sgRNAs based on reference sequences from Ensembl database.
    - Note that only set this argument if you are interested in designing with MDA-MB-231 related mutations. As for other cell lines, we haven't provided enough mutation info to design so far.
- save-csv: If set, the design result will be saved as a CSV file in the current directory for users to download. If not set, the script will only print out top 10 sgRNA candidates on terminal.



## Citation
Qingyang Liu, Wentian Wang, Yueze Liu, Jain Akanksha1, Yimin Liu, and Lijun Cheng. CRISPRtool: a Deep Learning for Customized Optimal CRISPR sgRNA Design. Nature Communications. 2025 (under review).











