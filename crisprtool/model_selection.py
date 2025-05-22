import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score
)

from model import (
    DeepCRISPR,
    Cas9_BiLSTM,
    Cas9_SimpleRNN,
    Cas9_MultiHeadAttention,
    Cas9_Transformer
)
from utils import PREPROCESS, PREPROCESS_for_DeepCRISPR



def load_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    x_test, y_test = PREPROCESS(lines)
    x_deepcrispr, _ = PREPROCESS_for_DeepCRISPR(lines)

    return {
        'DeepCRISPR':     x_deepcrispr,
        'Cas9_BiLSTM':    x_test,
        'Cas9_SimpleRNN': x_test,
        'Cas9_MultiHeadAttention': x_test,
        'Cas9_Transformer':       x_test
    }, y_test


def compute_metrics(y_true, pred_score, cutoff=60):
    y = y_true.flatten()
    p = pred_score.flatten()

    thr_y = np.percentile(y, cutoff)
    thr_p = np.percentile(p, cutoff)
    true_type = (y > thr_y).astype(int)
    pred_type = (p > thr_p).astype(int)

    mets = {}
    mets['Spearman']  = np.round(spearmanr(p, y).correlation, 4)
    mets['Accuracy']  = np.round(accuracy_score(true_type, pred_type), 4)
    mets['F1']        = np.round(f1_score(true_type, pred_type), 4)
    mets['Precision']= np.round(precision_score(true_type, pred_type), 4)
    mets['Recall']    = np.round(recall_score(true_type, pred_type), 4)
    mets['ROC_AUC']   = np.round(roc_auc_score(true_type, p), 4)
    mets['PR_AUC']    = np.round(average_precision_score(true_type, p), 4)

    return mets


def evaluate_all(data_path, weight_paths, cutoff, metric_weights=None):

    all_metrics = ['Spearman','Accuracy','F1','Precision','Recall','ROC_AUC','PR_AUC']
    if metric_weights is None:
        metric_weights = {m: 1.0 for m in all_metrics}

    x_dict, y_true = load_data(data_path)

    constructors = {
        'DeepCRISPR':            DeepCRISPR,
        'Cas9_BiLSTM':           Cas9_BiLSTM,
        'Cas9_SimpleRNN':        Cas9_SimpleRNN,
        'Cas9_MultiHeadAttention': Cas9_MultiHeadAttention,
        'Cas9_Transformer':      Cas9_Transformer
    }

    metrics_dict = {}

    for name, ctor in constructors.items():
        x_test = x_dict[name]
        if "DeepCRISPR" in name:
            shape = (1, 23, 4)
        else:
            shape = (23, 4)
        model = ctor(input_shape=shape)
        if name in weight_paths:
            model.load_weights(weight_paths[name])
        y_pred = model.predict(x_test, verbose=0)
        metrics_dict[name] = compute_metrics(y_true, y_pred, cutoff)

    df_metrics = pd.DataFrame(metrics_dict).T[all_metrics]
    df_ranks = df_metrics.rank(ascending=False, method='min')

    df_ranks['WeightedScore'] = sum(
        df_ranks[m] * metric_weights.get(m, 1.0)
        for m in all_metrics
    )

    df_ranks = df_ranks.sort_values('WeightedScore')
    best_model = df_ranks.index[0]

    return df_metrics, df_ranks, best_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate Cas9/Cas12 models and pick the best one"
    )
    parser.add_argument('--group',
                        choices=['cas9','cas12'],
                        required=True,
                        help="choose cas9 or cas12")
    parser.add_argument('--data', default='data/Cas9/Kim2019_test.csv')
    parser.add_argument('--weights-dir', default='saved_models/Cas9')
    parser.add_argument('--cutoff', type=float, default=60,
                        help="percentile cutoff for binarization")
    args = parser.parse_args()
    
    if args.group == 'cas9':
        weight_paths = {
            'DeepCRISPR':     f"{args.weights_dir}/DeepCRISPR_weights.keras",
            'Cas9_BiLSTM':    f"{args.weights_dir}/Cas9_BiLSTM_weights.keras",
            'Cas9_SimpleRNN': f"{args.weights_dir}/Cas9_SimpleRNN_weights.keras",
            'Cas9_MultiHeadAttention': f"{args.weights_dir}/Cas9_MultiHeadAttention_weights.keras",
            'Cas9_Transformer':       f"{args.weights_dir}/Cas9_Transformer_weights.keras"
        }
    else:
        weight_paths = {
            'DeepCRISPR_Cas12':     f"{args.weights_dir}/DeepCRISPR_Cas12.h5",
            'Cas12_BiLSTM':         f"{args.weights_dir}/Cas12_BiLSTM.h5",
            'Cas12_SimpleRNN':      f"{args.weights_dir}/Cas12_SimpleRNN.h5",
            'Cas12_MultiHeadAttention': f"{args.weights_dir}/Cas12_MultiHeadAttention.h5",
            'Cas12_Transformer':    f"{args.weights_dir}/Cas12_Transformer.h5"
        }
    
    metric_weights = {
        'Spearman': 1.0,
        'Accuracy': 1.0,
        'F1':       1.0, 
        'Precision':1.0,
        'Recall':   1.0,
        'ROC_AUC':  1.0,
        'PR_AUC':   1.0
    }

    df_m, df_r, best = evaluate_all(
        data_path=args.data,
        #group=args.group,
        weight_paths=weight_paths,
        cutoff=args.cutoff,
        metric_weights=metric_weights
    )

    print("=== Ensemble Metrics ===")
    print(df_m, "\n")
    print("=== Rank ===")
    print(df_r, "\n")
    print(">> The best model is: ", best)
