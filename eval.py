import numpy as np
import pandas as pd
import copy
import math
import os
import torch
import sys
from sklearn.metrics import average_precision_score, roc_auc_score
import scipy.sparse

def get_evaluation_matrix(TF_list, gene_list):
    """
    TF (columns) -> all genes (rows) evaluation matrix
    1 means TF->gene is evaluated, 0 otherwise
    """

    num_genes = len(gene_list)
    Evaluate_Mask = np.zeros((num_genes, num_genes), dtype=int)

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    for tf in TF_list:
        if tf in gene_to_idx:
            j = gene_to_idx[tf]          # column index (TF)
            Evaluate_Mask[j, :] = 1      # TF - all genes (sender - receiver)

    return Evaluate_Mask


def get_truth_edges(GT_path):
    """
    not giving specific Evaluate_Mask for each cell, only return the truth_edges
    """
    Ground_Truth = pd.read_csv(GT_path, index_col=0)
    # print(f"Ground_Truth: {Ground_Truth.head(5)}")
    
    nonzero_indices = np.where(Ground_Truth.values != 0)
    # print(f"nonzero_indices: {nonzero_indices}")
    # Convert numeric indices back to index/column labels
    # row, column == receptor, sender
    truth_edges = [(int(Ground_Truth.index[row])-1, int(Ground_Truth.columns[col])-1) for row, col in zip(*nonzero_indices)]
    # print(f"truth_edges: {truth_edges}")

    return list(set(truth_edges))


def eval_scGRN_CeSpGRN(result_matrix, cell_name_list, gt_scGRN_folder, TF_list, gene_list):
    """
    cell_name_list: list of cell got from normalized count index
    gt_scGRN_folder: gt scGRN path
    """

    TP_num_list = []
    EP_rate_list = []
    EPR_list = []

    AUPRC_list = []
    AUPRC_rate_list = []
    AUROC_list = []

    Evaluate_Mask = get_evaluation_matrix(TF_list, gene_list)

    for i, cell_name in enumerate(cell_name_list):
        print(i)
        pre_GRN_i = result_matrix[i]
        truth_edges = get_truth_edges(os.path.join(gt_scGRN_folder, f"{cell_name}.csv"))

        TP_num, EP_rate, EPR = evaluate(pre_GRN_i, truth_edges, Evaluate_Mask=Evaluate_Mask, cutoff_factor=1)
        TP_num_list.append(TP_num)
        EP_rate_list.append(EP_rate)
        EPR_list.append(EPR)

        auprc, auprc_ratio, auroc = get_AUPRC_AUROC(pre_GRN_i, truth_edges, Evaluate_Mask=Evaluate_Mask)
        AUPRC_list.append(auprc)
        AUPRC_rate_list.append(auprc_ratio)
        AUROC_list.append(auroc)
        

    ave_TP_num = sum(TP_num_list) / len(TP_num_list)
    ave_EP_rate = sum(EP_rate_list) / len(EP_rate_list)
    ave_EPR = sum(EPR_list) / len(EPR_list) 

    ave_AUPRC = sum(AUPRC_list) / len(AUPRC_list)
    ave_AUPRC_rate = sum(AUPRC_rate_list) / len(AUPRC_rate_list)
    ave_AUROC = sum(AUROC_list) / len(AUROC_list) 
    
    print(f"CeSpGRN scGRN eval: TP {ave_TP_num}, EP {ave_EP_rate}, ave_EPR {ave_EPR}")
    print(f"scGRN eval: AUPRC: {ave_AUPRC}, AUPRC_ratio: {ave_AUPRC_rate}, AUROC: {ave_AUROC}")
    return ave_TP_num, ave_EP_rate, ave_EPR

def eval_scGRN_grnboost2(result_df, num_genes, cell_name_list, gt_scGRN_folder, TF_list, gene_list):
    """
    cell_name_list: list of cell got from normalized count index
    gt_scGRN_folder: gt scGRN path
    """

    TP_num_list = []
    EP_rate_list = []
    EPR_list = []

    AUPRC_list = []
    AUPRC_rate_list = []
    AUROC_list = []

    Evaluate_Mask = get_evaluation_matrix(TF_list, gene_list)

    # have one predicted GRN matrix    
    result_df['TF'] = result_df['TF'].astype(int) - 1 # change to index
    result_df['target'] = result_df['target'].astype(int) - 1 # change to index
    pre_GRN_i = np.zeros((num_genes, num_genes), dtype=float)
    pre_GRN_i[result_df['TF'], result_df['target']] = result_df['importance'].values

    for i, cell_name in enumerate(cell_name_list):
        print(i)

        truth_edges = get_truth_edges(os.path.join(gt_scGRN_folder, f"{cell_name}.csv"))
        TP_num, EP_rate, EPR = evaluate(pre_GRN_i, truth_edges, Evaluate_Mask=Evaluate_Mask, cutoff_factor=1)
        TP_num_list.append(TP_num)
        EP_rate_list.append(EP_rate)
        EPR_list.append(EPR)

        auprc, auprc_ratio, auroc = get_AUPRC_AUROC(pre_GRN_i, truth_edges, Evaluate_Mask=Evaluate_Mask)
        AUPRC_list.append(auprc)
        AUPRC_rate_list.append(auprc_ratio)
        AUROC_list.append(auroc)
        

    ave_TP_num = sum(TP_num_list) / len(TP_num_list)
    ave_EP_rate = sum(EP_rate_list) / len(EP_rate_list)
    ave_EPR = sum(EPR_list) / len(EPR_list) 

    ave_AUPRC = sum(AUPRC_list) / len(AUPRC_list)
    ave_AUPRC_rate = sum(AUPRC_rate_list) / len(AUPRC_rate_list)
    ave_AUROC = sum(AUROC_list) / len(AUROC_list) 
    
    print(f"CeSpGRN scGRN eval: TP {ave_TP_num}, EP {ave_EP_rate}, ave_EPR {ave_EPR}")
    print(f"scGRN eval: AUPRC: {ave_AUPRC}, AUPRC_ratio: {ave_AUPRC_rate}, AUROC: {ave_AUROC}")
    return ave_TP_num, ave_EP_rate, ave_EPR


def eval_scGRN_SVGRN(result_path, cell_name_list, gt_scGRN_folder, TF_list, gene_list):
    """
    cell_name_list: list of cell got from normalized count index
    gt_scGRN_folder: gt scGRN path
    """

    TP_num_list = []
    EP_rate_list = []
    EPR_list = []

    AUPRC_list = []
    AUPRC_rate_list = []
    AUROC_list = []

    Evaluate_Mask = get_evaluation_matrix(TF_list, gene_list)

    for i, cell_name in enumerate(cell_name_list):
        print(cell_name)
        csv_path = os.path.join(result_path, f"{cell_name}", "RN_150.csv")
        if not os.path.exists(csv_path): #####
            continue
        
        pre_GRN_i = pd.read_csv(csv_path).values
        pre_GRN_i = pre_GRN_i.T   ###### transpose to make row as sender and column as receiver ######
        print(f"pre_GRN_i shape: {pre_GRN_i.shape}")

        truth_edges = get_truth_edges(os.path.join(gt_scGRN_folder, f"{cell_name}.csv"))

        TP_num, EP_rate, EPR = evaluate(pre_GRN_i, truth_edges, Evaluate_Mask=Evaluate_Mask, cutoff_factor=1)
        TP_num_list.append(TP_num)
        EP_rate_list.append(EP_rate)
        EPR_list.append(EPR)

        auprc, auprc_ratio, auroc = get_AUPRC_AUROC(pre_GRN_i, truth_edges, Evaluate_Mask=Evaluate_Mask)
        AUPRC_list.append(auprc)
        AUPRC_rate_list.append(auprc_ratio)
        AUROC_list.append(auroc)
        

    ave_TP_num = sum(TP_num_list) / len(TP_num_list)
    ave_EP_rate = sum(EP_rate_list) / len(EP_rate_list)
    ave_EPR = sum(EPR_list) / len(EPR_list) 

    ave_AUPRC = sum(AUPRC_list) / len(AUPRC_list)
    ave_AUPRC_rate = sum(AUPRC_rate_list) / len(AUPRC_rate_list)
    ave_AUROC = sum(AUROC_list) / len(AUROC_list) 
    
    print(f"SVGRN scGRN eval: TP {ave_TP_num}, EP {ave_EP_rate}, ave_EPR {ave_EPR}")
    print(f"scGRN eval: AUPRC: {ave_AUPRC}, AUPRC_ratio: {ave_AUPRC_rate}, AUROC: {ave_AUROC}")
    return ave_TP_num, ave_EP_rate, ave_EPR


def evaluate(A, truth_edges, Evaluate_Mask=None, cutoff_factor = 1):
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)

    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones_like(A) - np.eye(len(A))
    A = A * Evaluate_Mask

    A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0])
    A_val.reverse()

    cutoff_all = A_val[num_truth_edges * cutoff_factor - 1]
    
    if cutoff_all == 0 and all(x == 0 for x in A_val):
        ###### case happened in locCSN ########
        print("All values are 0")
        cutoff_all = 0.1 # all values are zero, no edge predicted
    elif cutoff_all == 0 and all(x >= 0 for x in A_val):
        # print("All values >= 0, and cutoff_all==0")
        ##### ! avoid the case cutoff_all is 0 because lots of the predicted values are 0 (happened in CespGRN) ####
        cutoff_all = min(x for x in A_val if x != 0) # renew the cutoff value as the smallest nonzero value in the list
    # print(f"cutoff_all {cutoff_all}")
    elif all(x >= cutoff_all for x in A_val):
        raise ValueError("something wrong with the cutoff_all value")

    A_indicator_all = np.zeros([num_nodes, num_nodes])
    A_indicator_all[abs(A) >= cutoff_all] = 1

    idx_send, idx_rec = np.where(A_indicator_all)
    A_edges = set(zip(idx_rec, idx_send))
    overlap_A = A_edges.intersection(truth_edges)

    # true positive value, EP rate, EPR
    return len(overlap_A), 1. * len(overlap_A) / num_truth_edges, 1. * len(overlap_A) / ((num_truth_edges ** 2) / np.sum(Evaluate_Mask))

def get_AUPRC_AUROC(A, truth_edges, Evaluate_Mask=None):
    """
    A: predicted GRN matrix (n_genes x n_genes)
    truth_edges: list of directed edges (receptor, sender) where sender -> receptor
                i.e., (receptor, sender)
    Evaluate_Mask: binary matrix (n_genes x n_genes). Evaluate only where mask==1.

    return: auprc, auprc_ratio, auroc
    """

    n_genes = A.shape[0]
    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones((n_genes, n_genes), dtype=bool)
        np.fill_diagonal(Evaluate_Mask, 0)
    else:
        Evaluate_Mask = Evaluate_Mask.astype(bool)

    # Build ground-truth matrix
    truth_matrix = np.zeros((n_genes, n_genes), dtype=np.int8)
    for receptor, sender in truth_edges:
        truth_matrix[sender, receptor] = 1  # sender -> receptor

    # filter to only masked positions
    y_true = truth_matrix[Evaluate_Mask].ravel()
    y_score = A[Evaluate_Mask].ravel()

    # Sanity checks
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0:
        print("Warning: no positive edges in ground truth within Evaluate_Mask.")
        return None, None, None
    if neg == 0:
        print("Warning: no negative edges in Evaluate_Mask (cannot compute AUROC/AUPRC).")
        return None, None, None

    auprc = average_precision_score(y_true, y_score)
    prevalence = y_true.mean()
    auprc_ratio = auprc / prevalence  # like BEELINE's AUPRC ratio idea
    auroc = roc_auc_score(y_true, y_score)

    return auprc, auprc_ratio, auroc