import numpy as np
import pandas as pd
import copy
import math

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def RBF_weights(row, target_r, target_c, weight):
    # use Radial Basis Function to calculate the weight for each sample 
    # based on distance with the target cell
    sigma = weight
    dist = euclidean_distance([row["x"], row["y"]], [target_r, target_c]) ** 2
    weight = math.exp(-dist / (2.0 * sigma ** 2))
    return weight


def extractEdgesFromMatrix(m, geneNames,TFmask):
    geneNames = np.array(geneNames)
    mat = copy.deepcopy(m)
    num_nodes = mat.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    if TFmask is not None:
        mat = mat*TFmask
    mat_indicator_all[abs(mat) > 0] = 1
    idx_rec, idx_send = np.where(mat_indicator_all)
    edges_df = pd.DataFrame(
        {'TF': geneNames[idx_send], 'Target': geneNames[idx_rec], 'EdgeWeight': (mat[idx_rec, idx_send])})
    edges_df = edges_df.sort_values('EdgeWeight', ascending=False)

    return edges_df


def evaluate(A, truth_edges, Evaluate_Mask):
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)
    A= abs(A)
    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones_like(A) - np.eye(len(A))
    A = A * Evaluate_Mask
    A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0])
    A_val.reverse()
    cutoff_all = A_val[num_truth_edges]
    A_indicator_all = np.zeros([num_nodes, num_nodes])
    A_indicator_all[abs(A) > cutoff_all] = 1
    idx_rec, idx_send = np.where(A_indicator_all)
    A_edges = set(zip(idx_send, idx_rec))
    overlap_A = A_edges.intersection(truth_edges)
    return len(overlap_A), 1. * len(overlap_A) / ((num_truth_edges ** 2) / np.sum(Evaluate_Mask))
