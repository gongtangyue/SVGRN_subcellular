import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from src.Con_Model_newED import CVAE_EAD_newED
from src.utils import evaluate, extractEdgesFromMatrix

class non_celltype_GRN_model:
    SUBCELLULAR_DATA_DIR = (
        Path(__file__).resolve().parents[1]
        / "in_sim"
        / "g110_c2k_n01"
        / "subcellular_data"
    )

    def __init__(self, opt):
        self.opt = opt
        os.makedirs(opt.save_name, exist_ok=True)
        self.subcellular_data_dir = self._resolve_subcellular_data_dir()

    def _resolve_subcellular_data_dir(self):
        user_dir_raw = getattr(self.opt, "subcellular_data_dir", "")
        if user_dir_raw and str(user_dir_raw).strip():
            return Path(user_dir_raw).expanduser()

        if getattr(self.opt, "data_file", None):
            data_file_dir = Path(self.opt.data_file).resolve().parent
            inferred = data_file_dir / "subcellular_data"
            if inferred.exists():
                return inferred

        return self.SUBCELLULAR_DATA_DIR

    def initalize_A_withTF(self, TF_mask):
        A = TF_mask.copy()
        for i in range(len(A)):
            A[i, i] = 0
        return A

    @staticmethod
    def _min_max_scale(df):
        if isinstance(df, pd.DataFrame):
            denom = df.max(0) - df.min(0)
            denom = denom.replace(0, 1)
            return (df - df.min(0)) / denom

        arr = np.asarray(df, dtype=np.float32)
        if arr.size == 0:
            return arr
        min_vals = arr.min(axis=0, keepdims=True)
        denom = arr.max(axis=0, keepdims=True) - min_vals
        denom[denom == 0] = 1
        return (arr - min_vals) / denom

    @staticmethod
    def _compute_gaussian_colocalization(sub_df, gene_list, sigma):
        gene_to_idx = {str(gene): idx for idx, gene in enumerate(gene_list)}
        n_gene = len(gene_list)
        colocalization = np.zeros((n_gene, n_gene), dtype=np.float32)
        required_cols = {"gene", "x", "y"}

        if sigma <= 0 or sub_df.empty or not required_cols.issubset(sub_df.columns):
            return colocalization

        filtered = sub_df.loc[:, ["gene", "x", "y"]].copy()
        filtered["gene"] = filtered["gene"].astype(str)
        filtered = filtered[filtered["gene"].isin(gene_to_idx)]
        filtered = filtered.dropna(subset=["x", "y"])
        if filtered.shape[0] < 2:
            return colocalization

        coords = filtered[["x", "y"]].to_numpy(dtype=np.float32)
        gene_idx = filtered["gene"].map(gene_to_idx).to_numpy(dtype=np.int64)
        denom = 2.0 * float(sigma) ** 2

        for i in range(coords.shape[0] - 1):
            deltas = coords[i + 1:] - coords[i]
            dist_sq = np.einsum("ij,ij->i", deltas, deltas)
            weights = np.exp(-dist_sq / denom).astype(np.float32, copy=False)
            if weights.size == 0:
                continue

            gi = gene_idx[i]
            gj = gene_idx[i + 1:]
            np.add.at(colocalization, (np.full(gj.shape, gi, dtype=np.int64), gj), weights)
            np.add.at(colocalization, (gj, np.full(gj.shape, gi, dtype=np.int64)), weights)

        total_weight = colocalization.sum()
        if total_weight > 0:
            colocalization /= total_weight

        return colocalization

    def _load_subcellular_features(self, cell_ids, gene_list):
        sub_features = []
        missing_or_empty = 0
        sigma = float(getattr(self.opt, "subcell_sigma", 0.03))
        feature_dim = len(gene_list) * len(gene_list)

        for cell_id in cell_ids:
            csv_path = self.subcellular_data_dir / f"{cell_id}_subcellular.csv"
            if csv_path.exists():
                sub_df = pd.read_csv(csv_path)
                colocalization = self._compute_gaussian_colocalization(sub_df, gene_list, sigma)
                if np.any(colocalization):
                    sub_features.append(colocalization)
                    continue
            missing_or_empty += 1
            sub_features.append(np.zeros((len(gene_list), len(gene_list)), dtype=np.float32))

        subcell_array = np.stack(sub_features, axis=0).reshape(len(cell_ids), feature_dim)
        subcell_array = self._min_max_scale(subcell_array)

        print(
            f"Subcellular features loaded from {self.subcellular_data_dir}. "
            f"sigma={sigma}. per-cell matrix: ({len(gene_list)}, {len(gene_list)}), "
            f"flattened dim: {feature_dim}. missing/empty files: {missing_or_empty}/{len(cell_ids)}"
        )
        return subcell_array

    def init_data(self):

        All_Data = pd.read_csv(self.opt.data_file, index_col=[0])
        All_Data.index = All_Data.index.astype(str)

        pos_df = All_Data[['x', 'y']].copy()
        data = All_Data.drop(columns=['x', 'y', 'ClusterID'], errors='ignore')
        data.columns = data.columns.astype(str)
        All_gene = list(data.columns)     # gene column names are all string
        gene_name = All_gene
        subcell_array = self._load_subcellular_features(All_Data.index, All_gene)
        pos_df = self._min_max_scale(pos_df)

        # load TF list from a file or other sources
        with open(self.opt.tf_list, "r") as f:
            TF = [line.strip() for line in f if line.strip()]

        print(f"TF {TF}, all gene {All_gene}")
        print(f"TF num {len(TF)}, All_gene num {len(All_gene)}")

        data_values = data.to_numpy(copy=True)
        Dropout_Mask = (data_values != 0).astype(float)
        
        num_genes, num_nodes = data.shape[1], data.shape[0]
        print(f"num_genes {num_genes}, num_nodes {num_nodes}")
        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                if item2 in TF:
                    TF_mask[i, j] = 1

        feat_train = torch.FloatTensor(data_values)
        pos_train = torch.FloatTensor(pos_df.to_numpy(copy=True))
        subcell_train = torch.FloatTensor(subcell_array.astype(np.float32, copy=False))

        # add spatial (x,y) and subcellular features for each cell input
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                torch.FloatTensor(Dropout_Mask), pos_train, subcell_train)

        # Use single-process loading for cross-platform stability (especially on Windows).
        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)

        if self.opt.net_file is None:
            print("No ground truth file provided.")
            truth_edges = None
        else:
            Ground_Truth = pd.read_csv(self.opt.net_file, index_col=0)
            nonzero_indices = np.where(Ground_Truth.values != 0)
            truth_edges = [(int(Ground_Truth.columns[col])-1, int(Ground_Truth.index[row])-1) for row, col in zip(*nonzero_indices)]
            truth_edges = set(truth_edges)   # idx_send, idx_rec

        return dataloader, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name, subcell_train.shape[1]


    def train_model(self):
        opt = self.opt
        use_gpu = bool(opt.GPU and torch.cuda.is_available())
        if opt.GPU and not torch.cuda.is_available():
            print("GPU requested but CUDA is not available. Falling back to CPU.")
        opt.device = torch.device('cuda:0' if use_gpu else 'cpu')
        print(opt.device)

        dataloader, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name, y_prime_input_dim = self.init_data()
        adj_A_init = self.initalize_A_withTF(TFmask2)

        y_pos_dim = 128

        cvae = CVAE_EAD_newED(
            adj_A_init,
            1,
            opt.n_hidden,
            opt.K,
            y_pos_dim,
            y_prime_input_dim=y_prime_input_dim,
        ).float().to(opt.device)

        print("Build model....")

        optimizer = optim.RMSprop(cvae.parameters(), lr=opt.lr)
        optimizer2 = optim.RMSprop([cvae.adj_A], lr=opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)
        best_Epr = 0
        cvae.train()

        if opt.GPU:
            RN_df = pd.DataFrame(cvae.adj_A.cpu().detach().numpy(), columns=list(gene_name))
        else:
            RN_df = pd.DataFrame(cvae.adj_A.detach().numpy(), columns=list(gene_name))
        RN_df.to_csv(opt.save_name + '/initial_RN.csv', index=False)

        for epoch in range(opt.n_epochs + 1):
            loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
            if epoch % (opt.K1 + opt.K2) < opt.K1:
                print(f"Epoch: {epoch} Not update adj_A")
                cvae.adj_A.requires_grad = False  # not update adj_A when epoch%3==0
            else:
                print(f"Epoch: {epoch} Only update adj_A")
                cvae.adj_A.requires_grad = True
            for i, data_batch in enumerate(dataloader, 0):
                # print(f"epoch: {epoch}, iter: {i}")
                optimizer.zero_grad()

                # add Y_pos = (x,y) as the corresponding pos for each cell input
                inputs, data_id, dropout_mask, Y_pos, Y_prime = data_batch

                inputs = inputs.to(opt.device)
                Y_pos = Y_pos.to(opt.device)
                Y_prime = Y_prime.to(opt.device)
                # print(f"Y_pos is tensor: {torch.is_tensor(Y_pos)}")
                data_ids.append(data_id.numpy())
                #data_ids.append(data_id.cpu().detach().numpy())
                temperature = max(0.95 ** epoch, 0.5)

                if opt.dropout_mask:
                    print("opt.dropout_mask")
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, Y_prime,
                                                                           dropout_mask=dropout_mask.to(opt.device),
                                                                           temperature=temperature, opt=opt)
                else:
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, Y_prime, dropout_mask=None,
                                                                           temperature=temperature, opt=opt)

                sparse_loss = opt.alpha * torch.mean(torch.abs(cvae.adj_A))
                loss = loss + sparse_loss
                loss.backward()
                mse_rec.append(loss_rec.item())
                loss_all.append(loss.item())
                loss_kl.append(loss_KL.item())
                loss_sparse.append(sparse_loss.item())
                if epoch % (opt.K1 + opt.K2) < opt.K1: # not update adj_A when epoch%3==0
                    optimizer.step()
                else:
                    optimizer2.step()  # only update adj_A
            scheduler.step()

            if epoch % (opt.K1 + opt.K2) >= opt.K1:
                if truth_edges is not None:
                    if opt.GPU:
                        Ep, Epr = evaluate(cvae.adj_A.cpu().detach().numpy(), truth_edges, TFmask2)
                    else:
                        Ep, Epr = evaluate(cvae.adj_A.detach().numpy(), truth_edges, TFmask2)

                    best_Epr = max(Epr, best_Epr)
                    print('epoch:', epoch, 'Ep:', Ep, 'Epr:', Epr, 'loss:',
                        np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                        np.mean(loss_sparse))

                    with open(opt.save_name + '/log1.txt', 'a') as f:
                        # Write the text to the file
                        f.write(f"Epoch: {epoch}, Ep: {Ep}, Epr: {Epr}, loss: {np.mean(loss_all)} " +
                        f"mse_loss: {np.mean(mse_rec)}, kl_loss: {np.mean(loss_kl)}, sparse_loss:{np.mean(loss_sparse)}\n")
                else:
                    print('epoch:', epoch, 'loss:',
                      np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                      np.mean(loss_sparse))

                    with open(opt.save_name + '/log1.txt', 'a') as f:
                        # Write the text to the file
                        f.write(f"Epoch: {epoch}, loss: {np.mean(loss_all)} " +
                        f"mse_loss: {np.mean(mse_rec)}, kl_loss: {np.mean(loss_kl)}, sparse_loss:{np.mean(loss_sparse)}\n")

        if opt.GPU:
            RN_df = pd.DataFrame(cvae.adj_A.cpu().detach().numpy(), columns=list(gene_name))
        else:
            RN_df = pd.DataFrame(cvae.adj_A.detach().numpy(), columns=list(gene_name))
        RN_df.to_csv(opt.save_name + f"/RN_{opt.n_epochs}.csv", index=False)
        
        # save model
        torch.save(cvae, opt.save_name + "/stage1.pt")        

