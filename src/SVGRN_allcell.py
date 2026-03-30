import os
import sys

import numpy as np
import pandas as pd
# import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from src.Con_Model_newED import CVAE_EAD_newED
from src.subcellular_feature_store import (
    load_precomputed_subcellular_features,
    min_max_scale,
)
from src.utils import evaluate, extractEdgesFromMatrix

class non_celltype_GRN_model:
    def __init__(self, opt):
        self.opt = opt
        os.makedirs(opt.save_name, exist_ok=True)
        self.subcellular_feature_dir = opt.subcellular_feature_dir

    def initalize_A_withTF(self, TF_mask):
        A = TF_mask.copy()
        for i in range(len(A)):
            A[i, i] = 0
        return A

    def init_data(self):

        All_Data = pd.read_csv(self.opt.data_file, index_col=[0])
        All_Data.index = All_Data.index.astype(str)

        pos_df = All_Data[['x', 'y']].copy()
        data = All_Data.drop(columns=['x', 'y', 'ClusterID'], errors='ignore')
        data.columns = data.columns.astype(str)
        All_gene = list(data.columns)     # gene column names are all string
        gene_name = All_gene
        subcell_coloc_array, subcell_array, loaded_feature_dir = load_precomputed_subcellular_features(
            feature_dir=self.subcellular_feature_dir,
            cell_ids=All_Data.index,
            gene_names=All_gene,
        )
        print(
            f"Precomputed subcellular features loaded from {loaded_feature_dir}. "
            f"coloc_shape={subcell_coloc_array.shape}, grid_shape={subcell_array.shape}"
        )
        pos_df = min_max_scale(pos_df)

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
        subcell_coloc_train = torch.FloatTensor(subcell_coloc_array.astype(np.float32, copy=False))
        subcell_train = torch.FloatTensor(subcell_array.astype(np.float32, copy=False))

        # add spatial (x,y) and subcellular features for each cell input
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                torch.FloatTensor(Dropout_Mask), pos_train, subcell_coloc_train, subcell_train)

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

        return (
            dataloader,
            num_nodes,
            num_genes,
            data,
            truth_edges,
            TF_mask,
            gene_name,
            tuple(subcell_coloc_train.shape[1:]),
            tuple(subcell_train.shape[1:]),
        )


    def train_model(self):
        opt = self.opt
        use_gpu = bool(opt.GPU and torch.cuda.is_available())
        if opt.GPU and not torch.cuda.is_available():
            print("GPU requested but CUDA is not available. Falling back to CPU.")
        opt.device = torch.device('cuda:0' if use_gpu else 'cpu')
        print(opt.device)

        (
            dataloader,
            num_nodes,
            num_genes,
            data,
            truth_edges,
            TFmask2,
            gene_name,
            y_prime_coloc_shape,
            y_prime_grid_shape,
        ) = self.init_data()
        adj_A_init = self.initalize_A_withTF(TFmask2)

        y_pos_dim = 64

        cvae = CVAE_EAD_newED(
            adj_A_init,
            1,
            opt.n_hidden,
            opt.K,
            y_pos_dim,
            y_prime_coloc_dim=opt.y_prime_coloc_dim,
            y_prime_grid_dim=opt.y_prime_grid_dim,
            y_prime_coloc_shape=y_prime_coloc_shape,
            y_prime_grid_shape=y_prime_grid_shape,
            gene_pool_channels=opt.subcell_gene_pool_channels,
            cnn_hidden=opt.subcell_cnn_hidden,
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
                inputs, data_id, dropout_mask, Y_pos, Y_prime_coloc, Y_prime_grid = data_batch

                inputs = inputs.to(opt.device)
                Y_pos = Y_pos.to(opt.device)
                Y_prime_coloc = Y_prime_coloc.to(opt.device)
                Y_prime_grid = Y_prime_grid.to(opt.device)
                # print(f"Y_pos is tensor: {torch.is_tensor(Y_pos)}")
                data_ids.append(data_id.numpy())
                #data_ids.append(data_id.cpu().detach().numpy())
                temperature = max(0.95 ** epoch, 0.5)

                if opt.dropout_mask:
                    print("opt.dropout_mask")
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, Y_prime_coloc, Y_prime_grid,
                                                                           dropout_mask=dropout_mask.to(opt.device),
                                                                           temperature=temperature, opt=opt)
                else:
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, Y_prime_coloc, Y_prime_grid, dropout_mask=None,
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

