import os
import sys
import time

import numpy as np
import pandas as pd
# import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from src.Con_Model_newED import CVAE_EAD_newED as CVAE_EAD_newED_original
from src.Con_Model_newED_direct import CVAE_EAD_newED as CVAE_EAD_newED_direct
from src.subcellular_feature_store import (
    load_precomputed_subcellular_features,
    min_max_scale,
)
from src.utils import evaluate, extractEdgesFromMatrix, RBF_weights

Tensor = torch.cuda.FloatTensor



class SC_GRN_model:
    def __init__(self, opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')
        self.subcellular_feature_dir = opt.subcellular_feature_dir

    def init_data(self):

        All_Data = pd.read_csv(self.opt.data_file, index_col=[0])
        All_Data.index = All_Data.index.astype(str)
        
        pos_df = All_Data[['x', 'y']].copy()
        data = All_Data.drop(columns=['x', 'y', 'ClusterID'], errors='ignore')
        data.columns = data.columns.astype(str)
        gene_name = list(data.columns)     # gene column names are all string
        subcell_coloc_array, subcell_array, loaded_feature_dir = load_precomputed_subcellular_features(
            feature_dir=self.subcellular_feature_dir,
            cell_ids=All_Data.index,
            gene_names=gene_name,
        )
        print(
            f"Precomputed subcellular features loaded from {loaded_feature_dir}. "
            f"coloc_shape={subcell_coloc_array.shape}, grid_shape={subcell_array.shape}"
        )
        
        # calculate weight for loss based on its distance to target cell
        target_pos = pos_df.loc[self.opt.target_cell_name]
        weight_df = pd.DataFrame(columns=['weight'])
        weight_df['weight'] = pos_df.apply(lambda row: RBF_weights(row, target_pos['x'], target_pos['y'], self.opt.W), axis=1)
        # print(f"dist weight: {weight_df.head(5)}")
        total_weight = weight_df['weight'].sum()
        # print(f"sum : {total_weight}")

        # scale the sum of weight to sample number (multiply first to avoid number become so small)
        weight_df['weight'] = (weight_df['weight'] * pos_df.shape[0]) / total_weight
        # print(f"rescaled weight : {weight_df.head(5)}")
        print(f"weight_df describe: {weight_df.describe()}")
        print(f"weight_df num: ", (weight_df['weight'] >= 1).sum())
        print(f"sum now: {weight_df['weight'].sum()}")

        # Normalize then change all Y_pos to the target cell's (x, y).
        pos_df = min_max_scale(pos_df)
        target_pos = pos_df.loc[self.opt.target_cell_name]
        target_pos = np.array([[target_pos['x'], target_pos['y']]])
        print(f"target_pos: {target_pos}")
        pos_df = pd.DataFrame(np.repeat(target_pos, pos_df.shape[0], axis=0), 
                    columns=pos_df.columns, index=pos_df.index)

        # Use target cell's subcellular feature for all samples in stage-2.
        target_idx = All_Data.index.get_loc(self.opt.target_cell_name)
        target_subcell_coloc = subcell_coloc_array[target_idx:target_idx + 1]
        target_subcell = subcell_array[target_idx:target_idx + 1]
        print(f"target_subcell_coloc shape: {target_subcell_coloc.shape}")
        print(f"target_subcell shape: {target_subcell.shape}")
        subcell_coloc_array = np.repeat(target_subcell_coloc, subcell_coloc_array.shape[0], axis=0)
        subcell_array = np.repeat(target_subcell, subcell_array.shape[0], axis=0)

        data_values = data.to_numpy(copy=True)
        Dropout_Mask = (data_values != 0).astype(float)
        
        num_genes, num_nodes = data.shape[1], data.shape[0]
        print(f"num_genes {num_genes}, num_nodes {num_nodes}")

        feat_train = torch.FloatTensor(data_values)
        pos_train = torch.FloatTensor(pos_df.to_numpy(copy=True))
        subcell_coloc_train = torch.FloatTensor(subcell_coloc_array.astype(np.float32, copy=False))
        subcell_train = torch.FloatTensor(subcell_array.astype(np.float32, copy=False))
        weight_train = torch.FloatTensor(weight_df["weight"].to_numpy(copy=True))

        # add Y_pos, Y_prime(subcell), and sample weight
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask), pos_train, subcell_coloc_train, subcell_train, weight_train)

        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)

        if self.opt.net_file is None:
            truth_edges, TF_mask = None, None
        
        else:
            Ground_Truth = pd.read_csv(self.opt.net_file, index_col=0)
            Ground_Truth.index = Ground_Truth.index.astype(str)
            Ground_Truth.columns = Ground_Truth.columns.astype(str)
            TF = set(Ground_Truth.columns)
            All_gene = set(Ground_Truth.index)
            print(f"TF {TF}")
            print(f"TF num {len(TF)}, All_gene num {len(All_gene)}")

            TF_mask = np.zeros([num_genes, num_genes])
            for i, item in enumerate(data.columns):
                for j, item2 in enumerate(data.columns):
                    if i == j:
                        continue
                    if item2 in TF:
                        TF_mask[i, j] = 1

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
            subcell_coloc_train.shape[-1],
            tuple(subcell_train.shape[1:]),
        )

    def train_model(self):
        opt = self.opt
        use_gpu = bool(opt.GPU and torch.cuda.is_available())
        if opt.GPU and not torch.cuda.is_available():
            print("GPU requested but CUDA is not available. Falling back to CPU.")
        opt.device = torch.device('cuda:0' if use_gpu else 'cpu')
        print(opt.device)
        print("Using GPU....." if use_gpu else "Not using GPU....")
        
        (
            dataloader,
            num_nodes,
            num_genes,
            data,
            truth_edges,
            TFmask2,
            gene_name,
            y_prime_coloc_input_dim,
            y_prime_grid_shape,
        ) = self.init_data()
        
        # y_pos_dim = 128
        # PyTorch 2.6 defaults torch.load(weights_only=True), but stage1 saves a full model object.
        cvae = torch.load(opt.model_file, map_location=opt.device, weights_only=False).to(opt.device)    # load stage1 model
        print("model loaded")
        if not hasattr(cvae, "y_prime_coloc_input_dim") or not hasattr(cvae, "y_prime_grid_shape"):
            raise ValueError(
                "Loaded stage1 model does not include both colocalization and grid subcellular conditions. "
                "Please use a stage1.pt trained with the dual-condition subcellular representation."
            )
        model_y_prime_coloc_input_dim = int(cvae.y_prime_coloc_input_dim)
        if model_y_prime_coloc_input_dim != y_prime_coloc_input_dim:
            raise ValueError(
                "Stage2 subcellular colocalization dim does not match the loaded stage1 model: "
                f"stage2 got {y_prime_coloc_input_dim}, but stage1 expects {model_y_prime_coloc_input_dim}. "
                "Please use the same gene set and colocalization parameters in stage1 and stage2."
            )
        model_y_prime_grid_shape = tuple(
            getattr(cvae, "y_prime_grid_shape", y_prime_grid_shape)
        )
        if model_y_prime_grid_shape != y_prime_grid_shape:
            raise ValueError(
                "Stage2 subcellular grid shape does not match the loaded stage1 model: "
                f"stage2 got {y_prime_grid_shape}, but stage1 expects {model_y_prime_grid_shape}. "
                "Please use the same gene set and grid parameters in stage1 and stage2."
            )
        print(
            "Stage2 subcellular condition shapes match loaded stage1 model: "
            f"coloc_dim={model_y_prime_coloc_input_dim}, grid_shape={model_y_prime_grid_shape}"
        )

        optimizer2 = optim.RMSprop([cvae.adj_A], lr=opt.lr * 0.2) # only update
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=opt.lr_step_size, gamma=opt.gamma)
        best_Epr = 0
        cvae.train()

        # freeze all other params, only update adj_A
        for param in cvae.parameters():
            param.requires_grad = False
        cvae.adj_A.requires_grad = True

        for epoch in range(opt.n_epochs + 1):
            loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
            print(f"Epoch: {epoch}")
            
            for i, data_batch in enumerate(dataloader, 0):
                # print(f"epoch: {epoch}, iter: {i}")
                optimizer2.zero_grad()

                # add Y_pos = (x,y) as the corresponding pos for each cell input
                inputs, data_id, dropout_mask, Y_pos, Y_prime_coloc, Y_prime_grid, weight = data_batch

                inputs = inputs.to(opt.device)
                Y_pos = Y_pos.to(opt.device)
                Y_prime_coloc = Y_prime_coloc.to(opt.device)
                Y_prime_grid = Y_prime_grid.to(opt.device)
                weight = weight.to(opt.device)

                data_ids.append(data_id.numpy())
                temperature = max(0.95 ** epoch, 0.5)

                if opt.dropout_mask:
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, Y_prime_coloc, Y_prime_grid, loss_weight=weight,
                                                                            dropout_mask=dropout_mask.to(opt.device),
                                                                            temperature=temperature, opt=opt)

                else:
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, Y_prime_coloc, Y_prime_grid, loss_weight=weight,
                                                                            dropout_mask=None,
                                                                            temperature=temperature, opt=opt)

                sparse_loss = opt.alpha * torch.mean(torch.abs(cvae.adj_A))
                loss = loss + sparse_loss
                loss.backward()
                mse_rec.append(loss_rec.item())
                loss_all.append(loss.item())
                loss_kl.append(loss_KL.item())
                loss_sparse.append(sparse_loss.item())
                optimizer2.step()  

            scheduler.step()

            if truth_edges is not None and TFmask2 is not None:
                if opt.GPU:
                    Ep, Epr = evaluate(cvae.adj_A.cpu().detach().numpy(), truth_edges, TFmask2)
                else:
                    Ep, Epr = evaluate(cvae.adj_A.detach().numpy(), truth_edges, TFmask2)
                
                best_Epr = max(Epr, best_Epr)
                print('epoch:', epoch, 'Ep:', Ep, 'Epr:', Epr, 'loss:',
                        np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                        np.mean(loss_sparse))

                with open(opt.save_name + '/log1.txt', 'a') as f:
                    f.write(
                        f"Cell: {opt.target_cell_name}, Epoch: {epoch}, Ep: {Ep}, Epr: {Epr}, "
                        f"loss: {np.mean(loss_all)} mse_loss: {np.mean(mse_rec)}, "
                        f"kl_loss: {np.mean(loss_kl)}, sparse_loss:{np.mean(loss_sparse)}\n"
                    )
            
            else:
                print('epoch:', epoch, 'loss:',
                    np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                    np.mean(loss_sparse))

                with open(opt.save_name + '/log1.txt', 'a') as f:
                    f.write(
                        f"Cell: {opt.target_cell_name}, Epoch: {epoch}, loss: {np.mean(loss_all)} "
                        f"mse_loss: {np.mean(mse_rec)}, kl_loss: {np.mean(loss_kl)}, "
                        f"sparse_loss:{np.mean(loss_sparse)}\n"
                    )

        if opt.GPU:
            RN_df = pd.DataFrame(cvae.adj_A.cpu().detach().numpy(), columns=list(gene_name))
        else:
            RN_df = pd.DataFrame(cvae.adj_A.detach().numpy(), columns=list(gene_name))
        RN_df.to_csv(opt.save_name + f"/RN_{opt.n_epochs}.csv", index=False)

