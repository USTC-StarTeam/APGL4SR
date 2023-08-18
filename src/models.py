# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import math
import os
import pickle
from tqdm import tqdm
import random
import copy
import numpy as np
import scipy.sparse as sp
import dgl
from dgl.nn.pytorch.conv import GraphConv

import torch
import torch.nn as nn
import torch.nn.functional as F

import gensim
import faiss

# from kmeans_pytorch import kmeans
import time

from modules import Encoder, LayerNorm


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]


class KMeans_Pytorch(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 10
        self.first_batch = True
        self.hidden_size = hidden_size
        self.gpu_id = gpu_id
        self.device = device
        print(self.device, "-----")

    def run_kmeans(self, x, Niter=20, tqdm_flag=False):
        if x.shape[0] >= self.num_cluster:
            seq2cluster, centroids = kmeans(
                X=x, num_clusters=self.num_cluster, distance="euclidean", device=self.device, tqdm_flag=False
            )
            seq2cluster = seq2cluster.to(self.device)
            centroids = centroids.to(self.device)
        # last batch where
        else:
            seq2cluster, centroids = kmeans(
                X=x, num_clusters=x.shape[0] - 1, distance="euclidean", device=self.device, tqdm_flag=False
            )
            seq2cluster = seq2cluster.to(self.device)
            centroids = centroids.to(self.device)
        return seq2cluster, centroids


class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.adaption_layer = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
            nn.Softsign(),
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.graph_conv = GraphConv(1, 1, weight=False, bias=False)
        self.US = nn.Linear(args.item_size - 2, 4 * args.hidden_size, bias=False)
        self.V = nn.Linear(args.item_size - 2, 4 * args.hidden_size, bias=False)
        if self.args.fuse:
            self.fuse_layer = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.apply(self.init_weights)
        if args.load_graph: # load pretrained graph
            path = args.data_name + '_'
            self.US.weight.data.copy_(torch.load(path + 'US.pth', map_location='cpu'))
            self.V.weight.data.copy_(torch.load(path + 'V.pth', map_location='cpu'))
            self.US.weight.requires_grad_ = False
            self.V.weight.requires_grad_ = False
            return

    def global_graph_construction(self, train_data):
        args = train_data.args
        user_seq, n_items, k = train_data.user_seq, args.item_size - 2, args.k
        row, col, data = [], [], []
        for item_list in user_seq:
            item_list = item_list[:-2] # remove valid/test data
            item_list_len = len(item_list)
            for item_idx in range(item_list_len - 1):
                target_num = min(k, item_list_len - item_idx - 1)
                row += [item_list[item_idx]] * target_num
                col += item_list[item_idx + 1: item_idx + 1 + target_num]
                data.append(1 / np.arange(1, 1 + target_num))
        data = np.concatenate(data)
        row, col = np.array(row), np.array(col)
        row, col = row - 1, col - 1 # remove padding
        sparse_matrix = sp.csc_matrix((data, (row, col)), shape=(n_items, n_items))
        sparse_matrix = sparse_matrix + sparse_matrix.T + sp.eye(n_items)
        degree = np.array((sparse_matrix > 0).sum(1)).flatten()
        degree = np.nan_to_num(1 / degree, posinf=0)
        degree = sp.diags(degree)
        norm_adj = (degree @ sparse_matrix + sparse_matrix @ degree).tocoo()
        g = dgl.from_scipy(norm_adj)
        g.edata['weight'] = torch.tensor(norm_adj.data)
        self.g = g
        self.g_cpu = g.clone()
        norm_adj = torch.sparse_coo_tensor(
            np.row_stack([norm_adj.row, norm_adj.col]),
            norm_adj.data,
            (n_items, n_items),
            dtype=torch.float32
        )
        self.norm_adj = norm_adj

    def get_svd(self):
        U, S, V = torch.svd_lowrank(self.norm_adj.to_dense(), self.args.hidden_size * 4)
        self.real_US = U @ torch.diag(S)
        self.real_V = V

    def get_gnn_embeddings(self, device, noise=True):
        self.g, self.norm_adj = self.g.to(device), self.norm_adj.to(device)
        if noise:
            US = torch.sparse.mm(self.norm_adj, self.US.weight.T)
            V = torch.sparse.mm(self.norm_adj, self.V.weight.T)
        emb = self.item_embeddings.weight[1:-1]
        emb_list = [emb]
        for idx in range(self.args.gnn_layers):
            if noise:
                if self.args.svd:
                    self.real_US, self.real_V = self.real_US.to(device), self.real_V.to(device)
                    emb = self.real_US @ (self.real_V.T @ emb)
                else:
                    emb = self.graph_conv(self.g, emb)
                    emb = emb + self.args.graph_noise * US @ (V.T @ emb)
            else:
                emb = self.graph_conv(self.g, emb)
            emb_list.append(emb)
        emb = torch.stack(emb_list, dim=1).mean(1)
        return emb
    
    # Positional Embedding
    def add_position_embedding(self, sequence, user_ids=None):
        self.g, self.norm_adj = self.g.to(sequence.device), self.norm_adj.to(sequence.device)

        seq_length = sequence.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        if self.args.pe:
            sequence_emb = item_embeddings + position_embeddings
        else:
            sequence_emb = item_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids, user_ids=None):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids, user_ids)
        if self.args.att_bias and user_ids != None:
            row, col = input_ids.repeat_interleave(max_len, dim=-1).flatten(), input_ids.repeat(1, max_len).flatten()
            self.norm_adj = self.norm_adj.to(input_ids.device)
            self.dense_norm_adj = self.dense_norm_adj.to(input_ids.device)
            g = self.dense_norm_adj
            att_bias = g[row - 1, col - 1].reshape(input_ids.shape[0], 1, max_len, max_len) # item 0 will get a wrong emb, but it will be masked
            if self.args.gsl_weight:
                unique_row, inv_row = torch.unique(row - 1, return_inverse=True)
                unique_col, inv_col = torch.unique(col - 1, return_inverse=True)
                g_gsl = self.US.weight.T[unique_row] @ self.V.weight.T[unique_col].T
                g_gsl = g_gsl[inv_row, inv_col].reshape(input_ids.shape[0], 1, max_len, max_len)
                att_bias = att_bias + self.args.graph_noise * g_gsl
            user_weight = self.adaption_layer(self.user_embeddings.weight[user_ids]).unsqueeze(-1).unsqueeze(-1)
            if input_ids.shape[0] == 2 * user_ids.shape[0]: # for aug
                user_weight = user_weight.repeat(2, 1, 1, 1)
            att_bias = self.args.att_bias * user_weight * att_bias
        else:
            att_bias = None
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True, att_bias=att_bias)

        sequence_output = item_encoded_layers[-1]
        if self.args.fuse:
            graph_emb = self.get_gnn_embeddings(input_ids.device, noise=False)
            sequence_output = self.fuse_layer(torch.cat([sequence_output, graph_emb[input_ids - 1]], dim=-1))
            att_bias = None
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class OnlineItemSimilarity:
    def __init__(self, item_size):
        self.item_size = item_size
        self.item_embeddings = None
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.total_item_list = torch.tensor([i for i in range(self.item_size)], dtype=torch.long).to(self.device)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def update_embedding_matrix(self, item_embeddings):
        self.item_embeddings = copy.deepcopy(item_embeddings)
        self.base_embedding_matrix = self.item_embeddings(self.total_item_list)

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item_idx in range(1, self.item_size):
            try:
                item_vector = self.item_embeddings(item_idx).view(-1, 1)
                item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
                max_score = max(torch.max(item_similarity), max_score)
                min_score = min(torch.min(item_similarity), min_score)
            except:
                continue
        return max_score, min_score

    def most_similar(self, item_idx, top_k=1, with_score=False):
        item_idx = torch.tensor(item_idx, dtype=torch.long).to(self.device)
        item_vector = self.item_embeddings(item_idx).view(-1, 1)
        item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
        item_similarity = (self.max_score - item_similarity) / (self.max_score - self.min_score)
        # remove item idx itself
        values, indices = item_similarity.topk(top_k + 1)
        if with_score:
            item_list = indices.tolist()
            score_list = values.tolist()
            if item_idx in item_list:
                idd = item_list.index(item_idx)
                item_list.remove(item_idx)
                score_list.pop(idd)
            return list(zip(item_list, score_list))
        item_list = indices.tolist()
        if item_idx in item_list:
            item_list.remove(item_idx)
        return item_list


class OfflineItemSimilarity:
    def __init__(self, data_file=None, similarity_path=None, model_name="ItemCF", dataset_name="Sports_and_Outdoors"):
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        # train_data_list used for item2vec, train_data_dict used for itemCF and itemCF-IUF
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(data_file)
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score

    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path="./similarity.pkl"):
        print("saving data to ", save_path)
        with open(save_path, "wb") as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(" ", 1)
            # only use training data
            items = items.split(" ")[:-3]
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid, itemid, int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self, train=None, save_path="./"):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()

        if self.model_name in ["ItemCF", "ItemCF_IUF"]:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                if self.model_name == "ItemCF":
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1
                elif self.model_name == "ItemCF_IUF":
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item, {})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == "Item2Vec":
            # details here: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
            print("Step 1: train item2vec model")
            item2vec_model = gensim.models.Word2Vec(
                sentences=self.train_data_list, vector_size=20, window=5, min_count=0, epochs=100
            )
            self.itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            print("Step 2: convert to item similarity dict")
            total_items = tqdm(item2vec_model.wv.index_to_key, total=total_item_nums)
            for cur_item in total_items:
                related_items = item2vec_model.wv.most_similar(positive=[cur_item], topn=20)
                self.itemSimBest.setdefault(cur_item, {})
                for (related_item, score) in related_items:
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score
            print("Item2Vec model saved to: ", save_path)
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == "LightGCN":
            # train a item embedding from lightGCN model, and then convert to sim dict
            print("generating similarity model..")
            itemSimBest = light_gcn.generate_similarity_from_light_gcn(self.dataset_name)
            print("LightGCN based model saved to: ", save_path)
            self._save_dict(itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError("invalid path")
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ["ItemCF", "ItemCF_IUF", "Item2Vec", "LightGCN"]:
            with open(similarity_model_path, "rb") as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == "Random":
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ["ItemCF", "ItemCF_IUF", "Item2Vec", "LightGCN"]:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(
                    self.similarity_model[str(item)].items(), key=lambda x: x[1], reverse=True
                )[0:top_k]
                if with_score:
                    return list(
                        map(
                            lambda x: (int(x[0]), (self.max_score - float(x[1])) / (self.max_score - self.min_score)),
                            top_k_items_with_score,
                        )
                    )
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(
                    self.similarity_model[int(item)].items(), key=lambda x: x[1], reverse=True
                )[0:top_k]
                if with_score:
                    return list(
                        map(
                            lambda x: (int(x[0]), (self.max_score - float(x[1])) / (self.max_score - self.min_score)),
                            top_k_items_with_score,
                        )
                    )
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == "Random":
            random_items = random.sample(self.similarity_model, k=top_k)
            return list(map(lambda x: int(x), random_items))


if __name__ == "__main__":
    onlineitemsim = OnlineItemSimilarity(item_size=10)
    item_embeddings = nn.Embedding(10, 6, padding_idx=0)
    onlineitemsim.update_embedding_matrix(item_embeddings)
    item_idx = torch.tensor(2, dtype=torch.long)
    similiar_items = onlineitemsim.most_similar(item_idx=item_idx, top_k=1)
    print(similiar_items)
