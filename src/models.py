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

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

import gensim
import faiss

# from kmeans_pymindspore import kmeans
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
        centroids = mindspore.tensor(centroids)
        self.centroids = ops.L2Normalize(centroids, axis=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = mindspore.LongTensor(seq2cluster)
        return seq2cluster, self.centroids[seq2cluster]


class KMeans_Pymindspore(object):
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
            seq2cluster = seq2cluster
            centroids = centroids
        # last batch where
        else:
            seq2cluster, centroids = kmeans(
                X=x, num_clusters=x.shape[0] - 1, distance="euclidean", device=self.device, tqdm_flag=False
            )
            seq2cluster = seq2cluster
            centroids = centroids
        return seq2cluster, centroids


class SASRecModel(nn.Cell):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.adaption_layer = nn.SequentialCell(
            nn.Dense(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Dense(args.hidden_size, 1),
            nn.Softsign(),
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.US = nn.Dense(args.item_size - 2, 4 * args.hidden_size, has_bias=False)
        self.V = nn.Dense(args.item_size - 2, 4 * args.hidden_size, has_bias=False)
        if self.args.fuse:
            self.fuse_layer = nn.Dense(args.hidden_size * 2, args.hidden_size)
        self.apply(self.init_weights)
        if args.load_graph: # load pretrained graph
            path = args.data_name + '_'
            self.US.weight.data.copy_(mindspore.load_checkpoint(path + 'US.pth'))
            self.V.weight.data.copy_(mindspore.load_checkpoint(path + 'V.pth'))
            self.US.weight.requires_grad_ = False
            self.V.weight.requires_grad_ = False

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
        norm_adj = (degree @ sparse_matrix + sparse_matrix @ degree)
        norm_adj = ops.dense_to_sparse_csr(mindspore.tensor(norm_adj.toarray(), dtype=mindspore.float32))
        self.norm_adj = norm_adj

    def get_svd(self):
        U, S, V = ops.svd(self.norm_adj.to_dense(), self.args.hidden_size * 4) # No lowrank_svd in mindspore
        self.real_US = U @ ops.diag(S)
        self.real_V = V

    def get_gnn_embeddings(self, noise=True):
        self.norm_adj = self.norm_adj
        if noise:
            US = ops.csr_mm(self.norm_adj, self.US.weight.T)
            V = ops.csr_mm(self.norm_adj, self.V.weight.T)
        emb = self.item_embeddings.weight[1:-1]
        emb_list = [emb]
        for idx in range(self.args.gnn_layers):
            if noise:
                if self.args.svd:
                    self.real_US, self.real_V = self.real_US, self.real_V
                    emb = self.real_US @ (self.real_V.T @ emb)
                else:
                    emb = ops.csr_mm(self.norm_adj, emb)
                    emb = emb + self.args.graph_noise * US @ (V.T @ emb)
            else:
                emb = ops.csr_mm(self.norm_adj, emb)
            emb_list.append(emb)
        emb = ops.stack(emb_list, dim=1).mean(1)
        return emb
    
    # Positional Embedding
    def add_position_embedding(self, sequence, user_ids=None):
        self.norm_adj = self.norm_adj

        seq_length = sequence.shape[1]

        position_ids = ops.arange(seq_length, dtype=mindspore.int64)
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
    def construct(self, input_ids, user_ids=None):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # mindspore.int64
        max_len = attention_mask.shape[-1]
        attn_shape = (1, max_len, max_len)
        subsequent_mask = ops.triu(ops.ones(attn_shape), diagonal=1)  # mindspore.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.trainable_params()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids, user_ids)
        if self.args.att_bias and user_ids != None:
            row, col = ops.repeat_interleave(input_ids, max_len, axis=-1).flatten(), ops.repeat_elements(input_ids, max_len, axis=-1).flatten()
            self.norm_adj = self.norm_adj
            self.dense_norm_adj = self.dense_norm_adj
            g = self.dense_norm_adj
            att_bias = g[row - 1, col - 1].reshape(input_ids.shape[0], 1, max_len, max_len) # item 0 will get a wrong emb, but it will be masked
            if self.args.gsl_weight:
                unique_row, inv_row = ops.unique(row - 1)
                unique_col, inv_col = ops.unique(col - 1)
                g_gsl = self.US.weight.T[unique_row] @ self.V.weight.T[unique_col].T
                g_gsl = g_gsl[inv_row, inv_col].reshape(input_ids.shape[0], 1, max_len, max_len)
                att_bias = att_bias + self.args.graph_noise * g_gsl
            user_weight = self.adaption_layer(self.user_embeddings(user_ids)).unsqueeze(-1).unsqueeze(-1)
            if input_ids.shape[0] == 2 * user_ids.shape[0]: # for aug
                user_weight = user_weight.repeat(2, 1, 1, 1)
            att_bias = self.args.att_bias * user_weight * att_bias
        else:
            att_bias = None
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True, att_bias=att_bias)

        sequence_output = item_encoded_layers[-1]
        if self.args.fuse:
            graph_emb = self.get_gnn_embeddings(noise=False)
            sequence_output = self.fuse_layer(ops.cat([sequence_output, graph_emb[input_ids - 1]], dim=-1))
            att_bias = None
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pymindspore/pymindspore/pull/5617
            module.weight.set_data(ops.normal(module.weight.shape, mean=0.0, stddev=self.args.initializer_range))
        elif isinstance(module, nn.Embedding):
            module.embedding_table.set_data(ops.normal(module.embedding_table.shape, mean=0.0, stddev=self.args.initializer_range))
