# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#


import numpy as np
from tqdm import tqdm
import random

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


from models import KMeans
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, PCLoss
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr


class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = True
        self.device = "cuda"

        self.model = model

        self.num_intent_clusters = [int(i) for i in self.args.num_intent_clusters.split(",")]
        self.clusters = []

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        self.optim = nn.Adam(
            self.model.trainable_params(),
            learning_rate=self.args.lr,
            beta1=self.args.adam_beta1,
            beta2=self.args.adam_beta2,
            weight_decay=self.args.weight_decay
        )
        self.grad_fn = mindspore.value_and_grad(self.train_step, None, self.optim.parameters, has_aux=True)

        print("Total Parameters:", sum([p.nelement() for p in self.model.trainable_params()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        self.pcl_criterion = PCLoss(self.args.temperature, self.device)

    def train_step(self):
        raise NotImplementedError

    def train_iteration(epoch):
        raise NotImplementedError
    
    def eval_iteration(epoch):
        raise NotImplementedError

    def train(self, epoch):
        self.train_iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        valid_rst = self.eval_iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)
        return valid_rst

    def test(self, epoch, full_sort=False):
        test_rst = self.eval_iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)
        return test_rst

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HIT@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        mindspore.save_checkpoint(self.model, file_name)

    def load(self, file_name):
        print(file_name)
        mindspore.load_param_into_net(self.model, mindspore.load_checkpoint(file_name + '.ckpt'))

    def binary_cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.shape[2])
        neg = neg_emb.view(-1, neg_emb.shape[2])
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = ops.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = ops.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.shape[0] * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = ops.sum(
            -ops.log(ops.sigmoid(pos_logits) + 1e-24) * istarget
            - ops.log(1 - ops.sigmoid(neg_logits) + 1e-24) * istarget
        ) / ops.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = ops.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.embedding_table.data
        # [batch hidden_size ]
        rating_pred = ops.matmul(seq_out, test_item_emb.transpose(1, 0))
        return rating_pred


class ICLRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(ICLRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )

    def _gsl_contrastive_learning(self, item_ids):
        item_ids = ops.unique(item_ids)[0] - 1
        if item_ids[0] == -1: # Remove mask item
            item_ids = item_ids[1:]
        item_all_vec1 = self.model.get_gnn_embeddings()
        item_all_vec2 = self.model.get_gnn_embeddings(noise=False)
        item_all_vec1 = ops.L2Normalize(axis=-1)(item_all_vec1[item_ids])
        item_all_vec2 = ops.L2Normalize(axis=-1)(item_all_vec2[item_ids])
        cl_loss = self.cf_criterion(item_all_vec1, item_all_vec2, temp=self.args.graph_temp)
        return cl_loss

    def _instance_cl_one_pair_contrastive_learning(self, inputs, intent_ids=None, temp=None, user_ids=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_batch = ops.Concat()(inputs)
        cl_sequence_output = self.model(cl_batch, user_ids)
        if self.args.seq_representation_instancecl_type == "mean":
            cl_sequence_output = ops.mean(cl_sequence_output, axis=1, keep_dims=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = ops.split(cl_sequence_flatten, batch_size)
        if self.args.de_noise:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids, temp=temp)
        else:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None, temp=temp)
        return cl_loss

    def _pcl_one_pair_contrastive_learning(self, inputs, intents, intent_ids):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        intents: [num_clusters batch_size hidden_dims]
        """
        n_views, (bsz, seq_len) = len(inputs), inputs[0].shape
        cl_batch = ops.Concat()(inputs)
        cl_batch = cl_batch
        cl_sequence_output = self.model(cl_batch)
        if self.args.seq_representation_type == "mean":
            cl_sequence_output = ops.mean(cl_sequence_output, axis=1, keep_dims=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_output_slice = ops.split(cl_sequence_flatten, bsz)
        if self.args.de_noise:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=intent_ids)
        else:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=None)
        return cl_loss

    def eval_iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=False):

        # Setting the tqdm progress bar
        rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
        self.model.set_train(False)

        pred_list = None

        if full_sort:
            answer_list = None
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers = batch
                recommend_output = self.model(input_ids, user_ids)

                recommend_output = recommend_output[:, -1, :]
                # recommendation results

                rating_pred = self.predict_full(recommend_output)

                rating_pred = rating_pred.numpy().copy()
                batch_user_index = user_ids.numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # argpartition T: O(n)  argsort O(nlogn)
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.numpy(), axis=0)
            return self.get_full_sort_score(epoch, answer_list, pred_list)

        else:
            for i, batch in rec_data_iter:
                batch = tuple(t for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                recommend_output = self.model.finetune(input_ids, user_ids)
                op = ops.Concat(-1)
                test_neg_items = op(answers, sample_negs)
                recommend_output = recommend_output[:, -1, :]

                test_logits = self.predict_sample(recommend_output, test_neg_items)
                test_logits = test_logits.numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_sample_scores(epoch, pred_list)

    def train_step(self, epoch, rec_batch, cl_batches, seq_class_label_batches):
        """
        rec_batch shape: key_name x batch_size x feature_dim
        cl_batches shape: 
            list of n_views x batch_size x feature_dim tensors
        """
        # 0. batch_data will be sent into the device(GPU or CPU)
        rec_batch = tuple(t for t in rec_batch)
        user_ids, input_ids, target_pos, target_neg, _ = rec_batch

        # ---------- recommendation task ---------------#
        sequence_output = self.model(input_ids, user_ids=user_ids)
        rec_loss = self.binary_cross_entropy(sequence_output, target_pos, target_neg)

        # ---------- contrastive learning task -------------#
        cl_losses = []
        cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
            cl_batches, intent_ids=seq_class_label_batches, user_ids=user_ids
        )
        cl_losses.append(self.args.cf_weight * cl_loss1)

        # graph contrastive loss
        if self.args.gcl_weight > 0:
            gcl_loss = self._gsl_contrastive_learning(target_pos)
            cl_losses.append(self.args.gcl_weight * gcl_loss)

        joint_loss = self.args.rec_weight * rec_loss
        for cl_loss in cl_losses:
            joint_loss += cl_loss
        return joint_loss, rec_loss, cl_losses

    def train_iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        print("Performing Rec model Training:")
        self.model.set_train()
        rec_avg_loss = 0.0
        cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
        cl_sum_avg_loss = 0.0
        joint_avg_loss = 0.0

        print(f"rec dataset length: {len(dataloader)}")
        rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
            (joint_loss, rec_loss, cl_losses), grads = self.grad_fn(epoch, rec_batch, cl_batches, seq_class_label_batches)
            self.optim(grads)
            rec_avg_loss += rec_loss.asnumpy()
            for i, cl_loss in enumerate(cl_losses):
                cl_sum_avg_loss += cl_loss.asnumpy()
            joint_avg_loss += joint_loss.asnumpy()
            

        post_fix = {
            "epoch": epoch,
            "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
            "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
        }
        if (epoch + 1) % self.args.log_freq == 0:
            print(str(post_fix))

        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")