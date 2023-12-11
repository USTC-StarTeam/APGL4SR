# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np

import copy
import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class PCLoss(nn.Cell):
    """ Reference: https://github.com/salesforce/PCL/blob/018a929c53fcb93fd07041b1725185e1237d2c0e/pcl/builder.py#L168
    """

    def __init__(self, temperature, device, contrast_mode="all"):
        super(PCLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.criterion = NCELoss(temperature, device)

    def construct(self, batch_sample_one, batch_sample_two, intents, intent_ids=None):
        """
        features: 
        intents: num_clusters x batch_size x hidden_dims
        """
        # instance contrast with prototypes
        mean_pcl_loss = 0
        # do de-noise
        if intent_ids is not None:
            raise NotImplementedError
        # don't do de-noise
        else:
            for intent in intents:
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_ids=None)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_ids=None)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        return mean_pcl_loss


class NCELoss(nn.Cell):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1000000)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity()

    # #modified based on impl: https://github.com/ae-foster/pymindspore-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def construct(self, batch_sample_one, batch_sample_two, intent_ids=None, temp=None):
        if temp == None:
            sim11 = ops.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
            sim22 = ops.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
            sim12 = ops.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        else:
            sim11 = ops.matmul(batch_sample_one, batch_sample_one.T) / temp
            sim22 = ops.matmul(batch_sample_two, batch_sample_two.T) / temp
            sim12 = ops.matmul(batch_sample_one, batch_sample_two.T) / temp
        d = sim12.shape[-1]

        sim11 = ops.clamp(sim11, -10, 10)
        sim12 = ops.clamp(sim12, -10, 10)

        # avoid contrast against positive intents
        if intent_ids is not None:
            raise NotImplementedError
        else:
            mask = ops.eye(d, dtype=mindspore.int32)
            sim11[mask == 1] = -1000000
            sim22[mask == 1] = -1000000
            # sim22 = sim22.masked_fill_(mask, -np.inf)
            # sim11[..., range(d), range(d)] = float('-inf')
            # sim22[..., range(d), range(d)] = float('-inf')

        op = ops.Concat(-1)
        raw_scores1 = op([sim12, sim11])
        raw_scores2 = op([sim22, sim12.transpose(1, 0)])
        op = ops.Concat(-2)
        logits = op([raw_scores1, raw_scores2])
        labels = ops.arange(2 * d, dtype=mindspore.int32)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + mindspore.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * mindspore.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + ops.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * ops.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": ops.relu, "swish": swish}


class LayerNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = mindspore.Parameter(ops.ones(hidden_size))
        self.bias = mindspore.Parameter(ops.zeros(hidden_size))
        self.variance_epsilon = eps

    def construct(self, x):
        u = x.mean(-1, keep_dims=True)
        s = (x - u).pow(2).mean(-1, keep_dims=True)
        x = (x - u) / ops.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Cell):
    """Construct the embeddings from item, position.
    """

    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=args.hidden_dropout_prob)

        self.args = args

    def construct(self, input_ids):
        seq_length = input_ids.shape[1]
        position_ids = ops.arange(seq_length, dtype=mindspore.int32)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        # 修改属性
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Cell):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size * 4 / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(args.hidden_size, self.all_head_size)
        self.key = nn.Dense(args.hidden_size, self.all_head_size)
        self.value = nn.Dense(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(p=args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Dense(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(p=args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(self, input_tensor, attention_mask, att_bias=None):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.transpose((0, 1, 3, 2)))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask
        if att_bias != None:
            attention_scores = attention_scores + att_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Cell):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Dense(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Dense(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=args.hidden_dropout_prob)

    def construct(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Cell):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def construct(self, hidden_states, attention_mask, att_bias=None):
        attention_output = self.attention(hidden_states, attention_mask, att_bias)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Cell):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.CellList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def construct(self, hidden_states, attention_mask, output_all_encoded_layers=True, att_bias=None, noise=False):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, att_bias)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
