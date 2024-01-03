#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import random
import numpy as np
import mindspore
import mindspore.ops as ops

from data_augmentation import Crop, Mask, Reorder, Random
from utils import neg_sample, nCr
import copy


class RecWithContrastiveLearningDataset():
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train", similarity_model_type="offline"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        # currently apply one transform, will extend to multiples
        self.augmentations = {
            "crop": Crop(tao=args.tao),
            "mask": Mask(gamma=args.gamma),
            "reorder": Reorder(beta=args.beta),
            "random": Random(tao=args.tao, gamma=args.gamma, beta=args.beta),
        }
        if self.args.augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.augment_type]
        # number of augmentations for each sequences, current support two
        self.n_views = self.args.n_views

    def _one_pair_data_augmentation(self, input_ids):
        """
        provides two positive samples given one sequence
        """
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids

            augmented_input_ids = augmented_input_ids[-self.max_len :]

            assert len(augmented_input_ids) == self.max_len

            cur_tensors = mindspore.tensor(augmented_input_ids)
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

    def _process_sequence_label_signal(self, seq_label_signal):
        seq_class_label = mindspore.tensor(seq_label_signal)
        return seq_class_label

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                mindspore.tensor(user_id),  # user_id for testing
                mindspore.tensor(copied_input_ids),
                mindspore.tensor(target_pos),
                mindspore.tensor(target_neg),
                mindspore.tensor(answer),
                mindspore.tensor(test_samples),
            )
        else:
            cur_rec_tensors = (
                mindspore.tensor(user_id),  # user_id for testing
                mindspore.tensor(copied_input_ids),
                mindspore.tensor(target_pos),
                mindspore.tensor(target_neg),
                mindspore.tensor(answer),
            )
        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            seq_label_signal = items[-2]
            answer = [0]  # no use
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
        if self.data_type == "train":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            cf_tensors_list = []
            # if n_views == 2, then it's downgraded to pair-wise contrastive learning
            total_augmentaion_pairs = nCr(self.n_views, 2)
            for i in range(total_augmentaion_pairs):
                cf_tensors_list.append(self._one_pair_data_augmentation(input_ids))

            # add supervision of sequences
            seq_class_label = self._process_sequence_label_signal(seq_label_signal)
            return (cur_rec_tensors, cf_tensors_list, seq_class_label)
        elif self.data_type == "valid":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)


class SASRecDataset():
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                mindspore.tensor(user_id),  # user_id for testing
                mindspore.tensor(input_ids),
                mindspore.tensor(target_pos),
                mindspore.tensor(target_neg),
                mindspore.tensor(answer),
                mindspore.tensor(test_samples),
            )
        else:
            cur_rec_tensors = (
                mindspore.tensor(user_id),  # user_id for testing
                mindspore.tensor(input_ids),
                mindspore.tensor(target_pos),
                mindspore.tensor(target_neg),
                mindspore.tensor(answer),
            )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)

class MyDataloader():
    def __init__(self, args, dataset, data_type) -> None:
        self.dataset = dataset
        self.batch_size = args.batch_size
        self.num_data = len(self.dataset)
        self.num_batch = self.num_data // self.batch_size
        self.batch_idx = np.arange(self.num_data)
        self.data_type = data_type
        self.stack = ops.Stack()

    def __len__(self):
        return self.num_batch

    @staticmethod
    def _split_into_batches(lst, batch_size):
        batches = []
        for i in range(0, len(lst), batch_size):
            batch = lst[i:i + batch_size]
            batches.append(batch)
        return batches

    def __iter__(self):
        self.iter_dataset = iter(self.dataset)
        random.shuffle(self.batch_idx)
        self.batch = self._split_into_batches(self.batch_idx, self.batch_size)
        
        for index in range(self.num_batch):
            batched_data_index = self.batch[index]
            cur_rec_tensors_list, cf_tensors_list, seq_class_label_list = [[], [], [], [], []], [[], []], []
            for idx in batched_data_index:
                if self.data_type == 'train':
                    cur_rec_tensors, cf_tensors, seq_class_label = self.dataset[idx]
                    user_id, input_ids, target_pos, target_neg, answer = cur_rec_tensors
                    cur_rec_tensors_list[0].append(user_id)
                    cur_rec_tensors_list[1].append(input_ids)
                    cur_rec_tensors_list[2].append(target_pos)
                    cur_rec_tensors_list[3].append(target_neg)
                    cur_rec_tensors_list[4].append(answer)
                    cf_tensors_list[0].append(cf_tensors[0][0])
                    cf_tensors_list[1].append(cf_tensors[0][1])
                    seq_class_label_list.append(seq_class_label)
                else:
                    cur_rec_tensors = self.dataset[idx]
                    user_id, input_ids, target_pos, target_neg, answer = cur_rec_tensors
                    cur_rec_tensors_list[0].append(user_id)
                    cur_rec_tensors_list[1].append(input_ids)
                    cur_rec_tensors_list[2].append(target_pos)
                    cur_rec_tensors_list[3].append(target_neg)
                    cur_rec_tensors_list[4].append(answer)

            if self.data_type == 'train':
                cur_rec_tensors_list = [self._pack(_) for _ in cur_rec_tensors_list]
                cf_tensors_list = [self._pack(_) for _ in cf_tensors_list]
                seq_class_label_list = self._pack(seq_class_label_list)
                yield cur_rec_tensors_list, cf_tensors_list, seq_class_label_list
            else:
                cur_rec_tensors_list = [self._pack(_) for _ in cur_rec_tensors_list]
                yield cur_rec_tensors_list

    def _pack(self, data):
        return self.stack(data)