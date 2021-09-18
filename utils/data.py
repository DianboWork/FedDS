import numpy as np
import torch
import random
import json
import os
import pickle


class GlobalBag:
    def __init__(self, path, rel2id, mode="Train"):
        self.data = self.load_data(path)
        if isinstance(rel2id, str):
            self.rel2id = json.load(open(rel2id))
        elif isinstance(rel2id, dict):
            self.rel2id = rel2id
        else:
            raise ValueError("Unsupported type: %s" % (type(rel2id)))
        if mode == "Train":
            self.fact2idx = {}
            for idx, item in enumerate(self.data):
                if 'id' in item["h"]:
                    if item['relation'] != 'NA':
                        fact = (item['h']['id'], item['t']['id'], item['relation'])
                        if fact not in self.fact2idx:
                            self.fact2idx[fact] = len(self.fact2idx)
                else:
                    if item['relation'] != 'NA':
                        fact = (item['entity_pair'], item['entity_pair'], item['relation']) #in order to align NYT Format
                        if fact not in self.fact2idx:
                            self.fact2idx[fact] = len(self.fact2idx)

    def load_data(self, path):
        f = open(path)
        data = []
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                data.append(eval(line))
        f.close()
        return data


class Bag_Data():
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """

    def __init__(self, data, rel2id, tokenizer, entpair_as_bag=False, bag_size=0):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring
                relation labels)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size
        self.bags = []
        self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
        self.bag_scope = []
        self.facts = {}
        self.bag_name = []
        self.bag_name2idx = {}

    def fact_generator(self):
        for idx, item in enumerate(self.data):
            if 'id' in item["h"]: #for nyt
                fact = (item['h']['id'], item['t']['id'], item['relation'])
                if item['relation'] != 'NA':
                    self.facts[fact] = 1
                if self.entpair_as_bag:
                    name = (item['h']['id'], item['t']['id'])
                else:
                    name = fact
                if name not in self.bag_name2idx:
                    self.bag_name2idx[name] = len(self.bag_name2idx)
                    self.bag_name.append(name)
                    self.bag_scope.append([])
                self.bag_scope[self.bag_name2idx[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0
            else: # for mirgene
                fact = (item["entity_pair"], item["entity_pair"], item["relation"])
                if item['relation'] != 'NA':
                    self.facts[fact] = 1
                if self.entpair_as_bag:
                    name = (item["entity_pair"], item["entity_pair"])
                else:
                    name = fact
                if name not in self.bag_name2idx:
                    self.bag_name2idx[name] = len(self.bag_name2idx)
                    self.bag_name.append(name)
                    self.bag_scope.append([])
                self.bag_scope[self.bag_name2idx[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05)
        self.weight = torch.from_numpy(self.weight)

    def __len__(self):
        return len(self.bags)

    def bag_generator(self):
        for index in range(len(self.bag_scope)):
            bag = self.bag_scope[index]
            if self.bag_size > 0:
                if self.bag_size <= len(bag):
                    resize_bag = random.sample(bag, self.bag_size)
                else:
                    resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
                bag = resize_bag
            rel = self.rel2id[self.data[bag[0]]['relation']]
            cur_bag = []
            for sent_id in bag:
                item = self.data[sent_id]
                indexed_tokens, pos1, pos2, mask = self.tokenizer(item)
                cur_bag.append([indexed_tokens, pos1, pos2, mask])
            self.bags.append([rel, self.bag_name[index], len(bag), cur_bag])
            # self.bag_name[index]: ('/guid/9202a8c04000641f8000000004fb88af', '/guid/9202a8c04000641f800000000054dd5d', 'NA')

    def shuffle_data(self):
        random.shuffle(self.bags)
        self.bag_name2idx = {}
        for i in range(len(self.bags)):
            self.bag_name2idx[self.bags[i][1]] = i

    def get_instance(self, bag_name, idx):
        bag_idx = self.bag_name2idx[bag_name]
        selected_instance = [self.bags[bag_idx][0]] + self.bags[bag_idx][3][idx]
        return selected_instance

    def batchify_bag(self, input, use_gpu):
        token, pos1, pos2, mask, label, scope, name = [], [], [], [], [], [], []
        start = 0
        for bag in input:
            label.append(bag[0])
            name.append(bag[1])
            scope.append((start, start + bag[2]))
            start += bag[2]
            for instance in bag[3]:
                token.append(instance[0])
                pos1.append(instance[1])
                pos2.append(instance[2])
                mask.append(instance[3])
        token = torch.Tensor(token).long()
        pos1 = torch.Tensor(pos1).long()
        pos2 = torch.Tensor(pos2).long()
        mask = torch.Tensor(mask).long()
        scope = torch.Tensor(scope).long()
        label = torch.Tensor(label).long()
        if use_gpu:
            token = token.cuda()
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()
            mask = mask.cuda()
            scope = scope.cuda()
            label = label.cuda()
        return token, pos1, pos2, mask, scope, label, name

    def batchify_sentence(self, input, use_gpu):
        token, pos1, pos2, mask, label, scope, name = [], [], [], [], [], [], []
        start = 0
        for bag in input:
            name.append(bag[1])
            scope.append((start, start + bag[2]))
            start += bag[2]
            for instance in bag[3]:
                label.append(bag[0])
                token.append(instance[0])
                pos1.append(instance[1])
                pos2.append(instance[2])
                mask.append(instance[3])
        token = torch.Tensor(token).long()
        pos1 = torch.Tensor(pos1).long()
        pos2 = torch.Tensor(pos2).long()
        mask = torch.Tensor(mask).long()
        scope = torch.Tensor(scope).long()
        label = torch.Tensor(label).long()
        if use_gpu:
            token = token.cuda()
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()
            mask = mask.cuda()
            scope = scope.cuda()
            label = label.cuda()
        return token, pos1, pos2, mask, scope, label, name


def build_data(path, loading_path, rel2id, tokenizer, args, entpair_as_bag=False):
    if os.path.exists(loading_path):
        with open(loading_path, 'rb') as fp:
            dataset = pickle.load(fp)
    else:
        global_bag = GlobalBag(path, rel2id)
        dataset = Bag_Data(global_bag.data, global_bag.rel2id, tokenizer, entpair_as_bag=entpair_as_bag,
                           bag_size=args.bag_size)
        dataset.fact_generator()
        dataset.bag_generator()
        with open(loading_path, 'wb') as fp:
            pickle.dump(dataset, fp)
    return dataset


def build_fed_data(path, loading_path, rel2id, tokenizer, args, entpair_as_bag=False):
    if os.path.exists(loading_path):
        with open(loading_path, 'rb') as fp:
            users_bag = pickle.load(fp)
    else:
        global_bag = GlobalBag(path, rel2id)
        num_items = int(len(global_bag.data) / args.num_users)
        users_sent, all_idxs = {}, [i for i in range(len(global_bag.data))]
        for i in range(args.num_users):
            sampled_idx = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - sampled_idx)
            users_sent[i] = [global_bag.data[ele] for ele in sampled_idx]
        users_bag = {}
        # users_bag_name =  {}
        for user in users_sent:
            dataset = Bag_Data(users_sent[user], global_bag.rel2id, tokenizer, entpair_as_bag=entpair_as_bag,
                               bag_size=args.bag_size)
            dataset.fact_generator()
            dataset.bag_generator()
            print("The number of bags in user %s is %s" % (str(user), str(len(dataset))), flush=True)
            print("The number of facts in user %s is %s" % (str(user), str(len(dataset.facts))), flush=True)
            # users_bag_name[user] = dataset.bag_name
            users_bag[user] = dataset
        with open(loading_path, 'wb') as fp:
            pickle.dump(users_bag, fp)
    return users_bag



