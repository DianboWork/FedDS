#
import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics



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
                if item['relation'] != 'NA':
                    fact = (item['h']['id'], item['t']['id'], item['relation'])
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


# class SubBag:
#     def __init__(self, data, rel2id, tokenizer, entpair_as_bag=False):
#         self.tokenizer = tokenizer
#         self.rel2id = rel2id
#         self.entpair_as_bag = entpair_as_bag
#         self.data = data
#         self.bag_scope = []
#         self.bags = []
#         self.name2id = {}
#         self.facts = {}
#
#     def parse_data(self):
#         for idx, item in enumerate(self.data):
#             if self.entpair_as_bag:
#                 name = (item['h']['id'], item['t']['id'])
#             else:
#                 name = (item['h']['id'], item['t']['id'], item['relation'])
#             if name not in self.name2id:
#                 self.name2id[name] = len(self.name2id)
#                 self.bag_scope.append([])
#             self.bag_scope[self.name2id[name]].append(idx)
#
#


class BagREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """

    def __init__(self, data, rel2id, tokenizer, entpair_as_bag=False, bag_size=0, mode=None):
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

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)

        if mode == None:
            self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
            self.bag_scope = []
            self.name2id = {}
            self.bag_name = []
            self.facts = {}
            for idx, item in enumerate(self.data):
                fact = (item['h']['id'], item['t']['id'], item['relation'])
                if item['relation'] != 'NA':
                    self.facts[fact] = 1
                if entpair_as_bag:
                    name = (item['h']['id'], item['t']['id'])
                else:
                    name = fact
                if name not in self.name2id:
                    self.name2id[name] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(name)
                self.bag_scope[self.name2id[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0
            self.weight = 1.0 / (self.weight ** 0.05)
            self.weight = torch.from_numpy(self.weight)
        else:
            pass

    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag

        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer(item))
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (n, L), n is the size of bag
        return [rel, self.bag_name[index], len(bag)] + seqs

    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (sumn, sent_length)
            seqs[i] = seqs[i].expand((1,) + seqs[i].size()) # (1, , 120)
            # multi GPU
            # seqs[i] = seqs[i].expand((torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1,) + seqs[i].size())
        scope = []  # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert (start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long()  # (B)
        return [label, bag_name, scope] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0)  # (batch, bag, L)
        scope = []  # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long()  # (B)
        return [label, bag_name, scope] + seqs

    def eval(self, pred_result):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        score_100 = np.array([ele["score"] for ele in sorted_pred_result[:100]])
        print("Top 100 logit value. max: %.4f; min: %.4f; mean: %.4f; var: %.8f" % (np.max(score_100), np.min(score_100), np.mean(score_100), np.var(score_100)), flush=True)
        score_200 = np.array([ele["score"] for ele in sorted_pred_result[:200]])
        print("Top 200 logit value. max: %.4f; min: %.4f; mean: %.4f; var: %.8f" % (np.max(score_200), np.min(score_200), np.mean(score_200), np.var(score_200)), flush=True)
        score_300 = np.array([ele["score"] for ele in sorted_pred_result[:300]])
        print("Top 300 logit value. max: %.4f; min: %.4f; mean: %.4f; var: %.8f" % (np.max(score_300), np.min(score_300),np.mean(score_300), np.var(score_300)), flush=True)
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))

        # auc = sklearn.metrics.auc(x=rec, y=prec)
        # print('AUC: %.4f' % auc)
        auc_2000 = sklearn.metrics.auc(x=rec[:2000], y=prec[:2000])
        print('AUC Top 2000: %.4f' % auc_2000, flush=True)
        print('P@100: %.4f' % prec[100], flush=True)
        print('P@200: %.4f' % prec[200], flush=True)
        print('P@300: %.4f' % prec[300], flush=True)
        print('Mean: %.4f' % ((float(prec[300]) + float(prec[200]) + float(prec[100])) / 3), flush=True)

        np_rec = np.array(rec)
        np_prec = np.array(prec)
        auc_01 = auc_f(np_rec, np_prec, 0.1)
        auc_02 = auc_f(np_rec, np_prec, 0.2)
        auc_03 = auc_f(np_rec, np_prec, 0.3)
        auc_04 = auc_f(np_rec, np_prec, 0.4)
        # print('AUC@0.2: %.4f' % auc_01)
        # print('AUC@0.3: %.4f' % auc_02)
        # print('AUC@0.4: %.4f' % auc_03)
        # print('AUC@0.5: %.4f' % auc_04)

        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        # print("f1: %.4f" % (f1))
        mean_prec = np_prec.mean()
        return {'prec': np_prec, 'rec': np_rec, 'mean_prec': mean_prec, 'f1': f1, 'auc': auc_2000, 'auc_detail': [auc_01, auc_02, auc_03, auc_04], 'P@100':prec[100], 'P@200': prec[200], 'p@300': prec[300]}


def BagRELoader(path, rel2id, tokenizer, args, batch_size,
                shuffle, entpair_as_bag=False, num_workers=0):
    if args.bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    print("Loading data from "+path, flush=True)
    global_bag = GlobalBag(path, rel2id)
    dataset = BagREDataset(global_bag.data, global_bag.rel2id, tokenizer, entpair_as_bag=entpair_as_bag,
                           bag_size=args.bag_size)
    print("The number of bags is " + str(len(dataset)), flush=True)
    print("The number of facts is " + str(len(dataset.facts)), flush=True)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
#
def auc_f(np_rec, np_prec, value):
    recalls = np_rec[np_rec < value]
    precisions = np_prec[np_rec < value]
    auc_v = sklearn.metrics.auc(x=recalls, y=precisions)
    return auc_v


def iid_sampling(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        sampled_idx = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - sampled_idx)
        dict_users[i] = [dataset[ele] for ele in sampled_idx]
    return dict_users


def BagRE_Fed_Loader(path, rel2id, tokenizer, args, batch_size,
                shuffle, entpair_as_bag=False, num_workers=0):
    if args.bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    print("Loading data from "+path, flush=True)
    global_bag = GlobalBag(path, rel2id)
    users_sent = iid_sampling(global_bag.data, args.num_users)
    users_bag = {}
    # users_bag_name =  {}
    for user in users_sent:
        dataset = BagREDataset(users_sent[user], global_bag.rel2id, tokenizer, entpair_as_bag=entpair_as_bag,
                           bag_size=args.bag_size)
        print("The number of bags in user %s is %s" % (str(user), str(len(dataset))), flush=True)
        print("The number of facts in user %s is %s" % (str(user), str(len(dataset.facts))), flush=True)
        # users_bag_name[user] = dataset.bag_name
        users_bag[user] = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      pin_memory=True,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn)
    return users_bag # , users_bag_name