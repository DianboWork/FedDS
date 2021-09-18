import numpy as np
import random


def iid_sampling(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def doc_dependent_sampling(dataset, num_users):
    """
    Sample same docid client data from dataset
    """
    doc_dict = {}
    i = 0
    """doc_dict : key是docid，value是这个sentence在dataset中的id"""
    for ele in dataset:
        if ele[-1] not in doc_dict:
            doc_dict[ele[-1]] = [i]
            i = i + 1
        else:
            doc_dict[ele[-1]].append(i)
            i = i + 1
    num_docs = int(len(doc_dict)/num_users)
    dict_users = {}
    doc_ids = list(doc_dict.keys())
    random.shuffle(doc_ids)
    for i in range(num_users):
        sample_doc_ids = doc_ids[i:i*num_docs]
        sent_ids = []
        for doc_id in sample_doc_ids:
            sent_ids = sent_ids + doc_dict[doc_id]
        dict_users[i] = set(sent_ids)
    return dict_users


def label_dependent_sampling(dataset, num_users, excluded_label_idx=[]):
    num_shards = num_users * 2
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    if len(excluded_label_idx) != 0:
        labels = []
        sent_ids = []
        i = 0
        for data in dataset:
            if data[-2] not in excluded_label_idx:
                labels.append(data[-2])
                sent_ids.append(i)
            i = i + 1
    else:
        labels = [ele[-2] for ele in dataset]
        sent_ids = [i for i in range(len(labels))]
    labels = np.array(labels)
    sent_ids = np.array(sent_ids)
    ids_labels = np.vstack((sent_ids, labels))
    ids_labels = ids_labels[:, ids_labels[1, :].argsort()]
    sent_ids = ids_labels[0, :]
    num_sent_per_shard = int(len(labels)/num_shards)
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], sent_ids[rand * num_sent_per_shard:(rand + 1) * num_sent_per_shard]), axis=0)
    for i in range(num_users):
        dict_users[i] = set(dict_users[i])
    return dict_users
