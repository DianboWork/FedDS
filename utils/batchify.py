import torch


def batchify(input_batch_list, gpu):
    batch_size = len(input_batch_list)
    words = [ele[0] for ele in input_batch_list]
    posi1s = [ele[1] for ele in input_batch_list]
    posi2s = [ele[2] for ele in input_batch_list]
    labels = [ele[3] for ele in input_batch_list]
    ent1_ids = [ele[4] for ele in input_batch_list]
    ent2_ids = [ele[5] for ele in input_batch_list]
    seq_lens = list(map(len, words))
    max_seq_len = max(seq_lens)
    word_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    posi1_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    posi2_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=False).byte()
    label_tensor = torch.zeros(batch_size, requires_grad=False).long()
    ent1_tensor = torch.zeros(batch_size, requires_grad=False).long()
    ent2_tensor = torch.zeros(batch_size, requires_grad=False).long()
    for idx, (word, posi1, posi2, label, seq_len, ent1_id, ent2_id) in enumerate(zip(words, posi1s, posi2s, labels, seq_lens, ent1_ids, ent2_ids)):
        word_tensor[idx, :seq_len] = torch.LongTensor(word)
        posi1_tensor[idx, :seq_len] = torch.LongTensor(posi1)
        posi2_tensor[idx, :seq_len] = torch.LongTensor(posi2)
        mask[idx, seq_len:] = torch.Tensor([1] * (max_seq_len-seq_len))
        label_tensor[idx] = label
        ent1_tensor[idx] = ent1_id
        ent2_tensor[idx] = ent2_id
    seq_lens = torch.LongTensor(seq_lens)
    seq_lens, perm_idx = seq_lens.sort(0, descending=True)
    word_tensor = word_tensor[perm_idx]
    posi1_tensor = posi1_tensor[perm_idx]
    posi2_tensor = posi2_tensor[perm_idx]
    mask = mask[perm_idx]
    label_tensor = label_tensor[perm_idx]
    ent1_tensor = ent1_tensor[perm_idx]
    ent2_tensor = ent2_tensor[perm_idx]
    _, seq_recover = perm_idx.sort(0, descending=False)
    if gpu:
        seq_lens = seq_lens.cuda()
        word_tensor = word_tensor.cuda()
        posi1_tensor = posi1_tensor.cuda()
        posi2_tensor = posi2_tensor.cuda()
        mask = mask.cuda()
        label_tensor = label_tensor.cuda()
        ent1_tensor = ent1_tensor.cuda()
        ent2_tensor = ent2_tensor.cuda()
    return word_tensor, posi1_tensor, posi2_tensor, mask, label_tensor, seq_lens, ent1_tensor, ent2_tensor
