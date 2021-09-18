import numpy as np
import datetime
import os

def eval_metric(true_y, pred_y, pred_p):
    assert len(true_y) == len(pred_y)
    positive_num = len([i for i in true_y if i[0] > 0])
    index = np.argsort(pred_p)[::-1]
    tp = 0
    fp = 0
    fn = 0
    all_pre = [0]
    all_rec = [0]
    fp_res = []
    for idx in range(len(true_y)):
        i = true_y[index[idx]]
        j = pred_y[index[idx]]
        if i[0] == 0:  # NA relation
            if j > 0:
                fp_res.append((index[idx], j, pred_p[index[idx]]))
                fp += 1
        else:
            if j == 0:
                fn += 1
            else:
                for k in i:
                    if k == -1:
                        break
                    if k == j:
                        tp += 1
                        break
        if fp + tp == 0:
            precision = 1.0
        else:
            precision = tp * 1.0 / (tp + fp)
        recall = tp * 1.0 / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)
    print("tp={}; fp={}; fn={}; positive_num={}, precsion={}, recall={}".format(tp, fp, fn, positive_num, tp/(tp+fp), tp/positive_num))
    return all_pre[1:], all_rec[1:], fp_res


def save_pr(out_dir, epoch, pre, rec, fp_res=None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    out = open('{}epoch_{}_{}_PR.txt'.format(out_dir, epoch, now), 'w')
    if fp_res is not None:
        fp_out = open('{}epoch_{}_{}_FP.txt'.format(out_dir, epoch, now), 'w')
        for idx, r, p in fp_res:
            fp_out.write('{} {} {}\n'.format(idx, r, p))
        fp_out.close()
    for p, r in zip(pre, rec):
        out.write('{} {}\n'.format(p, r))
    out.close()

