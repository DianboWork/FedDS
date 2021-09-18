import torch
from torch import nn, optim
from utils.data import build_data
from utils.average_meter import AverageMeter
from tqdm import tqdm
import gc
import numpy as np
import sklearn.metrics


class Centralized_Trainer(nn.Module):
    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 args
                 ):
        super().__init__()
        self.max_epoch = args.max_epoch
        self.bag_size = args.bag_size
        self.args = args
        # Load data
        if train_path is not None:
            self.train_loader = build_data(
                train_path,
                ckpt + '/data/train.pkl',
                model.rel2id,
                model.sentence_encoder.tokenize,
                args,
                entpair_as_bag=False)
        if val_path is not None:
            self.val_loader = build_data(
                val_path,
                ckpt + '/data/dev.pkl',
                model.rel2id,
                model.sentence_encoder.tokenize,
                args,
                entpair_as_bag=True)
        if test_path is not None:
            self.test_loader = build_data(
                test_path,
                ckpt + '/data/test.pkl',
                model.rel2id,
                model.sentence_encoder.tokenize,
                args,
                entpair_as_bag=True)

        # Model
        self.model = model
        # Criterion
        if args.loss_weight:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.model.parameters()

        if args.optimizer == 'SGD':
            self.optimizer = optim.SGD(params, args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            self.optimizer = optim.Adam(params, args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            from transformers import AdamW
            params = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': args.weight_decay,
                    'lr': args.lr,
                    'ori_lr': args.lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': args.lr,
                    'ori_lr': args.lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")
        # Cuda
        if self.args.use_gpu:
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self):
        best_auc = 0
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            if self.args.optimizer != "AdamW":
                self.optimizer, _ = self.lr_decay(self.optimizer, epoch, self.args.lr_decay, self.args.lr)
            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()
            self.train_loader.shuffle_data()
            batch_size = self.args.batch_size
            train_num = len(self.train_loader)
            total_batch = train_num // batch_size + 1
            for batch_id in tqdm(range(total_batch)):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                batch_bags = self.train_loader.bags[start:end]
                if not batch_bags:
                    continue
                token, pos1, pos2, mask, scope, label, _ = self.train_loader.batchify_bag(batch_bags, self.args.use_gpu)
                logits = self.model(label, scope, token, pos1, pos2, mask, bag_size=self.bag_size)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1)  # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                # t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)

                # Optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            print("    loss: %.4f; acc: %.4f; pos_acc: %.4f" % (avg_loss.avg, avg_acc.avg, avg_pos_acc.avg), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            # Val
            # print("=== Epoch %d Validation ===" % epoch)
            # result = self.eval_model(self.val_loader)
            print("=== Epoch %d Test ===" % epoch, flush=True)
            result = self.eval_model(self.test_loader)
            if result['auc'] > best_auc:
                print("Best ckpt and saved.", flush=True)
                # torch.save({'state_dict': self.model.state_dict()}, self.ckpt+"/param/%s_epoch_%d_auc_%.4f.model" %(self.model.name, epoch, result['auc']))
                best_auc = result['auc']
            gc.collect()
            torch.cuda.empty_cache()
        print("Best auc on val set: %f" % (best_auc), flush=True)

    def eval_model(self, eval_loader):
        self.model.eval()
        with torch.no_grad():
            pred_result = []
            batch_size = self.args.batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                batch_bags = eval_loader.bags[start:end]
                if not batch_bags:
                    continue
                token, pos1, pos2, mask, scope, label, bag_name = eval_loader.batchify_bag(batch_bags, self.args.use_gpu)
                logits = self.model(label, scope, token, pos1, pos2, mask, train=False, bag_size=self.bag_size)
                for i in range(logits.size(0)):
                    for relid in range(self.model.num_class):
                        if self.model.id2rel[relid] != 'NA':
                            pred_result.append({
                                'entpair': bag_name[i][:2],
                                'relation': self.model.id2rel[relid],
                                'score': logits[i][relid].item()
                            })

            sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
            prec = []
            rec = []
            correct = 0
            total = len(eval_loader.facts)
            score_100 = np.array([ele["score"] for ele in sorted_pred_result[:100]])
            print("Top 100 logit value. max: %.4f; min: %.4f; mean: %.4f; var: %.8f" % (
                np.max(score_100), np.min(score_100), np.mean(score_100), np.var(score_100)), flush=True)
            score_200 = np.array([ele["score"] for ele in sorted_pred_result[:200]])
            print("Top 200 logit value. max: %.4f; min: %.4f; mean: %.4f; var: %.8f" % (
                np.max(score_200), np.min(score_200), np.mean(score_200), np.var(score_200)), flush=True)
            score_300 = np.array([ele["score"] for ele in sorted_pred_result[:300]])
            print("Top 300 logit value. max: %.4f; min: %.4f; mean: %.4f; var: %.8f" % (
                np.max(score_300), np.min(score_300), np.mean(score_300), np.var(score_300)), flush=True)
            for i, item in enumerate(sorted_pred_result):
                if (item['entpair'][0], item['entpair'][1], item['relation']) in eval_loader.facts:
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
            auc_01 = self.auc_f(np_rec, np_prec, 0.1)
            auc_02 = self.auc_f(np_rec, np_prec, 0.2)
            auc_03 = self.auc_f(np_rec, np_prec, 0.3)
            auc_04 = self.auc_f(np_rec, np_prec, 0.4)
            # print('AUC@0.2: %.4f' % auc_01)
            # print('AUC@0.3: %.4f' % auc_02)
            # print('AUC@0.4: %.4f' % auc_03)
            # print('AUC@0.5: %.4f' % auc_04)

            f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
            # print("f1: %.4f" % (f1))
            mean_prec = np_prec.mean()
        return {'prec': np_prec, 'rec': np_rec, 'mean_prec': mean_prec, 'f1': f1, 'auc': auc_2000,
                    'auc_detail': [auc_01, auc_02, auc_03, auc_04], 'P@100': prec[100], 'P@200': prec[200],
                    'p@300': prec[300]}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def auc_f(np_rec, np_prec, value):
        recalls = np_rec[np_rec < value]
        precisions = np_prec[np_rec < value]
        auc_v = sklearn.metrics.auc(x=recalls, y=precisions)
        return auc_v

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate, init_lr):
        lr = init_lr * ((1 - decay_rate) ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer, lr