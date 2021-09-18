import random, torch
import torch.nn as nn
import torch.optim as optim
from utils.score import micro_score
from model.radam import RAdam

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr


class LocalUpdate(object):
    def __init__(self, args, train_data=None, dev_data=None, idxs=None, batchify=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if idxs:
            self.train_data = [train_data[idx] for idx in idxs]
        else:
            self.train_data = train_data
        self.dev_data = dev_data
        self.batchify = batchify

    def train(self, model, init_lr, with_pred=False, evaluation=False, no_relation_idx=0):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
       
        if self.args.local_optimizer == "SGD":
            optimizer = optim.SGD(parameters, lr=init_lr, momentum=self.args.momentum, weight_decay=self.args.l2_penalty)
        elif self.args.local_optimizer == "Adam":
            optimizer = optim.Adam(parameters, lr=init_lr, weight_decay=self.args.l2_penalty)
        elif self.args.local_optimizer == "RAdam":
            optimizer = RAdam(optimizer_grouped_parameters, lr=init_lr, weight_decay=self.args.l2_penalty)
        epoch_loss = []
        model.train()
        for iter in range(self.args.local_epoch):
            batch_loss = []
            optimizer, lr = lr_decay(optimizer, iter, self.args.lr_decay, init_lr)
            random.shuffle(self.train_data)
            model.zero_grad()
            bsz = self.args.local_bsz
            train_num = len(self.train_data)
            total_batch = train_num//bsz + 1
            for batch_id in range(total_batch):
                start = batch_id * bsz
                end = (batch_id + 1) * bsz
                if end > train_num:
                    end = train_num
                instance = self.train_data[start:end]
                if not instance:
                    continue
                input, label = self.batchify(instance, self.args.use_gpu)
                out = model(**input)
                loss = self.loss_func(out, label)
                loss.backward()
                if self.args.use_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.local_gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                if self.args.verbose and batch_id % 1000 == 0:
                    print('     Local Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                        iter, batch_id * bsz, len(self.train_data), 100. * batch_id * bsz/ len(self.train_data), loss.item()), flush=True)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        if with_pred:
            if evaluation:
                pred_onehot, pred_distrb, _, _, _, _ = self.predict(model, evaluation=evaluation, no_relation_idx=no_relation_idx)
            else:
                pred_onehot, pred_distrb = self.predict(model, evaluation=evaluation, no_relation_idx=no_relation_idx)
            return {"loss": sum(epoch_loss) / len(epoch_loss), "pred_onehot": pred_onehot, "pred_distr":pred_distrb, "final_lr": lr}
        else:
            param = model.state_dict()
            if self.args.use_gpu:
                for k, v in param.items():
                    param[k] = v.cpu()
            return {"param": param, "loss": sum(epoch_loss) / len(epoch_loss), "final_lr": lr} # , "model": model,

    def predict(self, model, pred_data=None, evaluation=False, no_relation_idx=0):
        pred_onehot = []
        pred_distrb = []
        if evaluation:
            gold_results = []
        model.eval()
        bsz = self.args.local_bsz
        if pred_data:
            pass
        else:
            pred_data = self.dev_data
        dev_num = len(pred_data)
        total_batch = dev_num // bsz + 1
        for batch_id in range(total_batch):
            start = batch_id * bsz
            end = (batch_id + 1) * bsz
            if end > dev_num:
                end = dev_num
            instance = pred_data[start:end]
            if not instance:
                continue
            input, label = self.batchify(instance, self.args.use_gpu)
            out = model(**input)
            distrb = torch.softmax(out, dim=1)
            predict = torch.argmax(distrb, dim=1).cpu()
            if self.args.use_gpu:
                tmp1 = predict.data.cpu().numpy().tolist()
                tmp2 = distrb.data.cpu().numpy().tolist()
            else:
                tmp1 = predict.data.numpy().tolist()
                tmp2 = distrb.data.numpy().tolist()
            pred_onehot = pred_onehot + tmp1
            pred_distrb = pred_distrb + tmp2
            if evaluation:
                gold = label.cpu().data.numpy().tolist()
                gold_results = gold_results + gold
        if evaluation:
            p, r, f1, verbose_information = micro_score(gold_results, pred_onehot, no_relation_idx)
            return pred_onehot, pred_distrb, p, r, f1, verbose_information
        else:
            return pred_onehot, pred_distrb
