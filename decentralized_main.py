from models.multi_instance import Multi_Instance
from models.bag_attention import BagAttention
from models.intra_bag_attention import IntraBagAttention
from models.bag_average import BagAverage
from models.bag_all_right import Bag_All_Right
from models.pcnn_encoder import PCNNEncoder
from trainer.decentralized_trainer import Decentralized_Trainer
import json, random, torch
import os
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name, parser):
    arg = parser.add_argument_group(name)
    return arg


parser = argparse.ArgumentParser()
training_arg = add_argument_group('Learning Hyperparamters', parser)
training_arg.add_argument('--bag_size', type=int, default=0)
training_arg.add_argument('--seed', type=int, default=1)
training_arg.add_argument('--batch_size', type=int, default=8)
training_arg.add_argument('--max_epoch', type=int, default=50)
training_arg.add_argument('--lr', type=float, default=0.01)
training_arg.add_argument('--lr_decay', type=float, default=0.01)
training_arg.add_argument('--weight_decay', type=float, default=0.00001)
training_arg.add_argument('--loss_weight', type=str2bool, default=False)
training_arg.add_argument('--use_gpu', type=str2bool, default=True)
training_arg.add_argument('--gpu', type=int, default=0)
training_arg.add_argument('--ds_algo', type=str, default="none", choices=['attn', 'one', 'avg', 'intra', 'none'])
encoder_arg = add_argument_group('Encoder Hyperparamters', parser)
encoder_arg.add_argument('--max_length', type=int, default=128)
encoder_arg.add_argument('--word_size', type=int, default=50)
encoder_arg.add_argument('--hidden_size', type=int, default=230)
encoder_arg.add_argument('--position_size', type=int, default=5)
encoder_arg.add_argument('--kernel_size', type=int, default=3)
encoder_arg.add_argument('--padding_size', type=int, default=1)
encoder_arg.add_argument('--blank_padding', type=str2bool, default=True)
encoder_arg.add_argument('--mask_entity', type=str2bool, default=False)
encoder_arg.add_argument('--dropout', type=float, default=0.5)
fed_arg = add_argument_group('Federated Hyperparamters', parser)
fed_arg.add_argument('--optimizer', type=str, default="SGD", choices=['Adam', 'SGD', 'AdamW'])
fed_arg.add_argument('--fed_algo', type=str, default="fed_avg", choices=['fed_avg', 'fed_attn'])
fed_arg.add_argument('--num_users', type=int, default=100)
fed_arg.add_argument('--frac', type=float, default=0.1)
fed_arg.add_argument('--local_epoch', type=int, default=3)
fed_arg.add_argument('--dp', type=float, default=0)

args = parser.parse_args()
print(args, flush=True)


seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

data_root = "./data/"
# rel2id = {"NA": 0, "R_MIRTAR": 1}
# word2id = json.load(open(os.path.join(data_root, 'BioWordVec/word2id.json')))
# word2vec = np.load(os.path.join(data_root, 'BioWordVec/embedding_200.npy'))
rel2id = json.load(open(os.path.join(data_root, 'nyt10/nyt10_rel2id.json')))
word2id = json.load(open(os.path.join(data_root, 'glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(data_root, 'glove/glove.6B.50d_mat.npy'))


sentence_encoder = PCNNEncoder(
    token2id=word2id,
    args=args,
    word2vec=word2vec
)
if args.ds_algo =="attn":
    model = BagAttention(sentence_encoder, len(rel2id), rel2id)
elif args.ds_algo == "one":
    model = Multi_Instance(sentence_encoder, len(rel2id), rel2id)
elif args.ds_algo == "avg":
    model = BagAverage(sentence_encoder, len(rel2id), rel2id)
elif args.ds_algo == "intra":
    model = IntraBagAttention(sentence_encoder, len(rel2id), rel2id)
elif args.ds_algo == "none":
    model = Bag_All_Right(sentence_encoder, len(rel2id), rel2id)
else:
    raise(ValueError("Unsupported Distant Algorithm: %s" % (args.ds_algo)))

# if not os.path.exists('mirgene_ckpt'):
#     os.mkdir('mirgene_ckpt')
#     os.mkdir("mirgene_ckpt/data")
#     os.mkdir("mirgene_ckpt/param")
# ckpt = 'mirgene_ckpt'
#
# train_path = os.path.join(data_root, 'mirgene/mirgene_train.txt')
# val_path = None
# test_path = os.path.join(data_root, 'mirgene/mirgene_test.txt')


if not os.path.exists('nyt_ckpt'):
    os.mkdir('nyt_ckpt')
    os.mkdir("nyt_ckpt/data")
    os.mkdir("nyt_ckpt/param")
ckpt = 'nyt_ckpt'


train_path = os.path.join(data_root, 'nyt10/nyt10_train.txt')
val_path = os.path.join(data_root, 'nyt10/nyt10_val.txt')
test_path = os.path.join(data_root, 'nyt10/nyt10_test.txt')

framework = Decentralized_Trainer(
    train_path=train_path,
    val_path=None,
    test_path=test_path,
    model=model,
    ckpt=ckpt,
    args=args)


framework.train_model()
