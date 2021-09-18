from models.multi_instance import Multi_Instance
from models.bag_attention import BagAttention
from models.intra_bag_attention import IntraBagAttention
from models.bag_average import BagAverage
from models.pcnn_encoder import PCNNEncoder
from trainer.centralized_trainer import Centralized_Trainer
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
training_arg = add_argument_group('Learning', parser)
training_arg.add_argument('--bag_size', type=int, default=0)
training_arg.add_argument('--seed', type=int, default=1)
training_arg.add_argument('--batch_size', type=int, default=24)
training_arg.add_argument('--max_epoch', type=int, default=100)
training_arg.add_argument('--lr', type=float, default=0.1)
training_arg.add_argument('--lr_decay', type=float, default=0.01)
training_arg.add_argument('--weight_decay', type=float, default=1e-5)
training_arg.add_argument('--optimizer', type=str, default="SGD", choices=['Adam', 'SGD', 'AdamW'])
training_arg.add_argument('--ds_algo', type=str, default="one", choices=['attn', 'one', 'avg', 'intra'])
training_arg.add_argument('--loss_weight', type=str2bool, default=False)
training_arg.add_argument('--use_gpu', type=str2bool, default=True)
training_arg.add_argument('--gpu', type=int, default=0)
encoder_arg = add_argument_group('Encoder', parser)
encoder_arg.add_argument('--max_length', type=int, default=128)
encoder_arg.add_argument('--word_size', type=int, default=50)
encoder_arg.add_argument('--hidden_size', type=int, default=230)
encoder_arg.add_argument('--position_size', type=int, default=5)
encoder_arg.add_argument('--kernel_size', type=int, default=3)
encoder_arg.add_argument('--padding_size', type=int, default=1)
encoder_arg.add_argument('--blank_padding', type=str2bool, default=True)
encoder_arg.add_argument('--mask_entity', type=str2bool, default=False)
encoder_arg.add_argument('--dropout', type=float, default=0.1)
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
else:
    raise(ValueError("Unsupported Distant Algorithm: %s" % (args.ds_algo)))

if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
    os.mkdir("ckpt/data")
    os.mkdir("ckpt/param")
ckpt = 'ckpt'


train_path = os.path.join(data_root, 'nyt10/nyt10_train.txt')
val_path = os.path.join(data_root, 'nyt10/nyt10_val.txt')
test_path = os.path.join(data_root, 'nyt10/nyt10_test.txt')

framework = Centralized_Trainer(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    model=model,
    ckpt=ckpt,
    args=args)

# Train the model
framework.train_model()

# from utils.data import BagREDataset, BagRELoader
# train_loader = BagRELoader(
#                 train_path,
#                 model.rel2id,
#                 model.sentence_encoder.tokenize,
#                 batch_size =11,
#                 shuffle = True,
#                 bag_size= 0,
#                 entpair_as_bag=False)
# #
# for iter, data in enumerate(train_loader):
#     label = data[0]
#     bag_name = data[1]
#     scope = data[2]
#     args = data[3:]
#     logits = model(label, scope, *args, 0, train=False)
#     logits = model(label, scope, *args, 0, train=True)

