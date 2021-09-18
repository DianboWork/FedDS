# """WordpieceTokenizer classes."""
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import numpy as np
# from utils.function import (load_vocab,
#                    convert_to_unicode,
#                    clean_text,
#                    split_on_whitespace,
#                    tokenize_chinese_chars, split_on_punctuation, build_embedding)
# import os
#
#
# class EmbedTokenizer(object):
#     """Runs WordPiece tokenziation."""
#     def __init__(self, path=None, unk_token="[UNK]", pad_token="[PAD]"):
#         self.path = path
#         self.vocab = load_vocab(self.path)
#         self.unk_token = unk_token
#         self.pad_token = pad_token
#         self.init_embedding()
#
#         # if not self.unk_token:
#         #     self.vocab[unk_token] = len(self.vocab)
#         # if not self.pad_token:
#         #     self.vocab[pad_token] = len(self.vocab)
#         # self.inv_vocab = {v: k for k, v in self.vocab.items()}
#         # self.vocab_num = len(self.vocab)
#         # self.embedding = None
#         # self.embedding_dim = 0
#
#     def init_embedding(self):
#         if len(self.vocab) != 0:
#             if os.path.exitst(os.path.join(self.path, "embedding.npy")):
#                 self.load_embedding(self.path)
#                 if self.embedding.shape[0] != len(self.vocab):
#                     print("Embedding file (%s) doesn't match with the vocab file (%s)." %(os.path.join(self.path, "embedding.npy"), os.path.join(self.path, "vocab.txt")))
#                     os.remove(os.path.join(self.path, "embedding.npy"))
#                     os.remove(os.path.join(self.path, "vocab.txt"))
#                     self.embedding = None
#                     self.embedding_dim = 0
#                     self.vocab = {}
#                     self.keep_growing = True
#                     if os.path.exists(os.path.join(self.path, 'embedding.txt')):
#                         print("Reloading vocab and embedding from %s" % (os.path.join(self.path, 'embedding.txt')))
#                     else:
#                         print("Embeding file (%s) is not exist and the embedding matrix will be randomly initialized." %(os.path.join(self.path, 'embedding.txt')))
#                 else:
#                     self.keep_growing = False
#                     if self.unk_token not in self.vocab:
#                         self.vocab[self.unk_token] = len(self.vocab)
#                         scale = np.sqrt(3.0 / self.embedding_dim)
#                         self.embedding = np.concatenate((self.embedding, np.random.uniform(-scale, scale, [1, self.embedding_dim])), 0)
#                     if self.pad_token not in self.vocab:
#                         self.vocab[self.pad_token] = len(self.vocab)
#                         self.embedding = np.concatenate(
#                             (self.embedding, np.zeros([1, self.embedding_dim])), 0)
#             else:
#                 self.embedding = None
#                 self.embedding_dim = 0
#                 self.vocab = {}
#                 self.keep_growing = False
#
#         else:
#             self.keep_growing = True
#
#         self.inv_vocab = {v: k for k, v in self.vocab.items()}
#         self.vocab_num = len(self.vocab)
#
#     def tokenize(self, text):
#         """
#             Args:
#                 text: A single token or whitespace separated tokens.
#             Returns:
#                 output_tokens: A list of wordpiece tokens.
#         """
#         text = convert_to_unicode(text)
#         text = clean_text(text)
#         text = tokenize_chinese_chars(text)
#         # text = " ".join(split_on_punctuation(text))
#         token_list = split_on_whitespace(text)
#         return token_list
#
#     def convert_tokens_to_ids(self, tokens, max_seq_length=None, uncased=True, keep_growing=False):
#
#         output = []
#         for token in tokens:
#             if uncased:
#                 token = token.lower()
#             if token in self.vocab:
#                 output.append(self.vocab[token])
#             else:
#                 if self.keep_growing:
#                     self.vocab[token] = len(self.vocab)
#                     output.append(self.vocab[token])
#                 else:
#                     output.append(self.unk_token)
#         if max_seq_length != None:
#             if len(output) > max_seq_length:
#                 output = output[:max_seq_length]
#             else:
#                 while len(output) < max_seq_length:
#                     if self.pad_token:
#                         output.append(self.vocab[self.pad_token])
#                     else:
#                         break
#         if self.vocab_num != len(self.vocab):
#             self.inv_vocab = {v: k for k, v in self.vocab.items()}
#             self.vocab_num = len(self.vocab)
#         return output
#
#     def convert_ids_to_tokens(self, ids):
#         output = []
#         for id in ids:
#             if id in self.inv_vocab:
#                 output.append(self.inv_vocab[id])
#             else:
#                 raise ValueError("Unsupported id: %s, Current vocab size: %s" % (str(id), str(self.vocab_num)))
#         return output
#
#     def load_embedding(self, skip_first_row=False, separator=" ", embedd_dim=100, norm=True, random_init=False):
#         if os.path.exists(os.path.join(self.path, 'vocab.txt')) and os.path.exitst(os.path.join(self.path, "embedding.npy")):
#             self.embedding = np.load(os.path.join(self.path, "embedding.npy"))
#         else:
#             self.embedding, self.embedding_dim = build_embedding(self, skip_first_row=skip_first_row, separator=separator, embedd_dim=embedd_dim, norm=norm, random_init=random_init)
#
#     def save(self, path):
#         """Save the tokenizer vocabulary and embedding to a directory."""
#         index = 0
#         if os.path.isdir(path):
#             vocab_file = os.path.join(path, 'vocab.txt')
#             embed_file = os.path.join(path, "embedding.npy")
#         else:
#             raise ValueError("Unsupported saving path: %s" % (path))
#         with open(vocab_file, "w", encoding="utf-8") as writer:
#             for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
#                 if index != token_index:
#                     print("Saving vocabulary to {}: vocabulary indices are not consecutive."
#                                    " Please check that the vocabulary is not corrupted!".format(vocab_file))
#                     index = token_index
#                 writer.write(token + u'\n')
#                 index += 1
#         np.save(self.embedding, embed_file)
#         self.keep_growing = False
#
#
#
#
