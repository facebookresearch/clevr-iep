# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for dealing with embeddings.
"""


def convert_pretrained_wordvecs(vocab, word2vec):
  N = len(vocab['question_idx_to_token'])
  D = word2vec['vecs'].size(1)
  embed = torch.nn.Embedding(N, D)
  print(type(embed.weight))
  word2vec_word_to_idx = {w: i for i, w in enumerate(word2vec['words'])}
  print(type(word2vec['vecs']))
  for idx, word in vocab['question_idx_to_token'].items():
    word2vec_idx = word2vec_word_to_idx.get(word, None)
    if word2vec_idx is not None:
      embed.weight.data[idx] = word2vec['vecs'][word2vec_idx]
  return embed


def expand_embedding_vocab(embed, token_to_idx, word2vec=None, std=0.01):
  old_weight = embed.weight.data
  old_N, D = old_weight.size()
  new_N = 1 + max(idx for idx in token_to_idx.values())
  new_weight = old_weight.new(new_N, D).normal_().mul_(std)
  new_weight[:old_N].copy_(old_weight)

  if word2vec is not None:
    num_found = 0
    assert D == word2vec['vecs'].size(1), 'Word vector dimension mismatch'
    word2vec_token_to_idx = {w: i for i, w in enumerate(word2vec['words'])}
    for token, idx in token_to_idx.items():
      word2vec_idx = word2vec_token_to_idx.get(token, None)
      if idx >= old_N and word2vec_idx is not None:
        vec = word2vec['vecs'][word2vec_idx]
        new_weight[idx].copy_(vec)
        num_found += 1
  embed.num_embeddings = new_N
  embed.weight.data = new_weight
  return embed
