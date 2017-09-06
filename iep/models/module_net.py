#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models

from iep.models.layers import ResidualBlock, GlobalAveragePool, Flatten
import iep.programs


class ConcatBlock(nn.Module):
  def __init__(self, dim, with_residual=True, with_batchnorm=True):
    super(ConcatBlock, self).__init__()
    self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)
    self.res_block = ResidualBlock(dim, with_residual=with_residual,
                        with_batchnorm=with_batchnorm)

  def forward(self, x, y):
    out = torch.cat([x, y], 1) # Concatentate along depth
    out = F.relu(self.proj(out))
    out = self.res_block(out)
    return out


def build_stem(feature_dim, module_dim, num_layers=2, with_batchnorm=True):
  layers = []
  prev_dim = feature_dim
  for i in range(num_layers):
    layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=3, padding=1))
    if with_batchnorm:
      layers.append(nn.BatchNorm2d(module_dim))
    layers.append(nn.ReLU(inplace=True))
    prev_dim = module_dim
  return nn.Sequential(*layers)


def build_classifier(module_C, module_H, module_W, num_answers,
                     fc_dims=[], proj_dim=None, downsample='maxpool2',
                     with_batchnorm=True, dropout=0):
  layers = []
  prev_dim = module_C * module_H * module_W
  if proj_dim is not None and proj_dim > 0:
    layers.append(nn.Conv2d(module_C, proj_dim, kernel_size=1))
    if with_batchnorm:
      layers.append(nn.BatchNorm2d(proj_dim))
    layers.append(nn.ReLU(inplace=True))
    prev_dim = proj_dim * module_H * module_W
  if downsample == 'maxpool2':
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    prev_dim //= 4
  elif downsample == 'maxpool4':
    layers.append(nn.MaxPool2d(kernel_size=4, stride=4))
    prev_dim //= 16
  layers.append(Flatten())
  for next_dim in fc_dims:
    layers.append(nn.Linear(prev_dim, next_dim))
    if with_batchnorm:
      layers.append(nn.BatchNorm1d(next_dim))
    layers.append(nn.ReLU(inplace=True))
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
    prev_dim = next_dim
  layers.append(nn.Linear(prev_dim, num_answers))
  return nn.Sequential(*layers)


class ModuleNet(nn.Module):
  def __init__(self, vocab, feature_dim=(1024, 14, 14),
               stem_num_layers=2,
               stem_batchnorm=False,
               module_dim=128,
               module_residual=True,
               module_batchnorm=False,
               classifier_proj_dim=512,
               classifier_downsample='maxpool2',
               classifier_fc_layers=(1024,),
               classifier_batchnorm=False,
               classifier_dropout=0,
               verbose=True):
    super(ModuleNet, self).__init__()


    self.stem = build_stem(feature_dim[0], module_dim,
                           num_layers=stem_num_layers,
                           with_batchnorm=stem_batchnorm)
    if verbose:
      print('Here is my stem:')
      print(self.stem)

    num_answers = len(vocab['answer_idx_to_token'])
    module_H, module_W = feature_dim[1], feature_dim[2]
    self.classifier = build_classifier(module_dim, module_H, module_W, num_answers,
                                       classifier_fc_layers,
                                       classifier_proj_dim,
                                       classifier_downsample,
                                       with_batchnorm=classifier_batchnorm,
                                       dropout=classifier_dropout)
    if verbose:
      print('Here is my classifier:')
      print(self.classifier)
    self.stem_times = []
    self.module_times = []
    self.classifier_times = []
    self.timing = False

    self.function_modules = {}
    self.function_modules_num_inputs = {}
    self.vocab = vocab
    for fn_str in vocab['program_token_to_idx']:
      num_inputs = iep.programs.get_num_inputs(fn_str)
      self.function_modules_num_inputs[fn_str] = num_inputs
      if fn_str == 'scene' or num_inputs == 1:
        mod = ResidualBlock(module_dim,
                with_residual=module_residual,
                with_batchnorm=module_batchnorm)
      elif num_inputs == 2:
        mod = ConcatBlock(module_dim,
                with_residual=module_residual,
                with_batchnorm=module_batchnorm)
      self.add_module(fn_str, mod)
      self.function_modules[fn_str] = mod

    self.save_module_outputs = False

  def expand_answer_vocab(self, answer_to_idx, std=0.01, init_b=-50):
    # TODO: This is really gross, dipping into private internals of Sequential
    final_linear_key = str(len(self.classifier._modules) - 1)
    final_linear = self.classifier._modules[final_linear_key]

    old_weight = final_linear.weight.data
    old_bias = final_linear.bias.data
    old_N, D = old_weight.size()
    new_N = 1 + max(answer_to_idx.values())
    new_weight = old_weight.new(new_N, D).normal_().mul_(std)
    new_bias = old_bias.new(new_N).fill_(init_b)
    new_weight[:old_N].copy_(old_weight)
    new_bias[:old_N].copy_(old_bias)

    final_linear.weight.data = new_weight
    final_linear.bias.data = new_bias

  def _forward_modules_json(self, feats, program):
    def gen_hook(i, j):
      def hook(grad):
        self.all_module_grad_outputs[i][j] = grad.data.cpu().clone()
      return hook

    self.all_module_outputs = []
    self.all_module_grad_outputs = []
    # We can't easily handle minibatching of modules, so just do a loop
    N = feats.size(0)
    final_module_outputs = []
    for i in range(N):
      if self.save_module_outputs:
        self.all_module_outputs.append([])
        self.all_module_grad_outputs.append([None] * len(program[i]))
      module_outputs = []
      for j, f in enumerate(program[i]):
        f_str = iep.programs.function_to_str(f)
        module = self.function_modules[f_str]
        if f_str == 'scene':
          module_inputs = [feats[i:i+1]]
        else:
          module_inputs = [module_outputs[j] for j in f['inputs']]
        module_outputs.append(module(*module_inputs))
        if self.save_module_outputs:
          self.all_module_outputs[-1].append(module_outputs[-1].data.cpu().clone())
          module_outputs[-1].register_hook(gen_hook(i, j))
      final_module_outputs.append(module_outputs[-1])
    final_module_outputs = torch.cat(final_module_outputs, 0)
    return final_module_outputs

  def _forward_modules_ints_helper(self, feats, program, i, j):
    used_fn_j = True
    if j < program.size(1):
      fn_idx = program.data[i, j]
      fn_str = self.vocab['program_idx_to_token'][fn_idx]
    else:
      used_fn_j = False
      fn_str = 'scene'
    if fn_str == '<NULL>':
      used_fn_j = False
      fn_str = 'scene'
    elif fn_str == '<START>':
      used_fn_j = False
      return self._forward_modules_ints_helper(feats, program, i, j + 1)
    if used_fn_j:
      self.used_fns[i, j] = 1
    j += 1
    module = self.function_modules[fn_str]
    if fn_str == 'scene':
      module_inputs = [feats[i:i+1]]
    else:
      num_inputs = self.function_modules_num_inputs[fn_str]
      module_inputs = []
      while len(module_inputs) < num_inputs:
        cur_input, j = self._forward_modules_ints_helper(feats, program, i, j)
        module_inputs.append(cur_input)
    module_output = module(*module_inputs)
    return module_output, j

  def _forward_modules_ints(self, feats, program):
    """
    feats: FloatTensor of shape (N, C, H, W) giving features for each image
    program: LongTensor of shape (N, L) giving a prefix-encoded program for
      each image.
    """
    N = feats.size(0)
    final_module_outputs = []
    self.used_fns = torch.Tensor(program.size()).fill_(0)
    for i in range(N):
      cur_output, _ = self._forward_modules_ints_helper(feats, program, i, 0)
      final_module_outputs.append(cur_output)
    self.used_fns = self.used_fns.type_as(program.data).float()
    final_module_outputs = torch.cat(final_module_outputs, 0)
    return final_module_outputs

  def forward(self, x, program):
    N = x.size(0)
    assert N == len(program)

    feats = self.stem(x)

    if type(program) is list or type(program) is tuple:
      final_module_outputs = self._forward_modules_json(feats, program)
    elif type(program) is Variable and program.dim() == 2:
      final_module_outputs = self._forward_modules_ints(feats, program)
    else:
      raise ValueError('Unrecognized program format')

    # After running modules for each input, concatenat the outputs from the
    # final module and run the classifier.
    out = self.classifier(final_module_outputs)
    return out
