#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

import argparse
import json
import random
import shutil

import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import h5py

import iep.utils as utils
import iep.preprocess
from iep.data import ClevrDataset, ClevrDataLoader
from iep.models import ModuleNet, Seq2Seq, LstmModel, CnnLstmModel, CnnLstmSaModel


parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_question_h5', default='data/train_questions.h5')
parser.add_argument('--train_features_h5', default='data/train_features.h5')
parser.add_argument('--val_question_h5', default='data/val_questions.h5')
parser.add_argument('--val_features_h5', default='data/val_features.h5')
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default='data/vocab.json')

parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=10000, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
        choices=['PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# RNN options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
        choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
        choices=['maxpool2', 'maxpool4', 'none'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='data/checkpoint.pt')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--record_loss_every', type=int, default=1)
parser.add_argument('--checkpoint_every', default=10000, type=int)


def main(args):
  if args.randomize_checkpoint_path == 1:
    name, ext = os.path.splitext(args.checkpoint_path)
    num = random.randint(1, 1000000)
    args.checkpoint_path = '%s_%06d%s' % (name, num, ext)

  vocab = utils.load_vocab(args.vocab_json)

  if args.use_local_copies == 1:
    shutil.copy(args.train_question_h5, '/tmp/train_questions.h5')
    shutil.copy(args.train_features_h5, '/tmp/train_features.h5')
    shutil.copy(args.val_question_h5, '/tmp/val_questions.h5')
    shutil.copy(args.val_features_h5, '/tmp/val_features.h5')
    args.train_question_h5 = '/tmp/train_questions.h5'
    args.train_features_h5 = '/tmp/train_features.h5'
    args.val_question_h5 = '/tmp/val_questions.h5'
    args.val_features_h5 = '/tmp/val_features.h5'

  question_families = None
  if args.family_split_file is not None:
    with open(args.family_split_file, 'r') as f:
      question_families = json.load(f)

  train_loader_kwargs = {
    'question_h5': args.train_question_h5,
    'feature_h5': args.train_features_h5,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'shuffle': args.shuffle_train_data == 1,
    'question_families': question_families,
    'max_samples': args.num_train_samples,
    'num_workers': args.loader_num_workers,
  }
  val_loader_kwargs = {
    'question_h5': args.val_question_h5,
    'feature_h5': args.val_features_h5,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'question_families': question_families,
    'max_samples': args.num_val_samples,
    'num_workers': args.loader_num_workers,
  }

  with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
       ClevrDataLoader(**val_loader_kwargs) as val_loader:
    train_loop(args, train_loader, val_loader)

  if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
    os.remove('/tmp/train_questions.h5')
    os.remove('/tmp/train_features.h5')
    os.remove('/tmp/val_questions.h5')
    os.remove('/tmp/val_features.h5')


def train_loop(args, train_loader, val_loader):
  vocab = utils.load_vocab(args.vocab_json)
  program_generator, pg_kwargs, pg_optimizer = None, None, None
  execution_engine, ee_kwargs, ee_optimizer = None, None, None
  baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
  baseline_type = None

  pg_best_state, ee_best_state, baseline_best_state = None, None, None

  # Set up model
  if args.model_type == 'PG' or args.model_type == 'PG+EE':
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = torch.optim.Adam(program_generator.parameters(),
                                    lr=args.learning_rate)
    print('Here is the program generator:')
    print(program_generator)
  if args.model_type == 'EE' or args.model_type == 'PG+EE':
    execution_engine, ee_kwargs = get_execution_engine(args)
    ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                    lr=args.learning_rate)
    print('Here is the execution engine:')
    print(execution_engine)
  if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
    baseline_model, baseline_kwargs = get_baseline_model(args)
    params = baseline_model.parameters()
    if args.baseline_train_only_rnn == 1:
      params = baseline_model.rnn.parameters()
    baseline_optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    print('Here is the baseline model')
    print(baseline_model)
    baseline_type = args.model_type
  loss_fn = torch.nn.CrossEntropyLoss().cuda()

  stats = {
    'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
    'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
    'best_val_acc': -1, 'model_t': 0,
  }
  t, epoch, reward_moving_average = 0, 0, 0

  set_mode('train', [program_generator, execution_engine, baseline_model])

  print('train_loader has %d samples' % len(train_loader.dataset))
  print('val_loader has %d samples' % len(val_loader.dataset))

  while t < args.num_iterations:
    epoch += 1
    print('Starting epoch %d' % epoch)
    for batch in train_loader:
      t += 1
      questions, _, feats, answers, programs, _ = batch
      questions_var = Variable(questions.cuda())
      feats_var = Variable(feats.cuda())
      answers_var = Variable(answers.cuda())
      if programs[0] is not None:
        programs_var = Variable(programs.cuda())

      reward = None
      if args.model_type == 'PG':
        # Train program generator with ground-truth programs
        pg_optimizer.zero_grad()
        loss = program_generator(questions_var, programs_var)
        loss.backward()
        pg_optimizer.step()
      elif args.model_type == 'EE':
        # Train execution engine with ground-truth programs
        ee_optimizer.zero_grad()
        scores = execution_engine(feats_var, programs_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        ee_optimizer.step()
      elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
        baseline_optimizer.zero_grad()
        baseline_model.zero_grad()
        scores = baseline_model(questions_var, feats_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        baseline_optimizer.step()
      elif args.model_type == 'PG+EE':
        programs_pred = program_generator.reinforce_sample(questions_var)
        scores = execution_engine(feats_var, programs_pred)

        loss = loss_fn(scores, answers_var)
        _, preds = scores.data.cpu().max(1)
        raw_reward = (preds == answers).float()
        reward_moving_average *= args.reward_decay
        reward_moving_average += (1.0 - args.reward_decay) * raw_reward.mean()
        centered_reward = raw_reward - reward_moving_average

        if args.train_execution_engine == 1:
          ee_optimizer.zero_grad()
          loss.backward()
          ee_optimizer.step()

        if args.train_program_generator == 1:
          pg_optimizer.zero_grad()
          program_generator.reinforce_backward(centered_reward.cuda())
          pg_optimizer.step()

      if t % args.record_loss_every == 0:
        print(t, loss.data[0])
        stats['train_losses'].append(loss.data[0])
        stats['train_losses_ts'].append(t)
        if reward is not None:
          stats['train_rewards'].append(reward)

      if t % args.checkpoint_every == 0:
        print('Checking training accuracy ... ')
        train_acc = check_accuracy(args, program_generator, execution_engine,
                                   baseline_model, train_loader)
        print('train accuracy is', train_acc)
        print('Checking validation accuracy ...')
        val_acc = check_accuracy(args, program_generator, execution_engine,
                                 baseline_model, val_loader)
        print('val accuracy is ', val_acc)
        stats['train_accs'].append(train_acc)
        stats['val_accs'].append(val_acc)
        stats['val_accs_ts'].append(t)

        if val_acc > stats['best_val_acc']:
          stats['best_val_acc'] = val_acc
          stats['model_t'] = t
          best_pg_state = get_state(program_generator)
          best_ee_state = get_state(execution_engine)
          best_baseline_state = get_state(baseline_model)

        checkpoint = {
          'args': args.__dict__,
          'program_generator_kwargs': pg_kwargs,
          'program_generator_state': best_pg_state,
          'execution_engine_kwargs': ee_kwargs,
          'execution_engine_state': best_ee_state,
          'baseline_kwargs': baseline_kwargs,
          'baseline_state': best_baseline_state,
          'baseline_type': baseline_type,
          'vocab': vocab
        }
        for k, v in stats.items():
          checkpoint[k] = v
        print('Saving checkpoint to %s' % args.checkpoint_path)
        torch.save(checkpoint, args.checkpoint_path)
        del checkpoint['program_generator_state']
        del checkpoint['execution_engine_state']
        del checkpoint['baseline_state']
        with open(args.checkpoint_path + '.json', 'w') as f:
          json.dump(checkpoint, f)

      if t == args.num_iterations:
        break


def parse_int_list(s):
  return tuple(int(n) for n in s.split(','))


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def get_program_generator(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    pg, kwargs = utils.load_program_generator(args.program_generator_start_from)
    cur_vocab_size = pg.encoder_embed.weight.size(0)
    if cur_vocab_size != len(vocab['question_token_to_idx']):
      print('Expanding vocabulary of program generator')
      pg.expand_encoder_vocab(vocab['question_token_to_idx'])
      kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
  else:
    kwargs = {
      'encoder_vocab_size': len(vocab['question_token_to_idx']),
      'decoder_vocab_size': len(vocab['program_token_to_idx']),
      'wordvec_dim': args.rnn_wordvec_dim,
      'hidden_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
    }
    pg = Seq2Seq(**kwargs)
  pg.cuda()
  pg.train()
  return pg, kwargs


def get_execution_engine(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.execution_engine_start_from is not None:
    ee, kwargs = utils.load_execution_engine(args.execution_engine_start_from)
    # TODO: Adjust vocab?
  else:
    kwargs = {
      'vocab': vocab,
      'feature_dim': parse_int_list(args.feature_dim),
      'stem_batchnorm': args.module_stem_batchnorm == 1,
      'stem_num_layers': args.module_stem_num_layers,
      'module_dim': args.module_dim,
      'module_residual': args.module_residual == 1,
      'module_batchnorm': args.module_batchnorm == 1,
      'classifier_proj_dim': args.classifier_proj_dim,
      'classifier_downsample': args.classifier_downsample,
      'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
      'classifier_batchnorm': args.classifier_batchnorm == 1,
      'classifier_dropout': args.classifier_dropout,
    }
    ee = ModuleNet(**kwargs)
  ee.cuda()
  ee.train()
  return ee, kwargs


def get_baseline_model(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.baseline_start_from is not None:
    model, kwargs = utils.load_baseline(args.baseline_start_from)
  elif args.model_type == 'LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = LstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'cnn_num_res_blocks': args.cnn_num_res_blocks,
      'cnn_res_block_dim': args.cnn_res_block_dim,
      'cnn_proj_dim': args.cnn_proj_dim,
      'cnn_pooling': args.cnn_pooling,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM+SA':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'stacked_attn_dim': args.stacked_attn_dim,
      'num_stacked_attn': args.num_stacked_attn,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmSaModel(**kwargs)
  if model.rnn.token_to_idx != vocab['question_token_to_idx']:
    # Make sure new vocab is superset of old
    for k, v in model.rnn.token_to_idx.items():
      assert k in vocab['question_token_to_idx']
      assert vocab['question_token_to_idx'][k] == v
    for token, idx in vocab['question_token_to_idx'].items():
      model.rnn.token_to_idx[token] = idx
    kwargs['vocab'] = vocab
    model.rnn.expand_vocab(vocab['question_token_to_idx'])
  model.cuda()
  model.train()
  return model, kwargs


def set_mode(mode, models):
  assert mode in ['train', 'eval']
  for m in models:
    if m is None: continue
    if mode == 'train': m.train()
    if mode == 'eval': m.eval()


def check_accuracy(args, program_generator, execution_engine, baseline_model, loader):
  set_mode('eval', [program_generator, execution_engine, baseline_model])
  num_correct, num_samples = 0, 0
  for batch in loader:
    questions, _, feats, answers, programs, _ = batch

    questions_var = Variable(questions.cuda(), volatile=True)
    feats_var = Variable(feats.cuda(), volatile=True)
    answers_var = Variable(feats.cuda(), volatile=True)
    if programs[0] is not None:
      programs_var = Variable(programs.cuda(), volatile=True)

    scores = None # Use this for everything but PG
    if args.model_type == 'PG':
      vocab = utils.load_vocab(args.vocab_json)
      for i in range(questions.size(0)):
        program_pred = program_generator.sample(Variable(questions[i:i+1].cuda(), volatile=True))
        program_pred_str = iep.preprocess.decode(program_pred, vocab['program_idx_to_token'])
        program_str = iep.preprocess.decode(programs[i], vocab['program_idx_to_token'])
        if program_pred_str == program_str:
          num_correct += 1
        num_samples += 1
    elif args.model_type == 'EE':
        scores = execution_engine(feats_var, programs_var)
    elif args.model_type == 'PG+EE':
      programs_pred = program_generator.reinforce_sample(
                          questions_var, argmax=True)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
      scores = baseline_model(questions_var, feats_var)

    if scores is not None:
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == answers).sum()
      num_samples += preds.size(0)

    if num_samples >= args.num_val_samples:
      break

  set_mode('train', [program_generator, execution_engine, baseline_model])
  acc = float(num_correct) / num_samples
  return acc


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
