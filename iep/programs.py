#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for working with and converting between the various data structures
used to represent programs.
"""


def is_chain(program_list):
  visited = [False for fn in program_list]
  cur_idx = len(program_list) - 1
  while True:
    visited[cur_idx] = True
    inputs = program_list[cur_idx]['inputs']
    if len(inputs) == 0:
      break
    elif len(inputs) == 1:
      cur_idx = inputs[0]
    elif len(inputs) > 1:
      return False
  return all(visited)


def list_to_tree(program_list):
  def build_subtree(cur):
    return {
      'function': cur['function'],
      'value_inputs': [x for x in cur['value_inputs']],
      'inputs': [build_subtree(program_list[i]) for i in cur['inputs']],
    }
  return build_subtree(program_list[-1])


def tree_to_prefix(program_tree):
  output = []
  def helper(cur):
    output.append({
      'function': cur['function'],
      'value_inputs': [x for x in cur['value_inputs']],
    })
    for node in cur['inputs']:
      helper(node)
  helper(program_tree)
  return output


def list_to_prefix(program_list):
  return tree_to_prefix(list_to_tree(program_list))


def tree_to_postfix(program_tree):
  output = []
  def helper(cur):
    for node in cur['inputs']:
      helper(node)
    output.append({
      'function': cur['function'],
      'value_inputs': [x for x in cur['value_inputs']],
    })
  helper(program_tree)
  return output


def tree_to_list(program_tree):
  # First count nodes
  def count_nodes(cur):
    return 1 + sum(count_nodes(x) for x in cur['inputs'])
  num_nodes = count_nodes(program_tree)
  output = [None] * num_nodes
  def helper(cur, idx):
    output[idx] = {
      'function': cur['function'],
      'value_inputs': [x for x in cur['value_inputs']],
      'inputs': [],
    }
    next_idx = idx - 1
    for node in reversed(cur['inputs']):
      output[idx]['inputs'].insert(0, next_idx)
      next_idx = helper(node, next_idx)
    return next_idx
  helper(program_tree, num_nodes - 1)
  return output


def prefix_to_tree(program_prefix):
  program_prefix = [x for x in program_prefix]
  def helper():
    cur = program_prefix.pop(0)
    return {
      'function': cur['function'],
      'value_inputs': [x for x in cur['value_inputs']],
      'inputs': [helper() for _ in range(get_num_inputs(cur))],
    }
  return helper()


def prefix_to_list(program_prefix):
  return tree_to_list(prefix_to_tree(program_prefix))


def list_to_postfix(program_list):
  return tree_to_postfix(list_to_tree(program_list))


def postfix_to_tree(program_postfix):
  program_postfix = [x for x in program_postfix]
  def helper():
    cur = program_postfix.pop()
    return {
      'function': cur['function'],
      'value_inputs': [x for x in cur['value_inputs']],
      'inputs': [helper() for _ in range(get_num_inputs(cur))][::-1],
    }
  return helper()


def postfix_to_list(program_postfix):
  return tree_to_list(postfix_to_tree(program_postfix))


def function_to_str(f):
  value_str = ''
  if f['value_inputs']:
    value_str = '[%s]' % ','.join(f['value_inputs'])
  return '%s%s' % (f['function'], value_str)


def str_to_function(s):
  if '[' not in s:
    return {
      'function': s,
      'value_inputs': [],
    }
  name, value_str = s.replace(']', '').split('[')
  return {
    'function': name,
    'value_inputs': value_str.split(','),
  }


def list_to_str(program_list):
  return ' '.join(function_to_str(f) for f in program_list)


def get_num_inputs(f):
  # This is a litle hacky; it would be better to look up from metadata.json
  if type(f) is str:
    f = str_to_function(f)
  name = f['function']
  if name == 'scene':
    return 0
  if 'equal' in name or name in ['union', 'intersect', 'less_than', 'greater_than']:
    return 2
  return 1
