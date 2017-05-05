# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

if [[ $1 == "0" ]]; then
  export CUDA_VISIBLE_DEVICES=1
  # python scripts/train.py \
  #   --train_question_h5 data/train_human_questions.h5 \
  #   --train_features_h5 data-ssd/train_features.h5 \
  #   --val_question_h5 data/val_human_questions.h5 \
  #   --val_features_h5 data-ssd/val_features.h5 \
  #   --loader_num_workers 1 \
  #   --vocab_json data/human_vocab.json \
  #   --model_type LSTM \
  #   --baseline_start_from data/models/lstm.pt \
  #   --baseline_train_only_rnn 1 \
  #   --learning_rate 5e-4 \
  #   --num_iterations 15000 \
  #   --checkpoint_every 500 \
  #   --checkpoint_path data/new_models_v2/lstm_human.pt

  python scripts/train.py \
    --train_question_h5 data/train_human_questions.h5 \
    --train_features_h5 data-ssd/train_features.h5 \
    --val_question_h5 data/val_human_questions.h5 \
    --val_features_h5 data-ssd/val_features.h5 \
    --loader_num_workers 1 \
    --vocab_json data/human_vocab.json \
    --model_type PG+EE \
    --program_generator_start_from data/models/program_generator_18k.pt \
    --execution_engine_start_from data/models/execution_engine_18k.pt \
    --train_program_generator 1 \
    --train_execution_engine 0 \
    --learning_rate 1e-4 \
    --num_iterations 100000 \
    --checkpoint_every 500 \
    --checkpoint_path data/new_models_v3/ours_human.pt
fi

if [[ $1 == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=0
  # python scripts/train.py \
  #   --train_question_h5 data/train_human_questions.h5 \
  #   --train_features_h5 data-ssd/train_features.h5 \
  #   --val_question_h5 data/val_human_questions.h5 \
  #   --val_features_h5 data-ssd/val_features.h5 \
  #   --vocab_json data/human_vocab.json \
  #   --loader_num_workers 1 \
  #   --model_type CNN+LSTM \
  #   --baseline_start_from data/models/cnn_lstm.pt \
  #   --baseline_train_only_rnn 1 \
  #   --learning_rate 5e-4 \
  #   --num_iterations 15000 \
  #   --checkpoint_every 500 \
  #   --checkpoint_path data/new_models_v2/cnn_lstm_human.pt

  python scripts/train.py \
    --train_question_h5 data/train_human_questions.h5 \
    --train_features_h5 data-ssd/train_features.h5 \
    --val_question_h5 data/val_human_questions.h5 \
    --val_features_h5 data-ssd/val_features.h5 \
    --loader_num_workers 1 \
    --vocab_json data/human_vocab.json \
    --model_type CNN+LSTM+SA \
    --baseline_start_from data/models/cnn_lstm_sa_mlp.pt \
    --baseline_train_only_rnn 1 \
    --learning_rate 1e-4 \
    --num_iterations 100000 \
    --checkpoint_every 500 \
    --checkpoint_path data/new_models_v3/cnn_lstm_sa_mlp_human.pt
fi
