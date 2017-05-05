# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_humans/CLEVR_humans_train.json \
  --input_vocab_json data/input_vocab.json \
  --output_h5_file data/train_human_questions.h5 \
  --output_vocab_json data/human_vocab.json \
  --expand_vocab 1 \
  --unk_threshold 10 \
  --encode_unk 1 \

python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_humans/CLEVR_humans_val.json \
  --input_vocab_json data/human_vocab.json \
  --output_h5_file data/val_human_questions.h5 \
  --encode_unk 1

python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_humans/CLEVR_humans_test.json \
  --input_vocab_json data/human_vocab.json \
  --output_h5_file data/test_human_questions.h5 \
  --encode_unk 1
