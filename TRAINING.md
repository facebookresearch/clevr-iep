# Training

Here we will walk through the process of training your own model on the CLEVR dataset,
and finetuning the model on the CLEVR-Humans dataset.
All training code runs on GPU, and assumes that CUDA and cuDNN already been installed.

- [Preprocessing CLEVR](#preprocessing-clevr)
- [Training on CLEVR](#training-on-clevr)
- [Training baselines on CLEVR](#training-baselines-on-clevr)
- [Preprocessing CLEVR-Humans](#preprocessing-clevr-humans)
- [Finetuning on CLEVR-Humans](#finetuning-on-clevr-humans)
- [Finetuning baselines on CLEVR-Humans](#finetuning-baselines-on-clevr-humans)

## Preprocessing CLEVR

Before you can train any models, you need to download the
[CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/);
you also need to extract features for the images, and preprocess the questions and programs.

### Step 1: Download the data

First you need to download and unpack the [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/).
For the purpose of this tutorial we assume that all data will be stored in a new directory called `data/`:

```bash
mkdir data
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O data/CLEVR_v1.0.zip
unzip data/CLEVR_v1.0.zip -d data
```

### Step 2: Extract Image Features

Extract ResNet-101 features for the CLEVR train, val, and test images with the following commands:

```bash
python scripts/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/train \
  --output_h5_file data/train_features.h5

python scripts/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/val \
  --output_h5_file data/val_features.h5

python scripts/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/test \
  --output_h5_file data/test_features.h5
```

### Step 3: Preprocess Questions

Preprocess the questions and programs for the CLEVR train, val, and test sets with the following commands:

```bash
python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_train_questions.json \
  --output_h5_file data/train_questions.h5 \
  --output_vocab_json data/vocab.json

python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_val_questions.json \
  --output_h5_file data/val_questions.h5 \
  --input_vocab_json data/vocab.json
  
python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_test_questions.json \
  --output_h5_file data/test_questions.h5 \
  --input_vocab_json data/vocab.json
```

When preprocessing questions, we create a file `vocab.json` which stores the mapping between
tokens and indices for questions and programs. We create this vocabulary when preprocessing
the training questions, then reuse the same vocabulary file for the val and test questions.

## Training on CLEVR

Models are trained through a three-step procedure:

1. Train the program generator using a small number of ground-truth programs
2. Train the execution engine using predicted outputs from the trained program generator
3. Jointly fine-tune both the program generator and the execution engine without any ground-truth programs

### Step 1: Train the Program Generator

In this step we use a small number of ground-truth programs to train the program generator:

```bash
python scripts/train_model.py \
  --model_type PG \
  --num_train_samples 18000 \
  --num_iterations 20000 \
  --checkpoint_every 1000 \
  --checkpoint_path data/program_generator.pt
```

### Step 2: Train the Execution Engine

In this step we train the execution engine, using programs predicted from the program generator
in the previous step:

```bash
python scripts/train_model.py \
  --model_type EE \
  --program_generator_start_from data/program_generator.py \
  --num_iterations 100000 \
  --checkpoint_path data/execution_engine.pt
```

### Step 3: Jointly train entire model

In this step we jointly train the program generator and execution engine using REINFORCE:

```bash
python scripts/train_model.py \
  --model_type PG+EE \
  --program_generator_start_from data/program_generator.pt \
  --execution_engine_start_from data/execution_engine.pt \
  --checkpoint_path data/joint_pg_ee.pt
```

### Step 4: Test the model

You can use the `run_model.py` script to test your model on the entire validation
and test sets. To test the version of the model before finetuning on the val set:

```bash
python scripts/run_model.py \
  --program_generator data/program_generator.pt \
  --execution_engine data/execution_engine.pt \
  --input_question_h5 data/val_questions.h5 \
  --input_features_h5 data/val_features.h5
```

You can test the jointly finetuned model like this:

```bash
python scripts/run_model.py \
  --program_generator data/joint_pg_ee.pt \
  --execution_engine data/joint_pg_ee.pt \
  --input_question_h5 data/val_questions.h5 \
  --input_features_h5 data/val_features.h5
```

## Training baselines on CLEVR

### Step 1: Train the model

You can use the `train_model.py` script to train the LSTM, CNN+LSTM, and CNN+LSTM+SA baselines.

For example you can train CNN+LSTM+SA+MLP like this:

```bash
python scripts/train_model.py \
  --model_type CNN+LSTM+SA \
  --classifier_fc_dims 1024 \
  --num_iterations 400000 \
  --checkpoint_path data/cnn_lstm_sa_mlp.pt
```

### Step 2: Test the model

You can use the `run_model.py` script to test baseline models on the entire validation or test sets.
You can run the model from the previous step on the entire val set like this:

```bash
python scripts/run_model.py \
  --baseline_model data/cnn_lstm_mlp.pt \
  --input_question_h5 data/val_questions.h5 \
  --input_features_h5 data/val_features.h5
```

## Preprocessing CLEVR-Humans

### Step 1: Download the data

You can download the CLEVR-Humans dataset like this:

```bash
wget http://cs.stanford.edu/people/jcjohns/iep/CLEVR-Humans.zip -O data/CLEVR-Humans.zip
unzip data/CLEVR-Humans.zip -d data
```

### Step 2: Preprocess the data

Preprocessing the CLEVR-Humans dataset is a bit tricky, since it contains words that do not appear
in the CLEVR dataset. In addition, unlike CLEVR, we wish to replace infrequent words (which may be
misspellings or typos) with a special `<UNK>` token. Furthermore, in order to use models trained on
CLEVR on CLEVR-Humans, we need to ensure that the vocabulary we compute on CLEVR-Humans is compatible
with that from CLEVR which we preprocessed earlier.

All of these issues are handled by the `preprocess_questions.py` script, but we need to pass a few
extra flags to control this behavior.

```bash
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
```

## Finetuning on CLEVR-Humans

### Step 1: Finetune the model

The CLEVR-Humans dataset does not provide ground-truth programs, but we can use REINFORCE to
jointly train our entire model on this dataset regardless. When finetuning on CLEVR-Humans,
we only update the program generator to prevent overfitting.

You can use the `train_model.py` script for finetuning like this:

```bash
python scripts/train_model.py \
  --model_type PG+EE \
  --train_question_h5 data/train_human_questions.h5 \
  --train_features_h5 data/train_features.h5 \
  --val_question_h5 data/val_human_questions.h5 \
  --val_features_h5 data/val_features.h5 \
  --vocab_json data/human_vocab.json \
  --program_generator_start_from data/joint_pg_ee.pt \
  --execution_engine_start_from data/joint_pg_ee.pt \
  --train_program_generator 1 \
  --train_execution_engine 0 \
  --learning_rate 1e-4 \
  --num_iterations 100000 \
  --checkpoint_every 500 \
  --checkpoint_path data/human_program_generator.pt
```

### Step 2: Test the model

You can use the `run_model.py` script to run the model on the entire CLEVR-Humans
validation or test set. In the previous step we only updated the program generator;
when testing the model we use the execution engine that was trained on CLEVR.

```bash
python scripts/run_model.py \
  --program_generator data/human_program_generator.pt \
  --execution_engine data/joint_pg_ee.pt \
  --input_question_h5 data/val_human_questions.h5 \
  --input_features_h5 data/val_features.h5
```

## Finetuning baselines on CLEVR-Humans

### Step 1: Finetune the model

You can use the `train_model.py` script to finetune the LSTM, CNN+LSTM, and CNN+LSTM+SA
models on the CLEVR-Humans dataset. When finetuning baselines on CLEVR-Humans we only
update the RNN to prevent overfitting. For example you can finetune the CNN+LSTM+SA+MLP
model we trained earlier like this:

```bash
python scripts/train.py \
  --model_type CNN+LSTM+SA \
  --train_question_h5 data/train_human_questions.h5 \
  --train_features_h5 data/train_features.h5 \
  --val_question_h5 data/val_human_questions.h5 \
  --val_features_h5 data-ssd/val_features.h5 \
  --vocab_json data/human_vocab.json \
  --baseline_start_from data/cnn_lstm_sa_mlp.pt \
  --baseline_train_only_rnn 1 \
  --learning_rate 1e-4 \
  --num_iterations 100000 \
  --checkpoint_every 500 \
  --checkpoint_path data/cnn_lstm_sa_mlp_human.pt
```

### Step 2: Test the model

You can use the `run_model.py` script to test the finetuned baseline models on
the entire val or test sets of the CLEVR-Humans dataset. For example you can 
run the finetuned CNN+LSTM+SA+MLP model on the entire validation set like this:

```bash
python scripts/run_model.py \
  --baseline_model data/cnn_lstm_sa_mlp_human.pt \
  --input_question_h5 data/val_human_questions.h5 \
  --input_features_h5 data/val_features.h5
```
