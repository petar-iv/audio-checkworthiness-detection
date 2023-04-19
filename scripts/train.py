import glob
import shutil
import argparse
import operator
from pathlib import Path
from operator import itemgetter

import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoConfig, \
  AutoTokenizer, AutoModelForSequenceClassification, \
  AutoFeatureExtractor, AutoModelForAudioClassification

from datasets import SequenceDataset, VectorsDataset, TextDataset, AudioDataset
from custom_models import RNNModel, TransformerModel, FeedforwardNetwork
from utils import measure_time, print_device_information, \
  prepare_output_dir, create_args_file, set_seed, \
  build_dataframe_for_reduced_dataset, get_device, \
  save_info, save_losses, move_tensors_to_device, \
  count_parameters, evaluate_model, save_evaluation, \
  build_dataframe_for_full_dataset

NON_INPUT_FIELDS = ['file_identifier', 'line_number', 'is_claim']


def train():
  args = process_command_line_args()
  print_device_information()
  prepare_output_dir(args)
  create_args_file(args)
  set_seed(args.seed)

  datasets = create_datasets(args)
  model = create_model(args, datasets)
  model.to(get_device())
  optimizer_artefacts = create_optimizer_artefacts(args, model, datasets['train'])

  epoch_to_dev_map_score, losses = train_all_epochs(args, model, datasets, optimizer_artefacts)
  best_epoch = get_best_performing_epoch(epoch_to_dev_map_score)
  print(f'Evaluating model epoch {best_epoch} over test')
  test_map_score = evaluate_on_test(args, best_epoch, datasets['test'])

  params_count = count_parameters(model)
  save_info(params_count, epoch_to_dev_map_score, best_epoch, test_map_score, args.output_dir / 'result.txt')
  save_losses(losses, args.output_dir)
  delete_all_epoch_dirs_but(best_epoch, args.output_dir)

def process_command_line_args():
  parser = argparse.ArgumentParser()

  # Output directory
  parser.add_argument('--output_dir', type=Path, required=True)
  parser.add_argument('--overwrite_output_dir', action='store_true')

  # Data
  parser.add_argument('--train_data_path', type=Path, required=True)
  parser.add_argument('--dev_data_path', type=Path, required=True)
  parser.add_argument('--test_data_path', type=Path, required=True)
  parser.add_argument('--dataset_type', type=str, choices=['full', 'reduced'], default='reduced')
  parser.add_argument('--data_type', type=str, required=True, choices=['sequence-features', 'features', 'text', 'audio'])
  parser.add_argument('--df_columns', type=str, nargs='+')
  parser.add_argument('--scalers_dir', type=Path)
  parser.add_argument('--max_sequence_length', type=int)

  # Model
  parser.add_argument('--model_type', type=str, choices=['custom', 'huggingface'], required=True)
  parser.add_argument('--model_name', type=str, required=True) # predefined: rnn, transformer-encoder, feedforward
  parser.add_argument('--dropout', type=float, default=0.2)
  ## RNN
  parser.add_argument('--rnn_num_layers', type=int)
  parser.add_argument('--rnn_num_directions', type=int)
  parser.add_argument('--rnn_hidden_size', type=int)
  ## Transformer Encoder
  parser.add_argument('--transformer_num_heads', type=int)
  parser.add_argument('--transformer_num_layers', type=int)
  parser.add_argument('--transformer_activation', type=str, choices=['relu', 'gelu'], default='gelu')
  ## Feedforward
  parser.add_argument('--ff_hidden_sizes', type=int, nargs='*')

  # Training
  parser.add_argument('--num_train_epochs', type=int, default=15)
  parser.add_argument('--learning_rate', type=float, required=True)
  parser.add_argument('--learning_rate_scheduler', type=str, choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], default='linear')
  parser.add_argument('--warmup_proportion', type=float, default=0.1) # learning rate scheduling
  parser.add_argument('--train_batch_size', type=int, default=15)
  parser.add_argument('--eval_batch_size', type=int, default=15)
  parser.add_argument('--mixed_precision', action='store_true') # fp16 support
  parser.add_argument('--max_grad_norm', type=float, default=1.0)
  ## Optimizer
  parser.add_argument('--optimizer_betas', type=float, default=[0.9, 0.999], nargs='+')
  parser.add_argument('--epsilon', type=float, default=1e-8)
  parser.add_argument('--weight_decay', type=float, default=0.01)

  # Misc
  parser.add_argument('--seed', type=int, default=42)

  return parser.parse_args()

def create_datasets(args):
  train_df, dev_df, test_df = load_dataframes(args)

  if args.data_type == 'sequence-features':
    column_info = {
      'name': args.df_columns[0],
      'scaler': torch.load(args.scalers_dir / f'{args.df_columns[0]}.bin') if args.scalers_dir else None
    }
    should_return_attention = args.model_name == 'transformer-encoder'

    return {
      'train': SequenceDataset(train_df, args.max_sequence_length, column_info, should_return_attention),
      'dev': SequenceDataset(dev_df, args.max_sequence_length, column_info, should_return_attention),
      'test': SequenceDataset(test_df, args.max_sequence_length, column_info, should_return_attention)
    }

  if args.data_type == 'features':
    column_infos = build_columns_info(args.df_columns, args.scalers_dir)

    return {
      'train': VectorsDataset(train_df, column_infos),
      'dev': VectorsDataset(dev_df, column_infos),
      'test': VectorsDataset(test_df, column_infos)
    }

  if args.data_type == 'text':
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, config=config)

    if not args.max_sequence_length:
      raise Exception(f'max_sequence_length must be provided for text classification')

    return {
      'train': TextDataset(train_df, tokenizer, args.max_sequence_length),
      'dev': TextDataset(dev_df, tokenizer, args.max_sequence_length),
      'test': TextDataset(test_df, tokenizer, args.max_sequence_length)
    }

  if args.data_type == 'audio':
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
    # See https://huggingface.co/docs/transformers/master/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.return_attention_mask
    should_return_attention = False if hasattr(config, 'feat_extract_norm') and config.feat_extract_norm == 'group' else True

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name,
            sampling_rate=16000, do_normalize=True,
            return_attention_mask=should_return_attention, config=config)

    return {
      'train': AudioDataset(train_df, args.df_columns[0], feature_extractor, args.max_sequence_length),
      'dev': AudioDataset(dev_df, args.df_columns[0], feature_extractor, args.max_sequence_length),
      'test': AudioDataset(test_df, args.df_columns[0], feature_extractor, args.max_sequence_length)
    }

  raise Exception(f'Could not handle data_type "{args.data_type}" while creating datasets')

def load_dataframes(args):
  if args.dataset_type == 'full':
    train_df = build_dataframe_for_full_dataset(args.train_data_path)
    dev_df = build_dataframe_for_full_dataset(args.dev_data_path)
    test_df = build_dataframe_for_full_dataset(args.test_data_path)
    return train_df, dev_df, test_df

  if args.dataset_type == 'reduced':
    train_df = build_dataframe_for_reduced_dataset(args.train_data_path)
    dev_df = build_dataframe_for_reduced_dataset(args.dev_data_path)
    test_df = build_dataframe_for_reduced_dataset(args.test_data_path)
    return train_df, dev_df, test_df

  raise Exception(f'Could not handle dataset_type "{args.dataset_type}" while loading dataframes')

def build_columns_info(df_columns, scalers_dir):
  if not df_columns:
    raise Exception('df_columns should be provided')

  pairs = []
  for df_column in df_columns:
    pairs.append({
      'name': df_column,
      'scaler': torch.load(scalers_dir / f'{df_column}.bin') if scalers_dir else None
    })
  return pairs

def create_model(args, datasets):
  if args.model_type == 'custom':
    if args.model_name == 'rnn':
      input_size = datasets['train'][0]['sequence'].shape[-1]
      return RNNModel(input_size, args.rnn_hidden_size, args.rnn_num_layers, args.rnn_num_directions, args.dropout)

    if args.model_name == 'transformer-encoder':
      input_size = datasets['train'][0]['sequence'].shape[-1]
      return TransformerModel(input_size, args.transformer_num_heads, args.transformer_activation, args.dropout, args.transformer_num_layers, args.max_sequence_length)

    if args.model_name == 'feedforward':
      input_size = datasets['train'][0]['features'].shape[-1]
      return FeedforwardNetwork(input_size, args.ff_hidden_sizes, args.dropout)

    raise Exception(f'Could not handle model_name "{args.model_name}" while creating custom model')

  if args.model_type == 'huggingface':
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)

    if args.data_type == 'text':
      return AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    if args.data_type == 'audio':
      model = AutoModelForAudioClassification.from_pretrained(args.model_name, config=config)
      model.freeze_feature_extractor()
      return model

    raise Exception(f'Could not handle data_type "{args.data_type}" while creating huggingface model')

  raise Exception(f'Could not handle model_type "{args.model_type}" while creating model')


def create_optimizer_artefacts(args, model, train_dataset):
  mixed_precision_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
  optimizer = torch.optim.AdamW(model.parameters(),
    lr=args.learning_rate,
    betas=(args.optimizer_betas[0], args.optimizer_betas[1]),
    eps=args.epsilon,
    weight_decay=args.weight_decay
  )
  num_training_steps = int(np.ceil(len(train_dataset) / args.train_batch_size)) * args.num_train_epochs
  num_warmup_steps = round(args.warmup_proportion * num_training_steps)
  scheduler = get_scheduler(name=args.learning_rate_scheduler, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
  return {
    'mixed_precision_scaler': mixed_precision_scaler,
    'optimizer': optimizer,
    'scheduler': scheduler
  }

def train_all_epochs(args, model, datasets, optimizer_artefacts):
  epoch_to_dev_map_score = {}
  all_losses = []

  for epoch in range(args.num_train_epochs):
    current_epoch = epoch + 1
    measure_time(f'training epoch {current_epoch}', lambda: train_epoch(args, model, datasets['train'], optimizer_artefacts, current_epoch, all_losses))

    print(f'Evaluating model epoch {current_epoch} over dev')
    dev_evaluation, dev_predictions = evaluate_model(args, model, datasets['dev'], NON_INPUT_FIELDS)
    save_evaluation(args.output_dir / f'epoch-{current_epoch}', 'dev', dev_evaluation, dev_predictions)
    epoch_to_dev_map_score[current_epoch] = dev_evaluation['mean_avg_precision']

  return epoch_to_dev_map_score, all_losses

def train_epoch(args, model, dataset, optimizer_artefacts, current_epoch, all_losses):
  model.train()

  criterion = nn.CrossEntropyLoss()
  mixed_precision_scaler, optimizer, scheduler = itemgetter('mixed_precision_scaler', 'optimizer', 'scheduler')(optimizer_artefacts)

  losses = []

  data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
  for _, batch in enumerate(tqdm(data_loader)):
    model.zero_grad()

    move_tensors_to_device(batch)
    model_input = {k: v for k, v in batch.items() if k != 'file_identifier' and k != 'line_number' and k != 'is_claim'}
    actual = batch['is_claim'].to(get_device()).view(-1)

    with torch.cuda.amp.autocast(enabled=args.mixed_precision):
      predicted_logits = model(**model_input).logits
      # Provide logits to CrossEntropyLoss as it performs LogSoftmax internally
      # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
      # "Note that this case is equivalent to the combination of LogSoftmax and NLLLoss"
      loss = criterion(predicted_logits, actual)

      mixed_precision_scaler.scale(loss).backward()
      mixed_precision_scaler.unscale_(optimizer)
      nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

      mixed_precision_scaler.step(optimizer)
      mixed_precision_scaler.update()
      scheduler.step()

    losses.append(loss.detach().cpu().item())

  epoch_dir = args.output_dir / f'epoch-{current_epoch}'
  epoch_dir.mkdir(parents=True, exist_ok=True)

  save_model(args, model, epoch_dir)
  all_losses.extend(losses)

def save_model(args, model, epoch_dir):
  if args.model_type == 'huggingface':
    return model.save_pretrained(epoch_dir)

  if args.model_type == 'custom':
    return torch.save(model, epoch_dir / 'model.pt')

  raise Exception(f'Could not handle model_type "{args.model_type}" while saving model')

def load_model(args, epoch_dir):
  if args.model_type == 'huggingface':
    if args.data_type == 'text':
      return AutoModelForSequenceClassification.from_pretrained(epoch_dir)

    if args.data_type == 'audio':
      return AutoModelForAudioClassification.from_pretrained(epoch_dir)

    raise Exception(f'Could not handle data_type "{args.data_type}" while loading huggingface model')

  if args.model_type == 'custom':
    return torch.load(epoch_dir / 'model.pt', map_location=get_device())

  raise Exception(f'Could not handle model_type "{args.model_type}" while loading model')

def evaluate_on_test(args, best_epoch, dataset):
  best_epoch_dir = args.output_dir / f'epoch-{best_epoch}'
  best_model = load_model(args, best_epoch_dir)
  best_model.to(get_device())

  test_evaluation, test_predictions = evaluate_model(args, best_model, dataset, NON_INPUT_FIELDS)
  save_evaluation(best_epoch_dir, 'test', test_evaluation, test_predictions)

  return test_evaluation['mean_avg_precision']

def get_best_performing_epoch(epoch_to_map_score):
  sorted_by_map_desc = sorted(epoch_to_map_score.items(), key=operator.itemgetter(1), reverse=True)
  return sorted_by_map_desc[0][0]

def delete_all_epoch_dirs_but(best_epoch, output_dir):
  all_epochs_dir = glob.glob(f'{output_dir}/epoch-*')
  best_epoch = f'epoch-{best_epoch}'

  for epoch_dir in all_epochs_dir:
    if not epoch_dir.endswith(best_epoch):
      shutil.rmtree(epoch_dir)

if __name__ == '__main__':
  measure_time('training', train)
