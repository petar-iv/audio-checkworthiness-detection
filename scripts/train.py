import os
import json
import random
import argparse
import operator
from typing import Any
from pathlib import Path
from shutil import rmtree
from operator import itemgetter

import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, \
  AutoFeatureExtractor, AutoModelForAudioClassification, \
  get_linear_schedule_with_warmup

from evaluate import evaluate
from datasets import TextDataset, AudioDataset
from utils import measure_time, reduced_dataset_dir


NON_INPUT_FIELDS = ['file_identifier', 'line_number', 'is_claim']
class CustomJSONEncoder(json.JSONEncoder):
  def default(self, obj: Any) -> Any:
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    elif isinstance(obj, Path):
      return str(obj)
    else:
      return super(CustomJSONEncoder, self).default(obj)

def train():
  args = process_command_line_args()
  model, datasets, optimizer_artefacts = prepare(args)
  model.to(get_device())

  epoch_to_dev_map_score, epoch_to_test_map_score, losses = train_all_epochs(args, model, datasets, optimizer_artefacts)
  save_map_scores(epoch_to_dev_map_score, args.output_dir / 'dev-maps')
  save_map_scores(epoch_to_test_map_score, args.output_dir / 'test-maps')
  save_losses(losses, args.output_dir)

def process_command_line_args():
  parser = argparse.ArgumentParser()

  # Model
  parser.add_argument('--model_type', type=str, choices=['text', 'audio'])
  parser.add_argument('--model_name', type=str, required=True)

  # Output directory
  parser.add_argument('--output_dir', type=Path, required=True)
  parser.add_argument('--overwrite_output_dir', action='store_true')

  # Data
  parser.add_argument('--data_dir', type=Path, required=True)
  parser.add_argument('--sentence_level_alignment_dir_name', type=Path, required=True)
  parser.add_argument('--audio_segments_dir_name', type=Path)
  parser.add_argument('--max_seq_length', type=int, required=True)

  # Training
  parser.add_argument('--num_train_epochs', type=int, default=10)
  parser.add_argument('--learning_rate', type=float, required=True)
  parser.add_argument('--warmup_proportion', type=float, default=0.1) # learning rate scheduling
  parser.add_argument('--train_batch_size', type=int, default=15)
  parser.add_argument('--eval_batch_size', type=int, default=15)
  parser.add_argument('--mixed_precision', action='store_true') # fp16 support
  parser.add_argument('--max_grad_norm', type=float, default=1.0)
  # Optimizer
  parser.add_argument('--optimizer_betas', type=float, default=[0.9, 0.999], nargs='+')
  parser.add_argument('--epsilon', type=float, default=1e-8)
  parser.add_argument('--weight_decay', type=float, default=0.01)

  # Misc
  parser.add_argument('--seed', type=int, default=42)

  return parser.parse_args()

def prepare(args):
  print_device_information()
  prepare_output_dir(args)
  create_args_file(args)
  set_seed(args.seed)

  model_artefacts = create_model_artefacts(args)
  datasets = create_datasets(args, model_artefacts)
  optimizer_artefacts = create_optimizer_artefacts(args, model_artefacts['model'], datasets['train'])

  return model_artefacts['model'], datasets, optimizer_artefacts

def print_device_information():
  if torch.cuda.is_available():
    print('CUDA is available')
    print(f'Device count: {torch.cuda.device_count()}')
    device = torch.cuda.current_device()
    print(f'Device name: {torch.cuda.get_device_name(device)}')
    print(f'Device properties: {torch.cuda.get_device_properties(device)}')
  else:
    print('CPU is available')
    print(f'Number of threads: {torch.get_num_threads()}')

def prepare_output_dir(args):
  if args.output_dir.exists() and os.listdir(args.output_dir):
    if not args.overwrite_output_dir:
      raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overwrite it.')
    else:
      print(f'Overwriting the output directory {args.output_dir}')
      rmtree(args.output_dir)

  args.output_dir.mkdir(parents=True, exist_ok=True)

def create_args_file(args):
  with (args.output_dir / 'args.json').open('w') as f:
    json.dump(vars(args), f, cls=CustomJSONEncoder, indent=2)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def create_model_artefacts(args):
  config = AutoConfig.from_pretrained(args.model_name, num_labels=2)

  if args.model_type == 'text':
    return {
      'tokenizer': AutoTokenizer.from_pretrained(args.model_name, config=config),
      'model': AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    }

  # audio
  return {
    'feature_extractor': AutoFeatureExtractor.from_pretrained(args.model_name, config=config),
    'model': AutoModelForAudioClassification.from_pretrained(args.model_name, config=config)
  }

def create_datasets(args, model_artefacts):
  root_dir = reduced_dataset_dir(args.data_dir, 'v1') / args.sentence_level_alignment_dir_name
  if args.model_type == 'text':
    return {
      'train': TextDataset(root_dir / 'train', model_artefacts['tokenizer'], args.max_seq_length),
      'dev': TextDataset(root_dir / 'dev', model_artefacts['tokenizer'], args.max_seq_length),
      'test': TextDataset(root_dir / 'test', model_artefacts['tokenizer'], args.max_seq_length)
    }


  if not args.audio_segments_dir_name:
    raise Exception(f'--audio_segments_dir_name must be provided when training on audio')

  root_audio_segments_dir = reduced_dataset_dir(args.data_dir, 'v1') / args.audio_segments_dir_name

  # audio
  return {
    'train': AudioDataset(root_dir / 'train', root_audio_segments_dir / 'train', model_artefacts['feature_extractor'], args.max_seq_length),
    'dev': AudioDataset(root_dir / 'dev', root_audio_segments_dir / 'dev', model_artefacts['feature_extractor'], args.max_seq_length),
    'test': AudioDataset(root_dir / 'test', root_audio_segments_dir / 'test', model_artefacts['feature_extractor'], args.max_seq_length)
  }

def create_optimizer_artefacts(args, model, train_dataset):
  mixed_precision_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
  optimizer = torch.optim.AdamW(model.parameters(),
    lr=args.learning_rate,
    betas=(args.optimizer_betas[0], args.optimizer_betas[1]),
    eps=args.epsilon,
    weight_decay=args.weight_decay
  )
  num_training_steps = int(np.ceil(len(train_dataset) / args.train_batch_size)) * args.num_train_epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=round(args.warmup_proportion * num_training_steps), num_training_steps=num_training_steps)
  return {
    'mixed_precision_scaler': mixed_precision_scaler,
    'optimizer': optimizer,
    'scheduler': scheduler
  }

def train_all_epochs(args, model, datasets, optimizer_artefacts):
  epoch_to_dev_map_score = {}
  epoch_to_test_map_score = {}
  all_losses = []

  for epoch in range(args.num_train_epochs):
    current_epoch = epoch + 1
    measure_time(f'training epoch {current_epoch}', lambda: train_epoch(args, model, datasets['train'], optimizer_artefacts, current_epoch, all_losses))

    print(f'Evaluating model epoch {current_epoch} over dev')
    dev_evaluation, dev_predictions = evaluate_model(args, model, datasets['dev'])
    save_evaluation(args.output_dir / f'epoch-{current_epoch}', 'dev', dev_evaluation, dev_predictions)
    epoch_to_dev_map_score[current_epoch] = dev_evaluation['mean_avg_precision']

    print(f'Evaluating model epoch {current_epoch} over test')
    test_evaluation, test_predictions = evaluate_model(args, model, datasets['test'])
    save_evaluation(args.output_dir / f'epoch-{current_epoch}', 'test', test_evaluation, test_predictions)
    epoch_to_test_map_score[current_epoch] = test_evaluation['mean_avg_precision']

  return epoch_to_dev_map_score, epoch_to_test_map_score, all_losses

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

  model.save_pretrained(epoch_dir)
  save_losses(losses, epoch_dir)
  all_losses.extend(losses)

def move_tensors_to_device(dictionary):
 for k, v in dictionary.items():
    if torch.is_tensor(v):
      dictionary[k] = v.to(get_device())
    elif type(v) == dict:
      move_tensors_to_device(v)

def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def save_losses(losses, target_path):
  np.save(target_path / 'losses.npy', np.array(losses))

def format_number(num):
  return '{:.4f}'.format(num)


def evaluate_model(args, model, dataset):
  def is_input_field(name):
    return name not in NON_INPUT_FIELDS

  model.eval()
  data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True, drop_last=False)

  unique_file_identifiers = data_loader.dataset.df['file_identifier'].unique().tolist()
  unique_file_identifiers.sort()
  actual = { file_identifier: {} for file_identifier in unique_file_identifiers }
  predicted = { file_identifier: {} for file_identifier in unique_file_identifiers }

  with torch.no_grad():
    for batch in tqdm(data_loader):
      move_tensors_to_device(batch)
      model_input = {k: v for k, v in batch.items() if is_input_field(k)}
      batch_predicted = model(**model_input).logits
      batch_predicted = nn.Softmax(dim=1)(batch_predicted)
      batch_predicted = batch_predicted[:, 1]

      for file_id, line_number, actual_value, predicted_score in zip(batch['file_identifier'], \
                                                                      batch['line_number'].tolist(), \
                                                                      batch['is_claim'].tolist(), \
                                                                      batch_predicted.tolist()):

        actual[file_id][line_number] = actual_value
        predicted[file_id][line_number] = predicted_score

  actual_list = []
  predicted_list = []
  for file_identifier in unique_file_identifiers:
    actual_list.append(actual[file_identifier])
    predicted_list.append(predicted[file_identifier])

  return evaluate(actual_list, predicted_list), predicted

def save_evaluation(dir_path, subset, evaluation, predictions):
  with open(dir_path / f'{subset}-evaluation.txt', 'w') as f:
    f.write(f'Mean Average Precision: {format_number(evaluation["mean_avg_precision"])}\n')
    f.write(f'Mean Reciprocal Rank: {format_number(evaluation["mean_reciprocal_rank"])}\n')
    f.write(f'Mean R-Precision: {format_number(evaluation["mean_r_precision"])}\n')
    f.write(f'Overall Precisions: { [format_number(num) for num in evaluation["overall_precisions"]]}\n')


  predictions_dir = dir_path / f'{subset}-predictions'
  predictions_dir.mkdir(parents=True, exist_ok=True)

  for file_identifier, predictions in predictions.items():
    sorted_by_line_number = sorted(predictions.items(), key=operator.itemgetter(0))
    with open(predictions_dir / f'{file_identifier}.txt', 'w') as f:
      for (line_number, score) in sorted_by_line_number:
        f.write(f'{line_number} {score}\n')

def save_map_scores(epoch_to_map_score, file_path_no_extension):
  tuples_sorted_by_epoch_asc = sorted(epoch_to_map_score.items(), key=operator.itemgetter(0))
  tuples_sorted_by_epoch_asc = [(epoch, format_number(score)) for epoch, score in tuples_sorted_by_epoch_asc]

  tuples_sorted_by_map_desc = sorted(epoch_to_map_score.items(), key=operator.itemgetter(1), reverse=True)
  tuples_sorted_by_map_desc = [(epoch, format_number(score)) for epoch, score in tuples_sorted_by_map_desc]

  with open(f'{file_path_no_extension}.txt', 'w') as f:
    f.write(str(tuples_sorted_by_epoch_asc))
    f.write('\n\n')
    f.write(str(tuples_sorted_by_map_desc))
    f.write('\n\n')

if __name__ == '__main__':
  measure_time('model training', train)
