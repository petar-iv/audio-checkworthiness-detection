import os
import json
import time
import random
import operator
import datetime as dt
from os import listdir
from typing import Any
from pathlib import Path
from shutil import rmtree
from os.path import abspath, splitext, basename

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from evaluate import evaluate


def full_dataset_dir(data_dir, subset=None):
  path = Path(data_dir) / 'clef2021' / 'task-1b-english' / 'full' / 'v1'
  return path if not subset else path / subset

def reduced_dataset_dir(data_dir):
  return Path(data_dir) / 'clef2021' / 'task-1b-english' / 'reduced' / 'v1'

def list_dir(dir, fn_filter):
  content = listdir(dir)
  content.sort()
  return [item for item in content if fn_filter(item)]

def iterate_dir(dir, fn_action, fn_filter):
  for item in list_dir(dir, fn_filter):
    fn_action(dir, item, abspath(Path(dir) / item))

def iterate_full_event_files(events_dir, fn_file_action):
  iterate_dir(events_dir, fn_file_action, fn_filter=filter_tsv)

def filter_tsv(file_name):
  return file_name.endswith('.tsv')

def filter_wav(file_name):
  return file_name.endswith('.wav')

def filter_event_name(file_name):
  return file_name.startswith('20') and '_' in file_name

def measure_time(description, fn):
  print(f'> {description} started at {dt.datetime.utcnow()}')
  start = time.time()
  fn()
  end = time.time()
  print(f'> ({dt.timedelta(seconds=end-start)}) {description} completed at {dt.datetime.utcnow()}')

def filename_without_extension(file_path):
  return splitext(basename(file_path))[0]

def iterate_reduced_event_files(events_dir, fn_file_action, fn_file_filter, should_measure_time=False):
  def process_single_event(event_name):
    iterate_dir(events_dir / event_name,
      lambda event_dir, file_name, file_path: fn_file_action(event_name, event_dir, file_name, file_path),
      fn_filter=fn_file_filter)

  for event_name in list_dir(events_dir, fn_filter=filter_event_name):
    if should_measure_time:
      measure_time(f'processing event {event_name}', lambda: process_single_event(event_name))
    else:
      process_single_event(event_name)

def adjust_segment(segment):
  segment = segment.set_frame_rate(16000)
  segment = segment.set_channels(1)
  segment = segment.set_sample_width(2)
  return segment

def sentence_audio_base_name(audio_base_name, line_number):
  return f'{audio_base_name}-{str(line_number).zfill(4)}'

def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def move_tensors_to_device(dictionary):
 for k, v in dictionary.items():
    if torch.is_tensor(v):
      dictionary[k] = v.to(get_device())
    elif type(v) == dict:
      move_tensors_to_device(v)

def load_vector(file_path):
  return np.load(file_path).tolist()

def build_dataframe_for_full_dataset(dataset_path):
    dataframes = []
    def add_event_df(subset_dir, file_name, file_path):
      df = pd.read_csv(file_path, sep='\t', header=None, names=['line_number', 'speaker', 'sentence', 'is_claim'])
      df['file_identifier'] = filename_without_extension(file_name)
      dataframes.append(df)
    iterate_full_event_files(dataset_path, add_event_df)

    return pd.concat(dataframes).reset_index()

def build_dataframe_for_reduced_dataset(dataset_path):
  all_dataframes = []
  def collect_data(event_name, event_dir, file_name, file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['file_identifier'] = df.apply(lambda row: f'{event_name}-{filename_without_extension(file_name)}', axis=1)
    all_dataframes.append(df)

  iterate_reduced_event_files(Path(dataset_path), collect_data, fn_file_filter=filter_tsv)

  return pd.concat(all_dataframes).reset_index()

def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def format_number(num):
  return '{:.4f}'.format(num)

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

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_info(params_count, epoch_to_map_score, best_epoch, test_map_score, file_path):
  tuples_sorted_by_epoch_asc = sorted(epoch_to_map_score.items(), key=operator.itemgetter(0))
  tuples_sorted_by_epoch_asc = [(epoch, format_number(score)) for epoch, score in tuples_sorted_by_epoch_asc]

  tuples_sorted_by_map_desc = sorted(epoch_to_map_score.items(), key=operator.itemgetter(1), reverse=True)
  tuples_sorted_by_map_desc = [(epoch, format_number(score)) for epoch, score in tuples_sorted_by_map_desc]

  with open(file_path, 'w') as f:
    f.write(f'Parameters count: {params_count}\n')
    f.write('\n')
    f.write(str(tuples_sorted_by_epoch_asc))
    f.write('\n')
    f.write(str(tuples_sorted_by_map_desc))
    f.write('\n\n')
    f.write(f'Best epoch {best_epoch}\n')
    f.write(f'MAP(test): {format_number(test_map_score)}\n')

def save_losses(losses, target_path):
  np.save(target_path / 'losses.npy', np.array(losses))

def move_tensors_to_device(dictionary):
 for k, v in dictionary.items():
    if torch.is_tensor(v):
      dictionary[k] = v.to(get_device())
    elif type(v) == dict:
      move_tensors_to_device(v)

def evaluate_model(args, model, dataset, non_input_fields):
  def is_input_field(name):
    return name not in non_input_fields

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
