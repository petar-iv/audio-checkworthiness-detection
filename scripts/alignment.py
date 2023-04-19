import os
import argparse
import operator
from pathlib import Path
from shutil import rmtree

import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoFeatureExtractor

from datasets import AudioAndTextRepresentationDataset
from custom_models import AlignedAudioModel
from utils import measure_time, set_seed, get_device, move_tensors_to_device, \
  save_losses, evaluate_model, save_evaluation, build_dataframe_for_reduced_dataset, create_args_file, format_number


NON_INPUT_FIELDS = ['file_identifier', 'line_number', 'is_claim', 'text_representation']

def train_alignment():
  args = process_command_line_args()
  set_seed(args.seed)

  model, datasets = prepare(args)
  model.to(get_device())

  epoch_to_dev_map_score, epoch_to_test_map_score, losses = train(args, model, datasets)
  save_map_scores(epoch_to_dev_map_score, args.output_dir / 'dev-maps')
  save_map_scores(epoch_to_test_map_score, args.output_dir / 'test-maps')
  save_losses(losses, args.output_dir)

  write_text_classifier_params(model._text_model, args.output_dir / 'text-classifier-after.txt')

def process_command_line_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--audio_model', required=True, type=str)
  parser.add_argument('--audio_df_column', required=True, type=str)
  parser.add_argument('--audio_max_seq_length', required=True, type=int)

  parser.add_argument('--text_representation_df_column', required=True, type=str)
  parser.add_argument('--text_model_path', required=True, type=str)

  parser.add_argument('--train_data_path', required=True, type=Path)
  parser.add_argument('--dev_data_path', required=True, type=Path)
  parser.add_argument('--test_data_path', required=True, type=Path)
  parser.add_argument('--output_dir', required=True, type=Path)
  parser.add_argument('--overwrite_output_dir', action='store_true')

  parser.add_argument('--train_batch_size', type=int, default=8)
  parser.add_argument('--eval_batch_size', type=int, default=8)
  parser.add_argument('--mixed_precision', action='store_true')
  parser.add_argument('--learning_rate', type=float, default=5e-5)
  parser.add_argument('--weight_decay', type=float, default=0.01)
  parser.add_argument('--max_grad_norm', type=float, default=1.0)
  parser.add_argument('--num_train_epochs', type=int, default=5)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--alignment_error_weight', required=True, type=float)

  args = parser.parse_args()

  if args.output_dir.exists() and os.listdir(args.output_dir):
    if not args.overwrite_output_dir:
      raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overwrite it.')
    else:
      print(f'Overwriting the output directory {args.output_dir}')
      rmtree(args.output_dir)

  args.output_dir.mkdir(parents=True, exist_ok=True)

  create_args_file(args)

  return args

def prepare(args):
  audio_config = AutoConfig.from_pretrained(args.audio_model, num_labels=2)
  # See https://huggingface.co/docs/transformers/master/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.return_attention_mask
  should_return_attention = False if hasattr(audio_config, 'feat_extract_norm') and audio_config.feat_extract_norm == 'group' else True

  audio_feature_extractor = AutoFeatureExtractor.from_pretrained(args.audio_model,
    sampling_rate=16000,
    do_normalize=True,
    return_attention_mask=should_return_attention,
    config=audio_config)
  audio_model = AutoModel.from_pretrained(args.audio_model, config=audio_config)

  train_df = build_dataframe_for_reduced_dataset(args.train_data_path)
  dev_df = build_dataframe_for_reduced_dataset(args.dev_data_path)
  test_df = build_dataframe_for_reduced_dataset(args.test_data_path)

  datasets = {
    'train': AudioAndTextRepresentationDataset(train_df, args.audio_df_column, audio_feature_extractor, args.audio_max_seq_length, args.text_representation_df_column),
    'dev': AudioAndTextRepresentationDataset(dev_df, args.audio_df_column, audio_feature_extractor, args.audio_max_seq_length, args.text_representation_df_column),
    'test': AudioAndTextRepresentationDataset(test_df, args.audio_df_column, audio_feature_extractor, args.audio_max_seq_length, args.text_representation_df_column)
  }

  text_model = AutoModelForSequenceClassification.from_pretrained(args.text_model_path)
  write_text_classifier_params(text_model, args.output_dir / 'text-classifier-before.txt')

  model = AlignedAudioModel(audio_model, text_model)
  return model, datasets

def write_text_classifier_params(text_model, filename):
  with open(filename, 'w') as f:
    for name, parameter in text_model.classifier.named_parameters():
      if not parameter.requires_grad: continue
      arr = parameter.detach().cpu().numpy()
      for i in arr:
        f.write(str(i))
        f.write('\n')

def train(args, model, datasets):
  mixed_precision_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  return train_all_epochs(args, model, datasets, mixed_precision_scaler, optimizer)

def train_all_epochs(args, model, datasets, mixed_precision_scaler, optimizer):
  epoch_to_dev_map_score = {}
  epoch_to_test_map_score = {}
  all_losses = []

  for epoch in range(args.num_train_epochs):
    current_epoch = epoch + 1
    measure_time(f'training epoch {current_epoch}', lambda: train_epoch(args, model, datasets['train'], mixed_precision_scaler, optimizer, current_epoch, all_losses))

    print(f'Evaluating model epoch {current_epoch} over dev')
    dev_evaluation, dev_predictions = evaluate_model(args, model, datasets['dev'], NON_INPUT_FIELDS)
    save_evaluation(args.output_dir / f'epoch-{current_epoch}', 'dev', dev_evaluation, dev_predictions)
    epoch_to_dev_map_score[current_epoch] = dev_evaluation['mean_avg_precision']

    print(f'Evaluating model epoch {current_epoch} over test')
    test_evaluation, test_predictions = evaluate_model(args, model, datasets['test'], NON_INPUT_FIELDS)
    save_evaluation(args.output_dir / f'epoch-{current_epoch}', 'test', test_evaluation, test_predictions)
    epoch_to_test_map_score[current_epoch] = test_evaluation['mean_avg_precision']

  return epoch_to_dev_map_score, epoch_to_test_map_score, all_losses

def train_epoch(args, model, dataset, mixed_precision_scaler, optimizer, current_epoch, all_losses):
  model.train()
  alignmentCriterion = nn.MSELoss(reduction='none')
  classificationCriterion = nn.CrossEntropyLoss()
  losses = []

  data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
  for _, batch in enumerate(tqdm(data_loader)):
    model.zero_grad()

    move_tensors_to_device(batch)
    model_input = {k: v for k, v in batch.items() if k != 'line_number' and k != 'file_identifier' and k != 'text_representation' and k != 'is_claim'}
    actual_representation = batch['text_representation'].to(get_device())
    actual_class = batch['is_claim'].to(get_device())

    with torch.cuda.amp.autocast(enabled=args.mixed_precision):
      predicted_representation, predicted_class = model.represent_and_classify(**model_input)

      representationLoss = alignmentCriterion(predicted_representation, actual_representation).mean()
      classificationLoss = classificationCriterion(predicted_class, actual_class)
      composite_loss = (args.alignment_error_weight * representationLoss) + ((1 - args.alignment_error_weight) * classificationLoss)

      mixed_precision_scaler.scale(composite_loss).backward()
      mixed_precision_scaler.unscale_(optimizer)
      nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

      mixed_precision_scaler.step(optimizer)
      mixed_precision_scaler.update()

      losses.append(composite_loss.detach().cpu().item())

  epoch_dir = args.output_dir / f'epoch-{current_epoch}'
  epoch_dir.mkdir(parents=True, exist_ok=True)

  torch.save(model, epoch_dir / 'model.pt')
  save_losses(losses, epoch_dir)
  all_losses.extend(losses)

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
  measure_time('training alignment', train_alignment)
