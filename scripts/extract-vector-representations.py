import re
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig, \
  AutoTokenizer, AutoModelForSequenceClassification, \
  AutoFeatureExtractor, AutoModelForAudioClassification

import torch.nn as nn
from tqdm.auto import tqdm
from datasets import TextDataset, AudioDataset

from utils import measure_time, set_seed, create_args_file, build_dataframe_for_reduced_dataset, get_device, \
  move_tensors_to_device, sentence_audio_base_name

NON_INPUT_FIELDS = ['file_identifier', 'line_number', 'is_claim']

##
## NOTE: this script is coupled with BERT for text and
## HuBERT and data2vec for audio.
##

def extract_vector_representations():
  args = process_command_line_args()
  set_seed(args.seed)

  args.output_dir.mkdir(parents=True, exist_ok=False)
  create_args_file(args)

  datasets = create_datasets(args)
  model = create_model(args)

  if args.data_type == 'text':
    extract_vectors_from_text(args, datasets, 'train', NON_INPUT_FIELDS, model)
    extract_vectors_from_text(args, datasets, 'dev', NON_INPUT_FIELDS, model)
    extract_vectors_from_text(args, datasets, 'test', NON_INPUT_FIELDS, model)
    return

  if args.data_type == 'audio':
    extract_vectors_from_audio(args, datasets, 'train', NON_INPUT_FIELDS, model)
    extract_vectors_from_audio(args, datasets, 'dev', NON_INPUT_FIELDS, model)
    extract_vectors_from_audio(args, datasets, 'test', NON_INPUT_FIELDS, model)
    return

  raise Exception(f'Could not handle data_type {args.data_type}')

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_data_path', type=Path, required=True)
  parser.add_argument('--dev_data_path', type=Path, required=True)
  parser.add_argument('--test_data_path', type=Path, required=True)

  parser.add_argument('--data_type', type=str, choices=['text', 'audio'], required=True)
  parser.add_argument('--df_column', type=str, required=True)

  parser.add_argument('--model_path', type=Path, required=True)
  parser.add_argument('--model_type', type=str, choices=['custom-aligned-audio', 'huggingface'], required=True)
  parser.add_argument('--audio_model_type', type=str, required=False, choices=['hubert', 'data2vec'])
  parser.add_argument('--tokenizer_name', type=str, required=True)
  parser.add_argument('--max_seq_length', type=int, required=True)

  parser.add_argument('--output_dir', type=Path, required=True)
  parser.add_argument('--seed', type=int, default=42)
  return parser.parse_args()

def create_datasets(args):
  train_df = build_dataframe_for_reduced_dataset(args.train_data_path)
  dev_df = build_dataframe_for_reduced_dataset(args.dev_data_path)
  test_df = build_dataframe_for_reduced_dataset(args.test_data_path)

  config = AutoConfig.from_pretrained(args.tokenizer_name, num_labels=2)

  if args.data_type == 'text':
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, config=config)

    return {
      'train': TextDataset(train_df, tokenizer, args.max_seq_length),
      'dev': TextDataset(dev_df, tokenizer, args.max_seq_length),
      'test': TextDataset(test_df, tokenizer, args.max_seq_length)
    }

  if args.data_type == 'audio':
    # See https://huggingface.co/docs/transformers/master/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.return_attention_mask
    should_return_attention = False if hasattr(config, 'feat_extract_norm') and config.feat_extract_norm == 'group' else True

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.tokenizer_name,
      sampling_rate=16000, do_normalize=True,
      return_attention_mask=should_return_attention, config=config)

    return {
      'train': AudioDataset(train_df, args.df_column, feature_extractor, args.max_seq_length),
      'dev': AudioDataset(dev_df, args.df_column, feature_extractor, args.max_seq_length),
      'test': AudioDataset(test_df, args.df_column, feature_extractor, args.max_seq_length)
    }

  raise Exception(f'Could not handle data_type {args.data_type} while creating datasets')

def create_model(args):
  config = AutoConfig.from_pretrained(args.tokenizer_name, num_labels=2)

  if args.data_type == 'text':
    if args.model_type != 'huggingface':
      raise Exception(f'model_type "{args.model_type}" is not suported for text')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config)
    model = model.to(get_device())
    model.eval()
    return model

  if args.data_type == 'audio':
    if args.model_type == 'huggingface':
      model = AutoModelForAudioClassification.from_pretrained(args.model_path, config=config)
      model = model.to(get_device())
    elif args.model_type == 'custom-aligned-audio':
      model = torch.load(args.model_path / 'model.pt', map_location=get_device())
    else:
      raise Exception(f'model_type "{args.model_type}" is not suported for audio')

    model.eval()
    return model

  raise Exception(f'Could not handle data_type {args.data_type} while creating model')

def extract_vectors_from_text(args, datasets, subset, non_input_fields, model):
  def is_input_field(name):
    return name not in non_input_fields

  data_loader = DataLoader(datasets[subset], batch_size=1, shuffle=True, drop_last=False)
  with torch.no_grad():
    for batch in tqdm(data_loader):
      move_tensors_to_device(batch)

      event_name, file_name = split_file_identifier(batch['file_identifier'][0])
      line_number = batch['line_number'][0].detach().cpu().item()

      model_input = {k: v for k, v in batch.items() if is_input_field(k)}

      # Vector representation
      # See https://github.com/huggingface/transformers/blob/ee0d001de71f0da892f86caa3cf2387020ec9696/src/transformers/models/bert/modeling_bert.py#L1556-L1571
      outputs = model.bert(**model_input)
      vector = outputs[1][0].detach().cpu().numpy()

      target_vectors_dir = args.output_dir / 'vector' / subset / event_name
      target_vectors_dir.mkdir(parents=True, exist_ok=True)

      np.save(target_vectors_dir / f'{sentence_audio_base_name(file_name, line_number)}.npy', vector)

      # Classification
      classification_result = model(**model_input).logits
      prediction = nn.Softmax(dim=1)(classification_result)
      prediction = prediction[:, 1]
      prediction = prediction.detach().cpu().numpy()

      target_classification_dir = args.output_dir / 'classification' / subset / event_name
      target_classification_dir.mkdir(parents=True, exist_ok=True)

      np.save(target_classification_dir / f'{sentence_audio_base_name(file_name, line_number)}.npy', prediction)


def extract_vectors_from_audio(args, datasets, subset, non_input_fields, model):
  if not args.audio_model_type:
    raise Exception('audio_model_type not provided')

  def is_input_field(name):
    return name not in non_input_fields

  data_loader = DataLoader(datasets[subset], batch_size=1, shuffle=True, drop_last=False)
  with torch.no_grad():
    for batch in tqdm(data_loader):
      move_tensors_to_device(batch)

      event_name, file_name = split_file_identifier(batch['file_identifier'][0])
      line_number = batch['line_number'][0].detach().cpu().item()

      model_input = {k: v for k, v in batch.items() if is_input_field(k)}

      # Vector representation
      if args.model_type == 'huggingface':
        vector, logits = extract_vectors_from_audio_huggingface(args, model, model_input)
      elif args.model_type == 'custom-aligned-audio':
        vector, logits = extract_vectors_from_audio_custom_alignment(args, model, model_input)
      else:
        raise Exception(f'Cannot handle model_type "${args.model_type}"')
      vector = vector.detach().cpu().numpy()

      target_vectors_dir = args.output_dir / 'vector' / subset / event_name
      target_vectors_dir.mkdir(parents=True, exist_ok=True)
      np.save(target_vectors_dir / f'{sentence_audio_base_name(file_name, line_number)}.npy', vector)

      # Classification
      prediction = nn.Softmax(dim=1)(logits)
      prediction = prediction[:, 1]
      prediction = prediction.detach().cpu().numpy()

      target_classification_dir = args.output_dir / 'classification' / subset / event_name
      target_classification_dir.mkdir(parents=True, exist_ok=True)

      np.save(target_classification_dir / f'{sentence_audio_base_name(file_name, line_number)}.npy', prediction)

def extract_vectors_from_audio_huggingface(args, model, model_input):
  if args.audio_model_type == 'hubert':
    # See https://github.com/huggingface/transformers/blob/2d956958252617a178a68a06582c99b133fe7d3d/src/transformers/models/hubert/modeling_hubert.py#L1303-L1327
    underlying_model = model.hubert
  elif args.audio_model_type == 'data2vec':
    # See https://github.com/huggingface/transformers/blob/2d956958252617a178a68a06582c99b133fe7d3d/src/transformers/models/data2vec/modeling_data2vec_audio.py#L1183-L1207
    underlying_model = model.data2vec_audio
  else:
    raise Exception(f'Cannot handle huggingface model with audio_model_type "${args.audio_model_type}"')

  outputs = underlying_model(**model_input)
  last_hidden_states = outputs[0]
  last_hidden_states = model.projector(last_hidden_states)
  vector = last_hidden_states.mean(dim=1)[0]

  logits = model(**model_input).logits

  return vector, logits

def extract_vectors_from_audio_custom_alignment(args, model, model_input):
  vector, logits = model.represent_and_classify(audio_data=model_input)
  return vector[0], logits

def split_file_identifier(file_identifier):
  return re.search('(.+)-(audio-\d+)', file_identifier).groups()

if __name__ == '__main__':
  measure_time('extracting vector representations', extract_vector_representations)
