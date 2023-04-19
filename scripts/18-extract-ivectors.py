import re
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from kaldiio import ReadHelper

from utils import measure_time, list_dir, filter_event_name, reduced_dataset_dir, iterate_reduced_event_files, \
  filter_tsv, filename_without_extension, sentence_audio_base_name


kaldi_recipe_dir = Path('repo') / 'egs' / 'librispeech' / 's5'

def extract_ivectors():
  args = process_command_line_args()

  for subset in ['train', 'dev', 'test']:
    create_kaldi_input(args, subset)
    create_mfcc_features(args, subset)
    create_ivectors(args, subset)
    create_ivector_lists(args, subset)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--segments_dir_name', required=True, type=str)
  parser.add_argument('--target_dir_name', required=True, type=str)
  parser.add_argument('--kaldi_dir', required=True, type=Path)
  parser.add_argument('--mfcc_config', required=True, type=str)
  return parser.parse_args()

def create_kaldi_input(args, subset):
  data_dir = args.data_dir

  sentence_level_alignments = reduced_dataset_dir(data_dir) / '07-sentence-level-alignment' / subset
  segments = reduced_dataset_dir(data_dir) / args.segments_dir_name / subset

  event_name_to_inputs = create_inputs_dict(sentence_level_alignments)

  def add_utterances(event_name, event_path, file_name, file_path):
    df = pd.read_csv(file_path, sep='\t')
    for _, row in df.iterrows():
      speaker_id = to_speaker_id(row['speaker'])
      utterance_audio_base_name = sentence_audio_base_name(filename_without_extension(file_name), row['line_number'])
      utterance_id = speaker_id + '_' +utterance_audio_base_name

      event_name_to_inputs[event_name]['utterance_id_to_audio_path'].append({
        'utterance_id': utterance_id,
        'audio_path': str(segments / event_name / f'{utterance_audio_base_name}.wav')
      })

      event_name_to_inputs[event_name]['speaker_id_to_utterance_id'].append({
        'speaker_id': speaker_id,
        'utterance_id': utterance_id
      })

  iterate_reduced_event_files(sentence_level_alignments, add_utterances, fn_file_filter=filter_tsv)
  create_input_files(args, subset, event_name_to_inputs)

def create_inputs_dict(events_dir):
  inputs_dict = {}
  event_names = list_dir(events_dir, fn_filter=filter_event_name)

  for event_name in event_names:
    inputs_dict[event_name] = {
      'utterance_id_to_audio_path': [],
      'speaker_id_to_utterance_id': []
    }

  return inputs_dict

def create_input_files(args, subset, event_name_to_inputs):
  input_dir = get_input_dir(args, subset)

  for event_name, inputs in event_name_to_inputs.items():
    unique_speakers = set()

    event_input_dir = input_dir / event_name
    event_input_dir.mkdir(parents=True, exist_ok=True)

    with open(event_input_dir / 'wav.scp', 'wt') as f:
      for pair in inputs['utterance_id_to_audio_path']:
        f.write(f'{pair["utterance_id"]} {pair["audio_path"]}\n')

    with open(event_input_dir / 'spk2utt', 'wt') as f:
      for pair in inputs['speaker_id_to_utterance_id']:
        unique_speakers.add(pair["speaker_id"])
        f.write(f'{pair["speaker_id"]} {pair["utterance_id"]}\n')

    with open(event_input_dir / 'utt2spk', 'wt') as f:
      for pair in inputs['speaker_id_to_utterance_id']:
        f.write(f'{pair["utterance_id"]} {pair["speaker_id"]}\n')

    with open(event_input_dir / 'speakers_count.txt', 'wt') as f:
      f.write(f'{len(unique_speakers)}')

    fix_input_dir(Path(args.kaldi_dir), event_input_dir)

def fix_input_dir(kaldi_dir, event_input_dir):
  subprocess.run(['./utils/fix_data_dir.sh', str(event_input_dir)], cwd=kaldi_dir / kaldi_recipe_dir)

def create_mfcc_features(args, subset):
  input_dir = get_input_dir(args, subset)
  mfcc_features_dir = get_mfcc_features_dir(args, subset)

  for event_name in list_dir(input_dir, fn_filter=filter_event_name):
    target_event_dir = mfcc_features_dir / event_name
    target_event_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(['./steps/make_mfcc.sh',
        '--mfcc-config', args.mfcc_config,
        str(input_dir / event_name),
        str(target_event_dir / 'logs'),
        str(target_event_dir)
      ],
      cwd=Path(args.kaldi_dir) / kaldi_recipe_dir
    )

def create_ivectors(args, subset):
  input_dir = get_input_dir(args, subset)
  mfcc_features_dir = get_mfcc_features_dir(args, subset)
  ivectors_dir = get_ivectors_dir(args, subset)

  for event_name in list_dir(mfcc_features_dir, fn_filter=filter_event_name):
    with open(input_dir / event_name / 'speakers_count.txt', 'rt') as f:
      speakers_count = int(f.read())

    target_event_dir = ivectors_dir / event_name
    target_event_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(['./steps/nnet/ivector/extract_ivectors.sh',
        '--nj', str(speakers_count),
        str(input_dir / event_name),
        str(Path(args.kaldi_dir) / 'pretrained-model' / 'data' / 'lang_test_tgsmall'),
        str(Path(args.kaldi_dir) / 'pretrained-model' / 'exp' / 'nnet3_cleaned' / 'extractor'),
        str(target_event_dir)
      ],
      cwd=Path(args.kaldi_dir) / kaldi_recipe_dir,
      env={'ali_or_decode_dir': ''}
    )

def create_ivector_lists(args, subset):
  ivectors_dir = get_ivectors_dir(args, subset)
  target_dir = reduced_dataset_dir(args.data_dir) / args.target_dir_name / subset

  for event_name in list_dir(ivectors_dir, fn_filter=filter_event_name):
    target_event_dir = target_dir / event_name
    target_event_dir.mkdir(parents=True, exist_ok=True)

    with ReadHelper(f'scp:{str(ivectors_dir/event_name)}/ivectors_utt.scp') as reader:
      for utterance_id, numpy_array in reader:
        file_name = f'{utterance_id[utterance_id.rindex("_")+1 : ]}.npy'
        np.save(target_event_dir / file_name, numpy_array)

def to_speaker_id(speaker_name):
  id = speaker_name.lower()
  id = re.compile('[^a-z ]').sub('', id)
  id = id.strip()
  id = id.replace(' ', '_')
  return id

def get_dir(args, subset, name):
  return args.kaldi_dir / 'data' / subset / name

def get_input_dir(args, subset):
  return get_dir(args, subset, 'input')

def get_mfcc_features_dir(args, subset):
  return get_dir(args, subset, 'mfcc_features')

def get_ivectors_dir(args, subset):
  return get_dir(args, subset, 'ivectors-bin')

if __name__ == '__main__':
  measure_time('extracting ivectors', extract_ivectors)
