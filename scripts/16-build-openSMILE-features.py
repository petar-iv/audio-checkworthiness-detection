import os
import argparse
import subprocess
from pathlib import Path

import arff
import numpy as np
from HTKFile import HTKFile

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_wav, filename_without_extension


def build_openSMILE_features():
  args = process_command_line_args()

  for subset in ['train', 'dev', 'test']:
    source_dir = reduced_dataset_dir(args.data_dir) / args.segments_dir_name / subset
    target_dir = reduced_dataset_dir(args.data_dir) / args.target_dir_name / subset

    measure_time(f'processing {subset}', lambda: create_features_for_subset(args, subset, source_dir, target_dir))

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--segments_dir_name', required=True, type=str)
  parser.add_argument('--target_dir_name', required=True, type=str)
  parser.add_argument('--openSMILE_dir', required=True, type=Path)
  parser.add_argument('--feature_set', required=True, type=str, choices=['mfcc', 'compare-2013', 'compare-2016'])
  args = parser.parse_args()

  if args.feature_set == 'mfcc':
    args.config_file = args.openSMILE_dir / 'config' / 'mfcc' / 'MFCC12_0_D_A_Z.conf'
  elif args.feature_set == 'compare-2013':
    args.config_file = args.openSMILE_dir / 'config' / 'is09-13' / 'IS13_ComParE.conf'
  else:
    args.config_file = args.openSMILE_dir / 'config' / 'compare16' / 'ComParE_2016.conf'

  return args

def create_features_for_subset(args, subset, source_dir, target_dir):
  def create_features_file(event_name, event_dir, file_name, file_path):
    target_event_dir = target_dir / event_name
    target_event_dir.mkdir(parents=True, exist_ok=True)

    input_audio_file = file_path
    openSMILE_output_file = target_event_dir / f'{filename_without_extension(file_name)}.htk' if args.feature_set == 'mfcc' else target_event_dir / f'{filename_without_extension(file_name)}.arff'
    output_numpy_file = target_event_dir / f'{filename_without_extension(file_name)}.npy'

    openSMILE_args = [
        args.openSMILE_dir / 'bin' / 'SMILExtract',
        '-C',
        args.config_file,
        '-I',
        input_audio_file,
        '-O',
        openSMILE_output_file
    ]

    if args.feature_set != 'mfcc':
      openSMILE_args.extend([
        '-instname',
        f'{subset}-{event_name}-{filename_without_extension(file_name)}'
      ])

    subprocess.run(openSMILE_args)

    features = load_openSMILE_features(openSMILE_output_file)
    np.save(output_numpy_file, np.array(features))
    os.remove(openSMILE_output_file)

  iterate_reduced_event_files(source_dir, create_features_file, fn_file_filter=filter_wav, should_measure_time=True)

def load_openSMILE_features(file_path):
  return load_arff_data(file_path) if str(file_path).endswith('.arff') else load_htk_data(file_path)

def load_arff_data(file_path):
  with open(file_path) as f:
    content = arff.load(f)
    assert len(content['data']) == 1, f'length of "data" for file "{file_path}" is {len(content["data"])}'
    data = content['data'][0]
    data = data[1 : -1] # omit name and class
    return data

def load_htk_data(file_path):
  htk_file = HTKFile()
  htk_file.load(file_path)
  return htk_file.data

if __name__ == '__main__':
  measure_time('building openSMILE features', build_openSMILE_features)
