import argparse
from pathlib import Path
from pprint import pprint

import spacy
nlp = spacy.load('en_core_web_sm')
import numpy as np
import pandas as pd

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_tsv, \
  sentence_audio_base_name, filename_without_extension


def perform_ner():
  args = process_command_line_args()
  labels_dict = build_labels_dict()

  for subset in ['train', 'dev', 'test']:
    perform_ner_for_subset(args, labels_dict, subset)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  return parser.parse_args()

def build_labels_dict():
  labels = nlp.pipe_labels['ner']
  labels_dict = {}
  for idx in range(len(labels)):
    labels_dict[labels[idx]] = idx

  pprint(labels_dict)

  return labels_dict

def perform_ner_for_subset(args, labels_dict, subset):
  source_dir = reduced_dataset_dir(args.data_dir) / '07-sentence-level-alignment' / subset
  target_dir = reduced_dataset_dir(args.data_dir) / 'features' / 'ner' / subset

  def ner(event_name, event_dir, file_name, file_path):
    target_event_dir = target_dir / event_name
    target_event_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path, sep='\t')
    for _, row in df.iterrows():
      vector = process_sentence(labels_dict, row['sentence'])
      output_numpy_file = target_event_dir / f'{sentence_audio_base_name(filename_without_extension(file_name), row["line_number"])}.npy'
      np.save(output_numpy_file, np.array(vector))

  iterate_reduced_event_files(source_dir,
    ner,
    fn_file_filter=filter_tsv
  )

def process_sentence(labels_dict, sentence):
  doc = nlp(sentence)
  vector = len(labels_dict) * [0]
  for ent in doc.ents:
    if ent.label_ not in labels_dict:
      raise Exception(f'No entry for entity "{ent.label_}"')

    idx = labels_dict[ent.label_]
    vector[idx] = vector[idx] + 1

  return vector

if __name__ == '__main__':
  measure_time('extracting named entities', perform_ner)
