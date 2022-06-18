from pathlib import Path

import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import iterate_reduced_event_files, filter_tsv, filename_without_extension, audio_segment_file_name


class TextDataset(Dataset):

  def __init__(self, dataset_path, tokenizer, max_seq_length):
    self.df = build_dataframe_for_reduced_dataset(dataset_path)
    self._tokenizer = tokenizer
    self._max_seq_length = max_seq_length

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    row_data = {k: v[0] for k, v in self._tokenizer.encode_plus(row['sentence'],
                                                    max_length=self._max_seq_length,
                                                    truncation=True,
                                                    padding='max_length',
                                                    return_tensors='np').items()}
    add_common_props(row, row_data)
    return row_data

  def __len__(self):
    return len(self.df)

class AudioDataset(Dataset):

  def __init__(self, dataset_path, segments_dir, feature_extractor, max_seq_length):
    self.df = build_dataframe_for_reduced_dataset(dataset_path)
    self._segments_dir = segments_dir
    self._feature_extractor = feature_extractor
    self._max_seq_length = max_seq_length

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    segment_file = Path(self._segments_dir) / row['event'] / audio_segment_file_name(row['audio_file'], row['line_number'])
    waveform, _ = torchaudio.load(segment_file)
    audio = waveform.squeeze(0).tolist()

    row_data = {k: v[0] for k, v in self._feature_extractor(audio,
                                                sampling_rate=16000,
                                                do_normalize=True,
                                                max_length=self._max_seq_length,
                                                truncation=True,
                                                padding='max_length',
                                                return_tensors='np').items()}
    add_common_props(row, row_data)
    return row_data

  def __len__(self):
    return len(self.df)

def build_dataframe_for_reduced_dataset(dataset_path):
  all_dataframes = []
  def collect_data(event_name, event_dir, file_name, file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['event'] = df.apply(lambda row: event_name, axis=1)
    df['audio_file'] = df.apply(lambda row: filename_without_extension(file_name), axis=1)
    df['file_identifier'] = df.apply(lambda row: f'{event_name}-{filename_without_extension(file_name)}', axis=1)
    all_dataframes.append(df)

  iterate_reduced_event_files(Path(dataset_path), collect_data, fn_file_filter=filter_tsv)

  return pd.concat(all_dataframes).reset_index()

def add_common_props(row, row_data):
  row_data['is_claim'] = np.int64(row['is_claim'])
  for column in ['file_identifier', 'line_number']:
    row_data[column] = row[column]
