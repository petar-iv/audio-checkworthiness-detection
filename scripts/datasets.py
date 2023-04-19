from operator import itemgetter

import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class SequenceDataset(Dataset):

  def __init__(self, df, max_sequence_length, column_info, return_attention_masks):
    super(SequenceDataset, self).__init__()
    self.df = df
    self._df_column, self._scaler = itemgetter('name', 'scaler')(column_info)
    self._max_sequence_length = max_sequence_length
    self._return_attention_masks = return_attention_masks

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    file_path = row[self._df_column]

    sequence = np.load(file_path)
    if self._scaler:
      sequence = self._scaler.transform(sequence)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    sequence = self._cut(sequence)
    sequence, attention_mask = self._pad(sequence)

    row_data = {
      'sequence': sequence
    }

    if self._return_attention_masks:
      row_data['attetion_mask'] = attention_mask

    add_common_props(row, row_data)

    return row_data

  def _cut(self, sequence):
    sequence_length = sequence.shape[0]
    if sequence_length <= self._max_sequence_length:
      return sequence

    return sequence[0:self._max_sequence_length, :]

  def _pad(self, sequence):
    # See https://stackoverflow.com/a/68396781

    sequence_length = sequence.shape[0]
    if sequence_length >= self._max_sequence_length:
      return sequence, torch.zeros(sequence_length, dtype=torch.bool) if self._return_attention_masks else None

    padding_length = self._max_sequence_length - sequence_length
    sequence = F.pad(sequence, (0, 0, 0, padding_length))

    return sequence, torch.concat([torch.zeros(sequence_length, dtype=torch.bool), torch.ones(padding_length, dtype=torch.bool)]) if self._return_attention_masks else None

  def __len__(self):
    return len(self.df)


class VectorsDataset(Dataset):

  def __init__(self, df, column_infos):
    super(VectorsDataset, self).__init__()
    self.df = df
    self._column_infos = column_infos

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    row_data = build_features_vector(row, self._column_infos)
    add_common_props(row, row_data)
    return row_data

  def __len__(self):
    return len(self.df)


class TextDataset(Dataset):

  def __init__(self, df, tokenizer, max_seq_length):
    self.df = df
    self._tokenizer = tokenizer
    self._max_seq_length = max_seq_length

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    row_data = extract_text_data(self._tokenizer, row['sentence'], self._max_seq_length)
    add_common_props(row, row_data)
    return row_data

  def __len__(self):
    return len(self.df)


class AudioDataset(Dataset):

  def __init__(self, df, df_column, feature_extractor, max_seq_length):
    self.df = df
    self._df_column = df_column
    self._feature_extractor = feature_extractor
    self._max_seq_length = max_seq_length

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    audio = get_audio(row[self._df_column])
    row_data = extract_audio_data(self._feature_extractor, audio, self._max_seq_length)
    add_common_props(row, row_data)
    return row_data

  def __len__(self):
    return len(self.df)

def add_common_props(row, row_data):
  row_data['is_claim'] = np.int64(row['is_claim'])
  for column in ['file_identifier', 'line_number']:
    row_data[column] = row[column]


class AudioAndTextRepresentationDataset(Dataset):

  def __init__(self, df, audio_df_column, feature_extractor, audio_max_seq_length, text_representation_column):
    self.df = df
    self._audio_df_column = audio_df_column
    self._feature_extractor = feature_extractor
    self._audio_max_seq_length = audio_max_seq_length
    self._text_representation_column = text_representation_column

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    audio = get_audio(row[self._audio_df_column])
    row_data = {}
    row_data['audio_data'] = extract_audio_data(self._feature_extractor, audio, self._audio_max_seq_length)
    row_data['text_representation'] = np.load(row[self._text_representation_column])
    add_common_props(row, row_data)
    return row_data

  def __len__(self):
    return len(self.df)

def extract_text_data(tokenizer, sentence, max_seq_length):
  return {k: v[0] for k, v in tokenizer.encode_plus(sentence,
                                                    max_length=max_seq_length,
                                                    truncation=True,
                                                    padding='max_length',
                                                    return_tensors='np').items()}

def get_audio(file_path):
  waveform, _ = torchaudio.load(file_path)
  return waveform.squeeze(0).tolist()

def build_features_vector(row, selected_columns_info):
  tensors = []
  for column_info in selected_columns_info:
    tensors.append(prepare_tensor(row, column_info))

  return { 'features': torch.cat(tensors) }

def prepare_tensor(row, column_info):
  column_name, scaler = itemgetter('name', 'scaler')(column_info)

  vector =  np.load(row[column_name])
  vectors = [vector]
  if scaler:
    vectors = scaler.transform(vectors)
  return torch.tensor(vectors[0], dtype=torch.float32)

def extract_audio_data(feature_extractor, audio, max_seq_length):
  return {k: v[0] for k, v in feature_extractor(audio,
                                                sampling_rate=16000,
                                                max_length=max_seq_length,
                                                truncation=True,
                                                padding='max_length',
                                                return_tensors='np').items()}
