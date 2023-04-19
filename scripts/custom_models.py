import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer

from utils import get_device


class RNNModel(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, num_directions, dropout):
    super(RNNModel, self).__init__()

    self._num_layers = num_layers
    self._hidden_size = hidden_size
    self._dropout = dropout
    self._num_directions = num_directions
    bidirectional = num_directions == 2

    self._rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
    self._linear = nn.Linear(num_directions * hidden_size, 2)

  def forward(self, sequence):
    batch_size = sequence.shape[0]
    h0 = torch.zeros(self._num_layers * self._num_directions, batch_size, self._hidden_size).to(get_device())
    result = self._rnn(sequence, h0)[0]
    # See https://youtu.be/0_PgWWmauHk?t=663
    result = result[:, -1, :] # get the hidden states for the last time step only
    result = F.dropout(result, self._dropout, self.training)
    result = self._linear(result)
    return CompatibleResult(result)


class TransformerModel(nn.Module):

  def __init__(self, input_size, num_heads, activation, dropout, num_layers, max_sequence_size):
    super(TransformerModel, self).__init__()

    self._input_size = input_size # input size = embedding size
    self._dropout = dropout

    full_sequence_size = max_sequence_size + 1 # + the [CLS] embedding

    self._pos_encoder = Summer(PositionalEncodingPermute1D(full_sequence_size))
    encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, activation=activation, dropout=dropout, batch_first=True)
    self._encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self._linear = nn.Linear(input_size, 2)

  def forward(self, sequence, attetion_mask):
    batch_size = sequence.shape[0]

    # add the [CLS] embedding
    sequence = torch.cat([torch.zeros(batch_size, 1, self._input_size).to(get_device()), sequence], dim=1)
    attetion_mask = torch.cat([torch.zeros(batch_size, 1).to(get_device()), attetion_mask], dim=1)

    # See https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    sequence = sequence * math.sqrt(self._input_size)
    sequence = self._pos_encoder(sequence)
    # See https://stackoverflow.com/a/68396781
    result = self._encoder(src=sequence, src_key_padding_mask=attetion_mask)
    result = result[:, 0, :] # get the [CLS] embedding
    result = F.dropout(result, self._dropout, self.training)
    result = self._linear(result)
    return CompatibleResult(result)


class FeedforwardNetwork(nn.Module):

  def __init__(self, input_size, hidden_layer_sizes, dropout):
    super(FeedforwardNetwork, self).__init__()

    output_size = 2
    activation = nn.ReLU()

    layer_pairs = self._build_layer_pairs(input_size, hidden_layer_sizes, output_size)

    layers = []
    for i, pair in enumerate(layer_pairs):
      layers.append(nn.Linear(pair[0], pair[1]))
      if i < len(layer_pairs) - 1:
        layers.append(activation)
        layers.append(nn.Dropout(dropout))

    self.layers = nn.Sequential(*layers)

  def _build_layer_pairs(self, input_size, hidden_layer_sizes, output_size):
    hidden_layer_sizes = [] if hidden_layer_sizes is None else hidden_layer_sizes
    duplicated_hidden_layer_sizes = []
    for hidden_layer_size in hidden_layer_sizes:
      duplicated_hidden_layer_sizes.extend([hidden_layer_size, hidden_layer_size])

    all_sizes = []
    all_sizes.append(input_size)
    all_sizes.extend(duplicated_hidden_layer_sizes)
    all_sizes.append(output_size)

    layer_pairs = []
    for i in range(0, len(all_sizes), 2):
      layer_pairs.append((all_sizes[i], all_sizes[i+1]))

    return layer_pairs

  def forward(self, features):
    result = self.layers(features)
    return CompatibleResult(result)


class AlignedAudioModel(nn.Module):

  def __init__(self, audio_model, text_model):
    super(AlignedAudioModel, self).__init__()

    self._audio_model = audio_model
    self._text_model = text_model

    # For adjusting the hidden size of the audio model to the hidden size of the text model
    self._size_adjustment =  None if audio_model.config.hidden_size == text_model.config.hidden_size else nn.Linear(audio_model.config.hidden_size, text_model.config.hidden_size)

  def classify(self, audio_data):
    _, result = self.represent_and_classify(audio_data)
    return CompatibleResult(result)

  def represent_and_classify(self, audio_data):
    audio_representation = self._audio_model(**audio_data).last_hidden_state
    audio_representation = audio_representation.mean(dim=1)

    if self._size_adjustment is not None:
      audio_representation = self._size_adjustment(audio_representation)

    logits = self._text_model.classifier(audio_representation)
    return audio_representation, logits

  def forward(self, audio_data):
    return self.classify(audio_data)

class CompatibleResult():

  def __init__(self, logits):
    self.logits = logits
