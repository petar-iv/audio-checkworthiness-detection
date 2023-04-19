import os
import argparse
from pathlib import Path

import torchaudio
import pandas as pd
from pydub import AudioSegment
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_tsv, \
  sentence_audio_base_name, filename_without_extension, adjust_segment, move_tensors_to_device, \
  get_device


def generate_voice():
  args = process_command_line_args()
  convert = prepare_convert_function()

  for subset in ['train', 'dev', 'test']:
    generate_voice_reduced(args, subset, convert)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--target_dir_name', required=True, type=str)
  return parser.parse_args()

def prepare_convert_function():
  models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    'facebook/fastspeech2-en-ljspeech',
    arg_overrides={'vocoder': 'hifigan', 'fp16': False}
  )
  model = models[0].to(get_device())
  TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
  generator = task.build_generator(models, cfg)

  def convert(text):
    sample = TTSHubInterface.get_model_input(task, text)
    move_tensors_to_device(sample)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return wav.unsqueeze(0), rate

  return convert

def generate_voice_reduced(args, subset, convert):
  source_dir = reduced_dataset_dir(args.data_dir) / '07-sentence-level-alignment' / subset
  target_dir = reduced_dataset_dir(args.data_dir) / args.target_dir_name / subset

  def generate_voice_for_file(event_name, event_path, file_name, file_path):
    df = pd.read_csv(file_path, sep='\t')
    for _, entry in df.iterrows():
      waveform, sample_rate = convert(entry['sentence'])

      event_target_dir = target_dir / event_name
      event_target_dir.mkdir(parents=True, exist_ok=True)
      audio_file_path = event_target_dir / f'{sentence_audio_base_name(filename_without_extension(file_name), entry["line_number"])}.wav'

      waveform = waveform.detach().cpu()
      torchaudio.save(audio_file_path, waveform, sample_rate)

      audio = AudioSegment.from_file(audio_file_path)
      os.remove(audio_file_path)
      audio = adjust_segment(audio)
      audio.export(audio_file_path, format='wav')

  iterate_reduced_event_files(source_dir, generate_voice_for_file, filter_tsv, should_measure_time=True)

if __name__ == '__main__':
  measure_time('generating voice', generate_voice)
