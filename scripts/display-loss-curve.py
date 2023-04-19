import argparse

import numpy as np
import matplotlib.pyplot as plt


def display_loss_curve():
  args = process_command_line_args()

  losses = np.load(args.losses_file)
  processed = process_losses(args, losses)

  plt.figure(figsize=(10, 7))
  plt.ylim((0, 0.75))
  plt.xticks([])
  plt.plot(processed)
  plt.show()

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--losses_file', required=True, type=str)
  parser.add_argument('--num_epochs', type=int, default=15)
  parser.add_argument('--up_to_epoch', required=True, type=int)
  parser.add_argument('--no_processing', action='store_true')
  parser.add_argument('--window_size', type=int, default=10)
  parser.add_argument('--with_overlapping', action='store_true')
  return parser.parse_args()

def process_losses(args, losses):
  if args.up_to_epoch:
    print(f'Take losses for {args.up_to_epoch}/{args.num_epochs} epochs')
    losses_count_per_epoch = len(losses) / args.num_epochs
    print(f'Losses per epoch: {losses_count_per_epoch}')
    losses_to_display = args.up_to_epoch * losses_count_per_epoch
    print(f'Losses to display: {losses_to_display}')
    losses = losses[0:int(losses_to_display)]

  if args.no_processing:
    return no_processing(losses)

  if args.with_overlapping:
    return processing_with_overlapping(args, losses)

  return processing_without_overlapping(args, losses)

def no_processing(losses):
  return losses

def processing_with_overlapping(args, losses):
  normalized_losses = []
  for idx in range(len(losses) - args.window_size + 1):
    window = losses[idx:idx+args.window_size]
    assert len(window) == args.window_size
    normalized_losses.append(sum(window) / len(window))

  return normalized_losses

def processing_without_overlapping(args, losses):
  normalized_losses = []
  for idx in range(0, len(losses), args.window_size):
    window = losses[idx:idx+args.window_size]
    normalized_losses.append(sum(window) / len(window))

  return normalized_losses

if __name__ == '__main__':
  display_loss_curve()
