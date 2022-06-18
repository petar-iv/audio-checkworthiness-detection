# Code taken from:
# - https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/blob/ec6370e19f67f63772aff963cd1ee48284d7a599/task1/scorer/subtask_1b.py
# and
# - https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/blob/ec6370e19f67f63772aff963cd1ee48284d7a599/task1/scorer/utils.py


MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 50]

def evaluate(actual, predicted, main_thresholds=MAIN_THRESHOLDS):
  """Evaluate prediction results.

    @param: actual List of dicts (key - line number, int; value - 1 if line contains claim, 0 otherwise, int)
    @param: predicted List of dicts (key - line number, int; value - check worthiness of line, float)
    """
  if len(predicted) != len(actual):
    raise Exception(f'Length mismatch - predicted: {len(predicted)}, actual: {len(actual)}')

  overall_precisions = [0.0] * len(main_thresholds)
  mean_r_precision = 0.0
  mean_avg_precision = 0.0
  mean_reciprocal_rank = 0.0

  for (single_actual, single_predicted) in zip(actual, predicted):
    thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate_single(single_actual, single_predicted, main_thresholds)

    threshold_precisions = [precisions[th - 1] for th in main_thresholds]
    r_precision = precisions[num_relevant - 1]

    for idx in range(0, len(main_thresholds)):
        overall_precisions[idx] += threshold_precisions[idx]
    mean_r_precision += r_precision
    mean_avg_precision += avg_precision
    mean_reciprocal_rank += reciprocal_rank

  events_count = len(predicted)
  if events_count > 1:
      overall_precisions = [item * 1.0 / events_count for item in overall_precisions]
      mean_r_precision /= events_count
      mean_avg_precision /= events_count
      mean_reciprocal_rank /= events_count

  return {
      'mean_avg_precision': mean_avg_precision,
      'mean_reciprocal_rank': mean_reciprocal_rank,
      'mean_r_precision': mean_r_precision,
      'overall_precisions': overall_precisions
  }

def evaluate_single(actual, predicted, main_thresholds):
  assert type(actual) == dict
  assert type(predicted) == dict
  assert actual.keys() == predicted.keys()

  ranked_lines = [t[0] for t in sorted(predicted.items(), key=lambda x: x[1], reverse=True)]
  thresholds = main_thresholds + [len(ranked_lines)]

  precisions = _compute_precisions(actual, ranked_lines, len(ranked_lines))
  avg_precision = _compute_average_precision(actual, ranked_lines)
  reciprocal_rank = _compute_reciprocal_rank(actual, ranked_lines)
  num_relevant = len({k for k, v in actual.items() if v == 1})

  return thresholds, precisions, avg_precision, reciprocal_rank, num_relevant

def _compute_average_precision(gold_labels, ranked_lines):
    """ Computes Average Precision. """

    precisions = []
    num_correct = 0
    num_positive = sum([1 if v == 1 else 0 for k, v in gold_labels.items()])

    for i, line_number in enumerate(ranked_lines):
        if gold_labels[line_number] == 1:
            num_correct += 1
            precisions.append(num_correct / (i + 1))
    if precisions:
        avg_prec = sum(precisions) / num_positive
    else:
        avg_prec = 0.0

    return avg_prec


def _compute_reciprocal_rank(gold_labels, ranked_lines):
    """ Computes Reciprocal Rank. """
    rr = 0.0
    for i, line_number in enumerate(ranked_lines):
        if gold_labels[line_number] == 1:
            rr += 1.0 / (i + 1)
            break
    return rr


def _compute_precisions(gold_labels, ranked_lines, threshold):
    """ Computes Precision at each line_number in the ordered list. """
    precisions = [0.0] * threshold
    threshold = min(threshold, len(ranked_lines))

    for i, line_number in enumerate(ranked_lines[:threshold]):
        if gold_labels[line_number] == 1:
            precisions[i] += 1.0

    for i in range(1, threshold): # accumulate
        precisions[i] += precisions[i - 1]
    for i in range(1, threshold): # normalize
        precisions[i] /= i+1
    return precisions
