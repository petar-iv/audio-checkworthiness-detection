import argparse
from pathlib import Path

import pandas as pd

from utils import reduced_dataset_dir


def create_begin_end_alignments():
  args = process_command_line_args()

  alignment_dir = reduced_dataset_dir(args.data_dir) / '03-begin-end-alignment'
  alignment_dir.mkdir(parents=True, exist_ok=True)

  create_train_alignment(alignment_dir)
  create_dev_alignment(alignment_dir)
  create_test_alignment(alignment_dir)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  return parser.parse_args()

def create_train_alignment(alignment_dir):
  alignments = [
    ('20121023_third_presidential_debate', 'audio-1.wav', 8, 1, 5550, 1073),
    ('20150805_GOP_Cleveland', 'audio-1.wav', 275, 1, 6798, 1564),
    ('20151013_democrats_las_Vegas', 'audio-1.wav', 428, 1, 7868, 1627),
    ('20151028_GOP_boulder', 'audio-1.wav', 100, 1, 6712, 1823),
    ('20151110_GOP_Milwaukee', 'audio-1.wav', 2, 1, 6998, 1645),
    ('20160112_Obama_state_of_the_union', 'audio-1.wav', 965, 1, 3629, 228),
    ('20160115_GOP_Charleston', 'audio-1.wav', 6, 1, 1235, 217),
    ('20160115_GOP_Charleston', 'audio-2.wav', 6, 220, 1292, 592),
    ('20160115_GOP_Charleston', 'audio-3.wav', 5, 594, 1182, 859),
    ('20160115_GOP_Charleston', 'audio-4.wav', 6, 861, 1653, 1276),
    ('20160115_GOP_Charleston', 'audio-5.wav', 7, 1278, 724, 1456),
    ('20160115_GOP_Charleston', 'audio-6.wav', 10, 1458, 929, 1663),
    ('20160115_GOP_Charleston', 'audio-7.wav', 7, 1668, 473, 1760),
    ('20160117_democrats_Charleston', 'audio-1.wav', 65, 1, 6819, 1090),
    ('20160204_democrats_new_Hampshire', 'audio-1.wav', 1, 13, 6332, 1209),
    ('20160211_democrats_Milwaukee', 'audio-1.wav', 94, 1, 6810, 1002),
    ('20160213_GOP_Greenville', 'audio-1.wav', 53, 1, 5936, 1615),
    ('20160225_GOP_Texas', 'audio-1.wav', 216, 37, 7615, 2216),
    ('20160303_GOP_Michigan', 'audio-1.wav', 0, 7, 6157, 1956),
    ('20160309_democrats_Miami', 'audio-1.wav', 570, 31, 7218, 1197),
    ('20160311_GOP_Miami', 'audio-1.wav', 216, 28, 6864, 1731),
    ('20160401_Trump_Milwaukee', 'audio-1.wav', 2, 1, 1694, 627),
    ('20160404_Hillary_meet_the_press', 'audio-1.wav', 0, 78, 211, 106),
    ('20160404_Hillary_meet_the_press', 'audio-2.wav', 0, 159, 123, 185),
    ('20160404_Hillary_meet_the_press', 'audio-3.wav', 0, 273, 362, 354),
    ('20160404_Hillary_meet_the_press', 'audio-4.wav', 0, 405, 370, 508),
    ('20160404_Hillary_meet_the_press', 'audio-5.wav', 6, 512, 306, 584),
    ('20160415_NY_democratic_debate', 'audio-1.wav', 230, 1, 7184, 1471),
    ('20160417_meet_the_press', 'audio-1.wav', 0, 35, 559, 148),
    ('20160417_meet_the_press', 'audio-2.wav', 0, 172, 42, 178),
    ('20160417_meet_the_press', 'audio-3.wav', 0, 296, 270, 361),
    ('20160417_meet_the_press', 'audio-4.wav', 0, 380, 580, 514),
    ('20160424_meet_the_press', 'audio-1.wav', 27, 314, 853, 444),
    ('20160508_Trump_meet_the_press', 'audio-1.wav', 0, 71, 946, 512),
    ('20160508_Trump_meet_the_press', 'audio-2.wav', 0, 692, 298, 749),
    ('20160529_Sanders_MeetThePress', 'audio-1.wav', 0, 74, 831, 232),
    ('20160529_Sanders_MeetThePress', 'audio-2.wav', 0, 380, 456, 473),
    ('20160622_Trump_NY_Soho', 'audio-1.wav', 34, 0, 2474, 193),
    ('20160721_Trump_draft_RNC', 'audio-1.wav', 39, 1, 4516, 266),
    ('20160901_Trump_immigration_Phoenix', 'audio-1.wav', 30, 1, 3818, 195),
    ('20160907_NBC_commander_in_chief_forum', 'audio-1.wav', 5, 1, 1633, 312),
    ('20160907_NBC_commander_in_chief_forum', 'audio-2.wav', 10, 321, 1541, 769),
    ('20160926_first_presidential_debate', 'audio-1.wav', 189, 1, 5916, 1403),
    ('20161005_vice_presidential_debate', 'audio-1.wav', 1, 1, 5541, 1358),
    ('20161010_second_presidential_debate', 'audio-1.wav', 67, 1, 5606, 1303),
    ('20161019_third_presidential_debate', 'audio-1.wav', 7, 1, 5598, 1351),
    ('20170111_Trump_press_conference', 'audio-1.wav', 0, 4, 3739, 732),
    ('20170120_Trump_inauguration', 'audio-1.wav', 35, 1, 1010, 102),
    ('20170126_Trump_ABC', 'audio-1.wav', 0, 1, 2280, 729),
    ('20170205_Trump_Oreilly', 'audio-1.wav', 5, 2, 1197, 388),
    ('20170207_Sanders_Cruz_healthcare_debate', 'audio-1.wav', 11, 1, 5607, 1201),
    ('20170216_Trump_press_conference', 'audio-1.wav', 0, 1, 4614, 1260),
    ('20170218_Trump_Florida', 'audio-1.wav', 710, 1, 3578, 532),
    ('20170224_Trump_CPAC', 'audio-1.wav', 60, 1, 2953, 598)
  ]

  create_alignment(alignment_dir, 'train', alignments)

def create_dev_alignment(alignment_dir):
  alignments = [
    ('20170228_Trump_Congress_joint_session', 'audio-1.wav', 742, 1, 4353, 413),
    ('20170315_Trump_Nashville', 'audio-1.wav', 41, 1, 2273, 404),
    ('20170404_Trump_CEO_TownHall', 'audio-1.wav', 415, 2, 1819, 316),
    ('20170430_Trump_CBS_FactTheNation', 'audio-1.wav', 24, 1, 1389, 548),
    ('20170601_Trump_Paris_Climate', 'audio-1.wav', 3, 18, 1862, 232),
    ('20170713_Trump_Robertson_interview', 'audio-1.wav', 0, 1, 1540, 563),
    ('20170724_Trump_healthcare', 'audio-1.wav', 115, 1, 878, 105)
  ]

  create_alignment(alignment_dir, 'dev', alignments)

def create_test_alignment(alignment_dir):
  alignments = [
    ('20170512_Trump_NBC_holt_interview', 'audio-1.wav', 0, 1, 781, 246),
    ('20170803_Trump_WV', 'audio-1.wav', 0, 57, 2273, 291),
    ('20170822_Trump_phoenix', 'audio-1.wav', 2, 2, 4547, 792),
    ('20180426_Trump_Fox_Friends', 'audio-1.wav', 7, 1, 1782, 597),
    ('20180525_Trump_Naval', 'audio-1.wav', 23, 3, 2102, 279),
    ('20180612_Trump_Singapore', 'audio-1.wav', 1, 1, 3924, 1244),
    ('20180615_Trump_lawn', 'audio-1.wav', 0, 1, 1824, 813),
    ('20180628_Trump_NorthDakota', 'audio-1.wav', 108, 2, 4324, 1036)
  ]

  create_alignment(alignment_dir, 'test', alignments)

def create_alignment(alignment_dir, subset, alignments):
  data = []

  for single_alignment in alignments:
    data.append({
      'event': single_alignment[0],
      'audio_file': single_alignment[1],

      'audio_begin': single_alignment[2],
      'transcript_begin': single_alignment[3],

      'audio_end': single_alignment[4],
      'transcript_end': single_alignment[5]
    })

  df = pd.DataFrame(data)
  df.to_csv(str(alignment_dir / f'{subset}.tsv'), sep='\t', index=False)

if __name__ == '__main__':
  create_begin_end_alignments()
