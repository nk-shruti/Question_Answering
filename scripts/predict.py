# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import argparse
import json
import os

from tqdm import tqdm

from keras import backend as K
from keras.models import Model, load_model

from data import get_text_from_indices, get_reverse_word_index
from evaluator import f1_score, exact_match_score, metric_max_over_ground_truths

def get_best_answer_pointers(starts, ends, clip_at=None):
    if clip_at != None:
        starts = starts[:clip_at]
        ends = ends[:clip_at]

    max_ends = [ends[-1]]
    max_ends_indices = [len(ends) - 1]
    for i in xrange(len(ends) - 2, -1, -1):
        if ends[i] > max_ends[-1]:
            max_ends.append(ends[i])
            max_ends_indices.append(i)
        else:
            max_ends.append(max_ends[-1])
            max_ends_indices.append(max_ends_indices[-1])

    max_ends = max_ends[::-1]
    max_ends_indices = max_ends_indices[::-1]

    scores = np.asarray([s*e for s, e in zip(starts, max_ends)])

    max_score_index = np.argmax(scores)
    start, end = max_score_index, max_ends_indices[max_score_index]

    assert start <= end

    return start, end

def get_metrics(model, dev_data_gen, steps):

    print('Running model for predictions...')

    j = 0
    f1s, ems = [], []
    reverse_word_index = get_reverse_word_index()

    for (c, q), (a_starts, a_ends) in tqdm(dev_data_gen, total=steps):
        preds = model.predict_on_batch([c, q])

        for j in xrange(len(c)):

            c_text = get_text_from_indices(c[j].tolist(), reverse_word_index).split()

            pred_a_start, pred_a_end = get_best_answer_pointers(preds[0][j], preds[1][j], clip_at=len(c_text))

            groundtruths = [' '.join(c_text[np.argmax(a_start[j]):np.argmax(a_end[j]) + 1]) for a_start, a_end in zip(a_starts, a_ends)]
            prediction = ' '.join(c_text[pred_a_start:pred_a_end + 1])
            
            f1s.append(metric_max_over_ground_truths(f1_score, prediction, groundtruths))
            ems.append(metric_max_over_ground_truths(exact_match_score, prediction, groundtruths))

    print('F1 score: {}'.format(float(sum(f1s)) / len(f1s)))
    print('EM score: {}'.format(float(sum(ems)) / len(ems)))


