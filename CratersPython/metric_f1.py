# coding: utf-8
from __future__ import absolute_import
from mxnet import metric
"""Online evaluation metric module."""

# from .base import string_types
import numpy

class F1(metric.EvalMetric):
    """Calculate F1-Score in terms of label 1"""
    def __init__(self):
        self.focusLabel = 1;
        super(F1, self).__init__('f1')

    def reset(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        super(F1, self)
        
    def get(self):
        details = "total:" + `self.fn + self.fp + self.tn + self.tp` + ",tp:" + `self.tp` + ",tn:" + `self.tn` + ",fp:" + `self.fp` + ",fn:" + `self.fn`
        try:
            precision = self.tp / (self.tp + self.fp)
            recall = self.tp / (self.tp + self.fn)
            return (details + ", " + self.name, 2 * (precision * recall) / (precision + recall))
        except:
            return (details + ", " + self.name, 0.0)

    def update(self, labels, preds):
        assert len(labels) == len(preds)
        for i in range(len(labels)):
            pred = preds[i].asnumpy()
            label = labels[i].asnumpy().astype('int32')
            pred_label = numpy.argmax(pred, axis=1)
            if label.shape[0] < pred_label.shape[0]:
                raise Exception("Predict label is more than data label? ")
            ### calc f1 here???!!!
            correct = pred_label == label[:pred_label.shape[0]]
            incorrect = pred_label != label[:pred_label.shape[0]]
            pos = pred_label == self.focusLabel
            neg = pred_label != self.focusLabel
            self.tp += numpy.sum(correct & pos)
            self.tn += numpy.sum(correct & neg)
            self.fp += numpy.sum(incorrect & pos)
            self.fn += numpy.sum(incorrect & neg)
            #print("tp = " + `self.tp`)
            #print("tn = " + `self.tn`)
            #print("fp = " + `self.fp`)
            #print("fn = " + `self.fn`)
