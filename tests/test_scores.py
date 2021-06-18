import numpy as np
from yoho.metrics.scores import (
    multiclass_precision, multiclass_recall, multiclass_f1,
)


def test_multiclass_precision_1():
    y_true = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    y_pred = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

    precision = multiclass_precision(y_true, y_pred)

    assert precision == 1


def test_multiclass_precision_0():
    y_true = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    y_pred = np.array([[0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0]])

    precision = multiclass_precision(y_true, y_pred)

    assert precision == 0


def test_multiclass_recall_1():
    y_true = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    y_pred = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])

    recall = multiclass_recall(y_true, y_pred)

    assert recall == 1


def test_multiclass_recall_0():
    y_true = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    y_pred = np.array([[0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0]])

    recall = multiclass_recall(y_true, y_pred)

    assert recall == 0


def test_multiclass_f1_1():
    y_true = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    y_pred = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])

    f1 = multiclass_f1(y_true, y_pred)

    assert f1 == 1


def test_multiclass_f1_0():
    y_true = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    y_pred = np.array([[0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0]])

    f1 = multiclass_f1(y_true, y_pred)

    assert f1 == 0
