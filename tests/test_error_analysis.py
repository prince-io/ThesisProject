import numpy as np

from litevit.training.error_analysis import _iou_matrix, _match, _size_bin


def test_iou_and_matching() -> None:
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=float)
    gt_classes = np.array([0])
    pred_boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    pred_classes = np.array([0, 1])
    assert _iou_matrix(gt_boxes, pred_boxes)[0, 0] == 1.0
    matches, used_pred = _match(
        gt_boxes, gt_classes, pred_boxes, pred_classes, 0.5, same_class=True
    )
    assert len(matches) == 1
    assert matches[0][0] == 0
    assert matches[0][1] == 0
    assert used_pred == {0}


def test_any_class_matching_reports_confusion() -> None:
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=float)
    gt_classes = np.array([2])
    pred_boxes = np.array([[0, 0, 10, 10]], dtype=float)
    pred_classes = np.array([1])
    matches, _ = _match(
        gt_boxes, gt_classes, pred_boxes, pred_classes, 0.5, same_class=False
    )
    assert len(matches) == 1


def test_size_bins() -> None:
    assert _size_bin(0.0001) == "tiny"
    assert _size_bin(0.002) == "small"
    assert _size_bin(0.02) == "medium"
    assert _size_bin(0.2) == "large"
