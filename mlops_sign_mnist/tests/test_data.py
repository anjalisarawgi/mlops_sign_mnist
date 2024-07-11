import os

import pytest
import torch

from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    X_train = torch.load(os.path.join(_PATH_DATA, "processed", "X_train.pt"))
    labels_train = torch.load(os.path.join(_PATH_DATA, "processed", "labels_train.pt"))
    X_test = torch.load(os.path.join(_PATH_DATA, "processed", "X_test.pt"))
    labels_test = torch.load(os.path.join(_PATH_DATA, "processed", "labels_test.pt"))

    assert len(X_train) == 27455, "Train dataset did not have the correct number of samples"
    assert len(X_test) == 7172, "Test dataset did not have the correct number of samples"

    assert labels_train.max().item() <= 24, "Train dataset contains invalid labels"
    assert labels_test.max().item() <= 24, "Test dataset contains invalid labels"

    # Check if all expected labels except 9 and 25 are present in the train and test sets
    expected_labels = set(range(25)) - {9, 25}
    assert all(label in labels_train for label in expected_labels), "Not all train labels are represented"
    assert all(label in labels_test for label in expected_labels), "Not all test labels are represented"


if __name__ == "__main__":
    test_data()
