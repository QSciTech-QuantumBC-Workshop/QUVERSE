import os
import numpy as np
import pandas as pd
import unittest

from helpers import get_dataset, create_animation, normalize_feat


class TestGetDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = "./dataset"

    def test_get_dataset(self):
        train_data, test_data = get_dataset(self.dataset_dir)

        # Assert that the train_data and test_data are not None
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(test_data)

        # Assert that the train_data and test_data are of type pandas DataFrame
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)

        # Assert that the train_data and test_data have the expected columns
        expected_columns = ["time", "displacement", "velocity", "acceleration"]
        self.assertListEqual(list(train_data.columns), expected_columns)
        self.assertListEqual(list(test_data.columns), expected_columns)

        # Assert that the train_data and test_data have the expected number of rows
        expected_train_rows = 200
        expected_test_rows = 300
        self.assertEqual(len(train_data), expected_train_rows)
        self.assertEqual(len(test_data), expected_test_rows)


class TestCreateAnimation(unittest.TestCase):
    def setUp(self) -> None:
        # Create a sample DataFrame for testing
        self.train_data = pd.DataFrame(
            {"time": [0, 1, 2, 3, 4], "displacement": [0, 1, 0, -1, 0]}
        )
        self.animation_name = "test_animation.mp4"

    def test_create_animation(self):
        # Call the function under test
        create_animation(self.train_data, self.animation_name)

        # Assert that the animation file is created
        assert os.path.exists(self.animation_name)

        # Clean up the animation file
        os.remove(self.animation_name)


# class TestNormalizeFeat(unittest.TestCase):
#     # Test case 1: Normalizing a 2D array with positive values
#     features1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     expected1 = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
#     output1 = normalize_feat(features1)
#     assert np.array_equal(output1, expected1), "Test case 1 failed"

#     # Test case 2: Normalizing a 2D array with negative values
#     features2 = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])
#     expected2 = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
#     output2 = normalize_feat(features2)
#     assert np.array_equal(output2, expected2), "Test case 2 failed"

#     # Test case 3: Normalizing a 2D array with mixed positive and negative values
#     features3 = np.array([[-1, 2, -3], [4, -5, 6], [-7, 8, -9]])
#     expected3 = np.array([[0, 0.5, 0], [0.75, 0, 0.75], [0.25, 1, 0.25]])
#     output3 = normalize_feat(features3)
#     assert np.array_equal(output3, expected3), "Test case 3 failed"

#     # Test case 4: Normalizing a 2D array with all zeros
#     features4 = np.zeros((3, 3))
#     expected4 = np.zeros((3, 3))
#     output4 = normalize_feat(features4)
#     assert np.array_equal(output4, expected4), "Test case 4 failed"

#     print("All test cases passed!")


if __name__ == "__main__":
    unittest.main()
