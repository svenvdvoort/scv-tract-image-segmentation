import numpy as np
import pandas as pd
from scv_utility import *
from torch.utils.data import DataLoader
import unittest

data_folder = "./data/"

class TestExample(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(1 + 1, 2, "1 + 1 should equal 2")

class TestMRIDataset(unittest.TestCase):

    example_labels = pd.DataFrame(
        [["case123_day20_slice_0001", "large_bowel", ""],
         ["case123_day20_slice_0065", "stomach", "28094 3 28358 7 28623 9 28889 9 29155 9 29421 9 29687 9 29953 9 30219 9 30484 10 30750 10 31016 10 31282 10 31548 10 31814 10 32081 9 32347 8 32614 6"]],
        columns=["id", "class", "segmentation"])

    def test_dataset(self):
        dataset = MRIDataset(data_folder, self.example_labels)
        self.assertEqual(len(dataset), 2)
        image, mask = dataset[0]
        self.assertEqual(image.shape, (266, 266))
        self.assertEqual(mask.shape, (266, 266))
        self.assertNotEqual(np.max(image), 0)
        self.assertEqual(np.max(mask), 0)
        image, mask = dataset[1]
        self.assertNotEqual(np.max(image), 0)
        self.assertNotEqual(np.max(mask), 0)

    def test_convert_segmentation_small(self):
        example_segmentation = "1 3 5 1 15 5"
        example_mask = np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ])
        mask = MRIDataset.convert_segmentation(example_segmentation, (5, 5))
        self.assertTrue((mask == example_mask).all())

    def test_convert_segmentation_empty(self):
        mask = MRIDataset.convert_segmentation("", (5, 5))
        self.assertEqual(mask.shape, (5, 5))
        self.assertEqual(np.max(mask), 0)
        mask = MRIDataset.convert_segmentation(np.NaN, (5, 5))
        self.assertEqual(mask.shape, (5, 5))
        self.assertEqual(np.max(mask), 0)

    def test_dataloader(self):
        dataset = MRIDataset(data_folder, self.example_labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
        # TODO test dataloader?

if __name__ == '__main__':
    unittest.main()
