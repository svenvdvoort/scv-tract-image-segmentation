import numpy as np
import pandas as pd
import sys
from scv_utility import *
from torch.utils.data import DataLoader
import unittest
import math
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

data_folder = "./data/"


class TestExample(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(1 + 1, 2, "1 + 1 should equal 2")

class TestMRISegmentationDataset(unittest.TestCase):
    example_labels = pd.DataFrame(
        [["case123_day20_slice_0001", "large_bowel", ""],
         ["case123_day20_slice_0065", "stomach",
          "28094 3 28358 7 28623 9 28889 9 29155 9 29421 9 29687 9 29953 9 30219 9 30484 10 30750 10 31016 10 31282 10 31548 10 31814 10 32081 9 32347 8 32614 6"]],
        columns=["id", "class", "segmentation"])

    def test_dataset(self):
        dataset = MRISegmentationDataset(data_folder, self.example_labels)
        self.assertEqual(len(dataset), 2)
        image, mask = dataset[0]                 # Join somehow swaps the order of indices here
        np.set_printoptions(threshold=sys.maxsize)
        self.assertEqual(image.shape, (1, 266, 266))
        self.assertEqual(mask.shape, (3, 266, 266))
        self.assertNotEqual(np.max(image), 0)
        self.assertNotEqual(np.max(mask[0]), 0)  # stomach channel has segmentation
        self.assertEqual(np.max(mask[1]), 0)     # small_bowel channel has no segmentation
        self.assertEqual(np.max(mask[2]), 0)     # large_bowel channel has no segmentation
        image, mask = dataset[1]                 # this dataset has no segmentation
        self.assertNotEqual(np.max(image), 0)
        self.assertEqual(np.max(mask), 0)

    def test_dataloader(self):
        dataset = MRISegmentationDataset(data_folder, self.example_labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        sample_images, sample_masks = next(iter(dataloader))
        self.assertEqual(sample_images.shape, (2, 1, 266, 266))
        self.assertEqual(sample_masks.shape, (2, 3, 266, 266))
        self.assertEqual(sample_images.dtype, torch.float32)
        self.assertEqual(sample_masks.dtype, torch.float32)

    def test_convert_segmentation_small(self):
        example_segmentation = "1 3 5 1 15 5"
        example_mask = np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ])
        mask = MRISegmentationDataset.convert_segmentation(example_segmentation, (5, 5))
        self.assertTrue((mask == example_mask).all())

    def test_convert_segmentation_empty(self):
        mask = MRISegmentationDataset.convert_segmentation("", (5, 5))
        self.assertEqual(mask.shape, (5, 5))
        self.assertEqual(np.max(mask), 0)
        mask = MRISegmentationDataset.convert_segmentation(np.NaN, (5, 5))
        self.assertEqual(mask.shape, (5, 5))
        self.assertEqual(np.max(mask), 0)
        
    def test_preprocess(self):
        d = {"id": ["slice_0", "slice_0", "slice_0", "slice_1", "slice_2"], "class": ["stomach", "large_bowel", "small_bowel", "stomach", "large_bowel"], "segmentation": ["1 2", "3 4", "5 6", "7 8", "9 10"]}
        df = pd.DataFrame(data=d, index=[0, 1, 2, 3, 4])

        res = MRISegmentationDataset.preprocess(df)

        assert res.shape == (3, 4)
        assert res[res['id'] == "slice_0"]["small_bowel_segmentation"].values[0] == "5 6"
        assert res[res['id'] == "slice_1"]["stomach_segmentation"].values[0] == "7 8"
        assert res[res['id'] == "slice_2"]["large_bowel_segmentation"].values[0] == "9 10"
        assert math.isnan(res[res['id'] == "slice_2"]["small_bowel_segmentation"].values[0])
        

    def test_label_smoothing(self):
        input_mask = np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ])
        smooth = LabelSmoothing(p=0)
        sample = {'image': np.zeros((5,5)), 'segmentation': input_mask}
        sample = smooth(sample)
        assert (input_mask == sample['segmentation']).all()

        smooth = LabelSmoothing(p=1)
        sample = smooth(sample)
        true_mask = np.array([[0.22, 1, 1, 1, 0.11],
                              [1, 0.33, 0.33, 0.22, 0.11],
                              [0.33, 0.44, 0.33, 0.33, 0.22],
                              [1, 1, 1, 1, 1],
                              [0.22, 0.33, 0.33, 0.33, 0.22]])
        assert (abs(true_mask - sample['segmentation']) < 1e-2).all()

    def test_random_crop(self, visualize=False):
        dataset = MRISegmentationDataset(data_folder, self.example_labels, transform=transforms.Normalize())
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        sample_image, sample_mask = next(iter(dataloader))

        dataset_crop = MRISegmentationDataset(data_folder, self.example_labels, transform=RandomCrop())
        dataloader_crop = DataLoader(dataset_crop, batch_size=1, shuffle=True)
        sample_image_crop, sample_mask_crop = next(iter(dataloader_crop))

        if visualize:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(sample_image[0][0])
            axs[0, 1].imshow(sample_mask[0][0])
            axs[1, 0].imshow(sample_image_crop[0][0])
            axs[1, 1].imshow(sample_mask_crop[0][0])
            plt.show()

        assert sample_image.shape == sample_image_crop.shape
        assert sample_mask.shape == sample_mask_crop.shape

    def test_rescale(self):
        shape = (5, 5)
        new_shape = (10, 10)
        scale = Rescale(new_shape)
        sample = {'image': np.zeros(shape), 'segmentation': np.zeros(shape)}
        sample = scale(sample)
        assert sample['image'].shape == new_shape
        assert sample['segmentation'].shape == new_shape

    def test_normalize(self):
        dataset = MRISegmentationDataset(data_folder, self.example_labels)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        sample_image, sample_mask = next(iter(dataloader))

        dataset_norm = MRISegmentationDataset(data_folder, self.example_labels, transform=Normalize(mean=0.485, std=0.229))
        dataloader_norm = DataLoader(dataset_norm, batch_size=1, shuffle=True)
        sample_image_norm, sample_mask_norm = next(iter(dataloader_norm))

        assert (sample_mask == sample_mask_norm).all()
        assert (sample_image != sample_mask_norm).any()

if __name__ == '__main__':
    unittest.main()
