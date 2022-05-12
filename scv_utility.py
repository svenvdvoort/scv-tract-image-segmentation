import cv2
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

import numpy as np

""" Gets image data from filesystem from id. Returns image, image resolution (tuple: height, width) and pixel size (float). """
def get_image_data_from_id(id, data_folder):
    [case, day, _, slice_number] = id.split("_")
    image_folder = f"{data_folder}/train/{case}/{case}_{day}/scans/"
    image_filename = f"slice_{slice_number}" # we only know partial filename from id
    for filename in os.listdir(image_folder):
        if image_filename in filename and filename.endswith(".png"):
            image_filename = filename
            break
    [_, image_slice_number, width, height, pixel_width, pixel_height] = image_filename[:-4].split("_")
    assert slice_number == image_slice_number, f"{image_filename} Slice number from filename does not match"
    assert pixel_height == pixel_width, f"{image_filename} Pixel width and height from filename are not equal"
    image = cv2.imread(image_folder + image_filename)
    image = image[:,:,0] / 255.0 # discard redundant RGB information and normalize
    assert image.shape[0] == int(height) and image.shape[1] == int(width), f"{image_filename} Image width or height does not match resolution from filename"
    return image, (int(height), int(width)), float(pixel_width)


class MRIDataset(Dataset):
    """MRI dataset."""

    def __init__(self, data_dir, labels, transform=None, target_transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            labels (pandas series): Pandas series with labels from CSV file.
            transform (callable, optional): Optional transform to be applied
                on an image.
            target_transform (callable, optional): Optional transform to be applied
                on a segmentation mask.
        """
        self.labels = labels.to_dict(orient="list")
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels['id'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id_string = self.labels['id'][idx]
        image, image_resolution, _ = get_image_data_from_id(id_string, self.data_dir)

        segmentation_rle = self.labels['segmentation'][idx]
        segmentation_mask = MRIDataset.convert_segmentation(segmentation_rle, image_resolution)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            segmentation_mask = self.target_transform(segmentation_mask)

        return image, segmentation_mask

    """ Convert run length encoded mask to mask image. """
    def convert_segmentation(rle_string, resolution):
        assert len(resolution) == 2 # mask should have single channel
        mask = np.zeros(resolution[0] * resolution[1])
        if isinstance(rle_string, str) and rle_string != "": # rle_string could be empty
            rle = list(map(int, rle_string.split(" ")))
            for i in range(0, len(rle), 2):
                index = rle[i]
                length = rle[i+1]
                mask[index:index+length] = 1.0
        mask = mask.reshape(resolution)
        return mask
