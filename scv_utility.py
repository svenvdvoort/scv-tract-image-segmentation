import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from scipy.signal import convolve2d
import random

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
    image = image[:,:,0] / 255.0 # discard redundant RGB information and normalize to [0, 1]
    assert image.shape[0] == int(height) and image.shape[1] == int(width), f"{image_filename} Image width or height does not match resolution from filename"
    return image, (int(height), int(width)), float(pixel_width)


class MRIClassificationDataset(Dataset):
    """MRI dataset."""

    def __init__(self, data_dir, labels, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            labels (pandas series): Pandas series with labels from CSV file.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.labels = labels.to_dict(orient="list")
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels['id'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id_string = self.labels['id'][idx]
        image, _, _ = get_image_data_from_id(id_string, self.data_dir)


        if self.transform:
            sample = {'image': image}
            transformed_sample = self.transform(sample)
            image = transformed_sample['image']

        segmentation_rle = self.labels['segmentation'][idx]
        target = 0 if segmentation_rle == "" else 1

        image = image[np.newaxis, ...].astype("float32") # add color dimension
        target = np.array([target], dtype="float32")
        return image, target

class MRISegmentationDataset(Dataset):
    """MRI dataset."""
    
     
    def preprocess(data):
        """ 
            This method takes a df of the ground-truth segmentation csv file and processes it such that the 
            segmentation information is over different columns. I.e. one column per organ. 
        """
        stomachs = data[data["class"] == "stomach"] \
            .rename(columns={"segmentation": "stomach_segmentation"}) \
            .drop(columns=["class"])
        sbowels = data[data["class"] == "small_bowel"] \
            .rename(columns={"segmentation": "small_bowel_segmentation"}) \
            .drop(columns=["class"])
        lbowels = data[data["class"] == "large_bowel"] \
            .rename(columns={"segmentation": "large_bowel_segmentation"}) \
            .drop(columns=["class"])

        alltogether = stomachs.merge(sbowels, on="id", how='outer').merge(lbowels, on="id", how='outer')
        return alltogether

    def __init__(self, data_dir, labels, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            labels (pandas series): Pandas series with labels from CSV file.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.labels = MRISegmentationDataset.preprocess(labels).to_dict(orient="list")
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels['id'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id_string = self.labels['id'][idx]
        image, image_resolution, _ = get_image_data_from_id(id_string, self.data_dir)

        stomach_segmentation_rle = self.labels['stomach_segmentation'][idx]
        stomach_segmentation_mask = MRISegmentationDataset.convert_segmentation(stomach_segmentation_rle, image_resolution)
        
        small_bowel_segmentation_rle = self.labels['small_bowel_segmentation'][idx]
        small_bowel_segmentation_mask = MRISegmentationDataset.convert_segmentation(small_bowel_segmentation_rle, image_resolution)
        
        large_bowel_segmentation_rle = self.labels['large_bowel_segmentation'][idx]
        large_bowel_segmentation_mask = MRISegmentationDataset.convert_segmentation(large_bowel_segmentation_rle, image_resolution)

        if self.transform:
            sample = {'image': image,
                      'segmentation_stomach': stomach_segmentation_mask,
                      'segmentation_small_bowel': large_bowel_segmentation_mask,
                      'segmentation_large_bowel': small_bowel_segmentation_mask, }
            transformed_sample = self.transform(sample)
            image = transformed_sample['image']
            stomach_segmentation_mask = transformed_sample['segmentation_stomach']
            small_bowel_segmentation_mask = transformed_sample['segmentation_small_bowel']
            large_bowel_segmentation_mask = transformed_sample['segmentation_large_bowel']

        image = image[np.newaxis, ...].astype("float32") # add color dimension
        stomach_segmentation_mask = stomach_segmentation_mask[np.newaxis, ...].astype("float32")
        small_bowel_segmentation_mask = small_bowel_segmentation_mask[np.newaxis, ...].astype("float32")
        large_bowel_segmentation_mask = large_bowel_segmentation_mask[np.newaxis, ...].astype("float32")
        
        return image, np.concatenate((stomach_segmentation_mask, small_bowel_segmentation_mask, large_bowel_segmentation_mask), axis=0)

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
    
   

""" Train a given network using training and test data. """
def train(net, train_data, val_data, test_data, criterion, optimizer, batch_size, epochs, checkpoints_name="net", output_selector=lambda out : out):
    # CREDITS for a big portion of the training loop: CS4240 DL assignment 3
    device = next(net.parameters()).device
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size)
    test_loader = DataLoader(test_data, batch_size)
    
    train_losses, val_losses, test_losses = [], [], []

    # initialize early stopping variables
    patience = 10
    val_best_loss = float("inf")
    patience_cnt = 0

    print(f"Start training on device {device}, batch size {batch_size}, {len(train_data)} train samples ({len(train_loader)} batches)")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}", end="", flush=True)
        train_epoch_loss = compute_loss_train(net, train_loader, optimizer, criterion, device, output_selector)
        test_epoch_loss = compute_loss_eval(net, test_loader, criterion, device, output_selector)
        val_epoch_loss = compute_loss_eval(net, val_loader, criterion, device, output_selector)

        # Calculate the average training and validation loss
        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_test_loss = test_epoch_loss / len(test_loader)
        avg_val_loss = val_epoch_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Train loss: {avg_train_loss}, val loss: {avg_val_loss}, test loss: {avg_test_loss}")
        if (epoch + 1) % 10 == 0:
            store_model(net, optimizer, "epoch_" + str(epoch) + "_" + checkpoints_name)
        if avg_val_loss < val_best_loss:
            val_best_loss = avg_val_loss
            patience_cnt = 0
            store_model(net, optimizer, "best_" + checkpoints_name)
        else:
            patience_cnt += 1
            if patience_cnt == patience:
                print(f"Training is stopped after {patience} epochs without improvement on the validation data")
                store_model(net, optimizer, "patience_" + checkpoints_name)
                break

    print("Training done")
    return train_losses, val_losses, test_losses


def compute_loss_train(net, data_loader, optimizer, criterion, device, output_selector):
    net.train()  # Switch network to train mode
    epoch_loss = 0
    print(" 0% [", end="", flush=True)
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.expand(-1, 3, -1, -1)  # TODO adjust networks to take 1 channels instead of 3
        optimizer.zero_grad()  # Set the gradients to zero
        y_pred = output_selector(net(x_batch))  # Perform forward pass
        loss = criterion(y_pred, y_batch)  # Compute the loss
        loss.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update parameters
        epoch_loss += loss.item()  # Discard gradients and store total loss
        if i % (max(len(data_loader) // 20, 1)) == 0:
            print("#", end="")
    print("] 100% ", end="")
    return epoch_loss

def compute_loss_eval(net, data_loader, criterion, device, output_selector):
    epoch_loss = 0
    with torch.no_grad():
        net.eval()  # Switch network to eval mode
        for (x_batch, y_batch) in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.expand(-1, 3, -1, -1)  # TODO adjust networks to take 1 channels instead of 3
            y_pred = output_selector(net(x_batch))  # Perform forward pass
            loss = criterion(y_pred, y_batch)  # Compute the loss
            epoch_loss += loss.item()  # Discard gradients and store total loss
    return epoch_loss

def store_model(net, optimizer, filename):
    model_state = {"model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    torch.save(model_state, f"{filename}.pkl")

def evaluate_segmentation_model(net, threshold, dataset):
    device = next(net.parameters()).device
    net.eval()
    dice_scores = []
    for x_batch, y_batch in DataLoader(dataset, batch_size=16):
        test_predictions = torch.sigmoid(net(x_batch.expand(-1, 3, -1, -1).to(device))["out"]).detach().cpu().numpy()
        y_batch = y_batch.detach().cpu().numpy()
        test_predictions[test_predictions > threshold] = 1.
        test_predictions[test_predictions < 1.] = 0.
        dice_scores.append(1 - dice_loss(test_predictions, y_batch).item())
    return np.average(dice_scores)

####################################################################################################################################
# Loss function from: https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388

""" Calculates DICE loss, averaged over batch size. """
def dice_loss(inputs, targe):
    num = targe.shape[0]
    inputs = inputs.reshape(num, -1)
    targe = targe.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * targe)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + targe.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice

""" Calculates combined BCE and DICE loss, averaged over batch size. """
def bce_dice_loss(inputs, target):
    dicescore = dice_loss(inputs, target)
    bcescore = torch.nn.BCELoss()
    bceloss = bcescore(inputs, target)
    return bceloss + dicescore

####################################################################################################################################


####################################################################################################################################
# CREDITS TO: https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        new_h, new_w = self.output_size

        for key in sample:
            sample[key] = transform.resize(sample[key], (new_h, new_w))

        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __call__(self, sample):
        h, w = sample['image'].shape[:2]

        right = np.random.randint(1, 10)
        left = np.random.randint(0, 10)
        top = np.random.randint(0, 10)
        bottom = np.random.randint(0, 10)
        assert right + left < w
        assert top + bottom < h

        for key in sample:
            new_image = sample[key][top:h-bottom,left:w-right]
            sample[key] = transform.resize(new_image, (h, w))

        return sample

class LabelSmoothing(object):
    """Smooth the segmentation of a slice.

    Args:
        p (float): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, p):
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, sample):
        k = np.asarray(np.random.rand(9) < self.p, dtype=int).reshape(3,3) / 9
        k[1, 1] = 1
        k = np.clip(k, 0, 1)

        for key in sample:
            new_segmentation = convolve2d(sample[key], k, mode="same")
            sample[key] = np.clip(new_segmentation, 0, 1)

        return sample

class Normalize(object):
    """Normalize the input image of a slice.

    Args:
        mean (float): Desired mean of normalized image
        std (float): Desired std of normalized image
    """

    def __init__(self, mean, std):
        assert 0 <= mean <= 1
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] = (sample['image'] - self.mean) / self.std

        return sample

def get_all_cases(data_folder):
    bad = ["case129", "case133",  "case134", "case145", "case148", "case116",
           "case114", "case113", "case110", "case102", "case89", "case85",
           "case78", "case49", "case41", "case131", "case36", "case19",
           "case18", "case16", "case11", "case9", "case6"]
    image_folder = f"{data_folder}/train/"
    out = []
    for filename in os.listdir(image_folder):
        if filename not in bad:
            out.append(filename + "_")
    return out

# https://www.codespeedy.com/how-to-create-a-stopwatch-in-python/
import time
def time_convert(activity, sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time lapsed for {0} = {1}:{2}:{3}".format(activity,int(hours),int(mins),sec))
