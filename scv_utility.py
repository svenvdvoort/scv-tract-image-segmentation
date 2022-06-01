import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from scipy.signal import convolve2d

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
            image = self.transform(image)

        if self.target_transform:
            stomach_segmentation_mask = self.target_transform(stomach_segmentation_mask)
            small_bowel_segmentation_mask = self.target_transform(small_bowel_segmentation_mask)
            large_bowel_segmentation_mask = self.target_transform(large_bowel_segmentation_mask)
            sample = {'image': image, 'segmentation': segmentation_mask}
            transformed_sample = self.transform(sample)
            image, segmentation_mask = transformed_sample['image'], transformed_sample['segmentation']

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
def train(net, train_data, test_data, criterion, optimizer, batch_size, epochs, checkpoints_name="net", output_selector=lambda out : out):
    # CREDITS for a big portion of the training loop: CS4240 DL assignment 3
    device = next(net.parameters()).device
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size)
    print(f"Start training on device {device}, batch size {batch_size}, {len(train_data)} train samples ({len(train_loader)} batches)")
    for epoch in range(epochs):
        train_epoch_loss = 0
        net.train() # Switch network to train mode
        print(f"Epoch {epoch+1} 0% [", end="")
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.expand(-1, 3, -1, -1) # TODO adjust networks to take 1 channels instead of 3
            optimizer.zero_grad()                   # Set the gradients to zero
            y_pred = output_selector(net(x_batch))  # Perform forward pass
            loss = criterion(y_pred, y_batch)       # Compute the loss
            loss.backward()                         # Backward pass to compute gradients
            optimizer.step()                        # Update parameters
            train_epoch_loss += loss.item()         # Discard gradients and store total loss
            if i % (max(len(train_loader) // 20, 1)) == 0:
                print("#", end="")                  # Print progress every 5%
        print("] 100%", end="")
        test_epoch_loss = 0
        with torch.no_grad():
            net.eval() # Switch network to eval mode
            for (x_batch, y_batch) in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch = x_batch.expand(-1, 3, -1, -1)  # TODO adjust networks to take 1 channels instead of 3
                y_pred = output_selector(net(x_batch))   # Perform forward pass
                loss = criterion(y_pred, y_batch)        # Compute the loss
                test_epoch_loss += loss.item()           # Discard gradients and store total loss
        # Calculate the average training and validation loss
        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_test_loss = test_epoch_loss / len(test_loader)
        print(f" Train loss: {avg_train_loss}, test loss: {avg_test_loss}")
        if (epoch + 1) % 10 == 0:
            model_state = {"model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
            torch.save(model_state, f"{checkpoints_name}_checkpoint{epoch+1}.pkl")
    print("Training done")


####################################################################################################################################
# CREDITS TO: https://amaarora.github.io/2020/09/13/unet.html
# Loss function from: https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.sigmoid     = nn.Sigmoid()
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        out      = self.sigmoid(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out

""" Calculates DICE loss, averaged over batch size. """
def dice_loss(inputs, target):
    num = target.size(0)
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
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
        image, segmentation = sample['image'], sample['segmentation']

        new_h, new_w = self.output_size

        new_image = transform.resize(image, (new_h, new_w))
        new_segmentation = transform.resize(segmentation, (new_h, new_w))

        sample['image'], sample['segmentation'] = new_image, new_segmentation
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        h, w = image.shape[:2]

        right = np.random.randint(1, 10)
        left = np.random.randint(0, 10)
        top = np.random.randint(0, 10)
        bottom = np.random.randint(0, 10)
        assert right + left < w
        assert top + bottom < h


        new_image = image[top:h-bottom,left:w-right]
        resized_image = transform.resize(new_image, (h, w))

        if segmentation is None:
            sample['image'], sample['segmentation'] = resized_image, None
            return sample
        else:
            new_segmentation = segmentation[top:h-bottom,left:w-right]
            resized_segmentation = transform.resize(new_segmentation, (h, w))

            sample['image'], sample['segmentation'] = resized_image, resized_segmentation
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
        image, segmentation = sample['image'], sample['segmentation']
        k = np.asarray(np.random.rand(9) < self.p, dtype=int).reshape(3,3) / 9
        k[1, 1] = 1
        k = np.clip(k, 0, 1)

        new_segmentation = convolve2d(segmentation, k, mode="same")
        new_segmentation = np.clip(new_segmentation, 0, 1)

        sample['image'], sample['segmentation'] = image, new_segmentation
        return sample

def get_all_cases(data_folder):
    image_folder = f"{data_folder}/train/"
    out = []
    for filename in os.listdir(image_folder):
        out.append(filename + "_")
    return out
