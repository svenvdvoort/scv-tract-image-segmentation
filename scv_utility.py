import cv2
import os

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
    image = cv2.imread(image_folder + image_filename, cv2.IMREAD_GRAYSCALE)
    assert image.shape[0] == int(height) and image.shape[1] == int(width), f"{image_filename} Image width or height does not match resolution from filename"
    return image, (int(height), int(width)), float(pixel_width)
