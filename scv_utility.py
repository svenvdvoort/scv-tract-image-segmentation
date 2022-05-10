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


""" Converts run-length encoding to x and y coordinates. Useful together with plt.fill: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.fill.html """
def rle_to_xy(rle, width, height):
    x, y = [], []

    for i in range(0,len(rle),2):
        x.append(rle[i] % width)
        x.append(rle[i] % width + rle[i+1])
        y.append(rle[i] // height)
        y.append(rle[i] // height)
        
    return x, y

""" Extracts rle, given id (e.g. "case123_day20_slice_0067") and organ (a.k.a. 'class' but class is a python keyword) """
def extract_rle(data, id, organ):
    x = data[data['id'] == id]
    x = x[x['class']  == organ]
    rle = x["segmentation"]

    rle = rle.values[0] # Extract the run-length encoding
    rle = rle.split(' ') # Make a list from it
    
    if rle[0] == '':
        return []
    
    rle = list(map(int, rle)) # Map elements to integers
    
    return rle
    