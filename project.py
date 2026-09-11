import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def generate_segmentation_map(X):

    def lapla_blur(img):
            kernel_size = 1    
            img = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel_size)
            return np.absolute(img)
    x_grad = np.array([lapla_blur(x) for x in X])
    height, width = X[0].shape
    max_index = np.zeros((height, width), dtype=np.uint8)
    max_grad = np.zeros((height, width), dtype=np.uint8)
    value_max_grad = np.zeros((height, width), dtype=np.uint8)

    # generate the in-focused map
    for y in range(height):
        for x in range(width):
            pixel_values = [image[y, x] for image in x_grad]
            max_index = np.argmax(pixel_values)
            max_grad[y,x] = np.max(pixel_values)
            value_max_grad[y,x] = X[max_index][y][x]
    blur_value_max_grad = cv2.medianBlur(value_max_grad,3)
    # generate an approximate segmentation which contains many noisy patches
    _, otsu_thresh_value_inv = cv2.threshold(blur_value_max_grad, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masked_image = cv2.bitwise_and(value_max_grad, otsu_thresh_value_inv)
    _, masked_image_threshold = cv2.threshold(masked_image, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    highlighted_image = np.zeros_like(masked_image_threshold)
    for y in range(masked_image_threshold.shape[0]):
        for x in range(masked_image_threshold.shape[1]):
            if masked_image_threshold[y, x] != otsu_thresh_value_inv[y, x]:
                highlighted_image[y, x] = 255 
    kernel_size = 41
    sigma = 150

    # pixels -> density
    mask_blur = cv2.GaussianBlur(highlighted_image, (kernel_size, kernel_size), sigma)
    mask_blur = cv2.bitwise_and(mask_blur,mask_blur,mask = otsu_thresh_value_inv)
    _, object_mask = cv2.threshold(mask_blur, 3, 255, cv2.THRESH_BINARY)

    # close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    object_mask = cv2.morphologyEx(object_mask,cv2.MORPH_CLOSE,kernel,iterations=1)



    # generate the segmentation for individual image
    predictions = []
    for img in tqdm(X, desc="Processing", total=len(X)):
        
        coords = np.argwhere(object_mask == 255)


        # generate the gradient map and threshold based on intensity
        kernel_size = 1  
        img_grad = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel_size)
        img_grad = np.absolute(img_grad)
        img_grad = img_grad.astype(np.uint8)
        mask = np.zeros_like(img_grad)
        coords_array = np.array(coords)
        indices = np.where(img_grad[coords_array[:, 0], coords_array[:, 1]] >= 20)
        mask[coords_array[indices][:, 0], coords_array[indices][:, 1]] = 255



        # GaussianBlur and threshold based on density of the filtered pixels
        kernel_size = 19
        sigma = 15
        mask_blur = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
        mask_blur = cv2.bitwise_and(mask_blur,mask_blur,mask = object_mask)
        _, mask_blur_thresh = cv2.threshold(mask_blur, 10, 255, cv2.THRESH_BINARY)
        predictions.append(mask_blur_thresh)
        
    return np.array(predictions, dtype = np.uint8)



def main():
    
    p1 = '2_image.tiff'
    p2 = '3_image.tiff'
    p3 = '6_image.tiff'
    p4 = '7_image.tiff'
    l1 = '2_mask.tiff'
    l2 = '3_mask.tiff'
    l3 = '6_mask.tiff'
    l4 = '7_mask.tiff'
    # Load the TIF stack
    x1 = imageio.volread(p1)
    x2 = imageio.volread(p2)
    x3 = imageio.volread(p3)
    x4 = imageio.volread(p4)
    y1 = imageio.volread(l1)
    y2 = imageio.volread(l2)
    y3 = imageio.volread(l3)
    y4 = imageio.volread(l4)
    for index, (X,Y) in enumerate(zip([x1,x2,x3,x4],[y1,y2,y3,y4])):
        
        predictions = generate_segmentation_map(X)
        predictions_flat = np.array(predictions).flatten()/255
        label_flat = Y.flatten()/255
        f1 = f1_score(label_flat, predictions_flat)
        accuracy = accuracy_score(label_flat, predictions_flat)
        print("F1 Score:", f1, "Accuracy:", accuracy)

    
    return

if __name__ == "__main__":
    main()