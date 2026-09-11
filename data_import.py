# Replace 'path/to/your/image.tif' with the actual path to your TIF image

import imageio

# Replace 'path/to/your/multi_page_image.tif' with the actual path to your multi-page TIF file
image_path = '2_image_and_mask.tiff'

# Load the TIF stack
tif_stack = imageio.volread(image_path)

# Now you can work with the loaded stack (e.g., display or process it)
# ...

# Access stack properties if needed
num_frames, height, width = tif_stack.shape
print(f"Number of frames: {num_frames}, Image dimensions: {width} x {height}")