from PIL import Image
import numpy as np

# Load the grayscale image. Replace 'path_to_image.jpg' with the actual file path
image_path = 'pneumonia_heatmap.jpg'
gray_image = Image.open(image_path)

# Resize the image to 1024x1024
resized_gray_image = gray_image.resize((600, 600))

# Convert the grayscale image to 3-channel (RGB) format
# rgb_image = np.stack([resized_gray_image]*3, axis=-1)

# Convert the numpy array back to an image
three_channel_image = Image.fromarray(resized_gray_image)

# Optionally, save the new image
three_channel_image.save(image_path)
