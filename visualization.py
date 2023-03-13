import torch
from torchviz import make_dot
import ML_model as ml
import torch.onnx
import os
import cv2 as cv
import numpy as np

# Load the three images
image1 = cv.imread('/Users/avrahamhrinevitzky/Downloads/model.onnx-2-4.png')
image2 = cv.imread('/Users/avrahamhrinevitzky/Downloads/model.onnx-2-2.png')
image3 = cv.imread('/Users/avrahamhrinevitzky/Downloads/model.onnx copy-2.png')

# Resize the images if needed
image1 = cv.resize(image1, (2179, 446))
image2 = cv.resize(image2, (2179, 446))
image3 = cv.resize(image3, (2179, 446))

# Stack the images vertically
stacked_image = np.vstack((image1, image2, image3))

# Display the stacked image
cv.imshow('Stacked Image', stacked_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the stacked image to a file
cv.imwrite('stacked_image.png', stacked_image)