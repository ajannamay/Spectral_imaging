import numpy as np
from skimage import io, transform


# Define the similarity transformation matrix
matrix_similarity = np.array(
    [[ 1.08203125,  -0.046875,  -3.0],
     [0.046875,  1.08203125,  -82.0],
     [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
similarity_transform = transform.SimilarityTransform(matrix=matrix_similarity)

# Define the preprocessing steps
yeaz_preprocesses = [
    lambda x: transform.rotate(x, 90),
    #lambda x: transform.rescale(x, 0.25, anti_aliasing=True),
    lambda x: transform.warp(x, similarity_transform),
    lambda x: (x - x.min()) / (x.max() - x.min())
]

# Load the image
image = io.imread('path_to_your_image.jpg')

# Apply the preprocessing steps sequentially
processed_image = image
for preprocess in yeaz_preprocesses:
    processed_image = preprocess(processed_image)