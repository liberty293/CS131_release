import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    out = image[start_row:start_row+num_rows, start_col:start_col+num_cols, :]

    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = (image)**2*.5
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    scale_r = input_rows/output_rows #i decrease by this
    scale_c = input_cols/output_cols
    for i in range(output_rows):
        for j in range(output_cols):          
            output_image[i,j,:] = input_image[int(i*scale_r), int(j*scale_c),:]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])

    return  R@point
    ### END YOUR CODE


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3
    center = np.array([int(input_rows/2),int(input_cols/2)])
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)
    
    ## YOUR CODE HERE
    for i in range(input_rows):
        for j in range(input_cols):     
            T = np.array([[1, 0, -center[0]],
                          [0,1,-center[1]],
                          [0,0,1]])
            new_index = np.linalg.inv(T)@R@T@np.array([i,j,1]).T
            
            if 0 <= new_index[0] <= input_rows and 0 <= new_index[1] <= input_cols:
                
                output_image[i,j,:] = input_image[int(new_index[0]), int(new_index[1]),:]
            else:
                pass
            
    ### END YOUR CODE

    # 3. Return the output image
    return output_image
