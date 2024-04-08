from typing import Tuple

import numpy as np
x = np.eye(4)


def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
    T = np.eye(4)
    theta = (45+90)* 2* np.pi /360 #amount it is rotated on y axis 
    #print(np.cos(theta))

    T = np.array([[np.cos(theta), 0.0, np.sin(theta), 0],
         [0.0,1.0,0.0,0],
         [-np.sin(theta), 0.0, np.cos(theta), d],
         [0,0,0,1]])
    # END YOUR CODE
    assert T.shape == (4, 4)
    return T

def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)

    row_of_ones = np.ones((1, N))

# Append the row of 1s to the original array
    homo_array = np.vstack((points, row_of_ones))
    apply = np.dot(T,homo_array)
    points_transformed = apply[:-1, :]

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray: the intersection of the two lines definied by (a0, a1)
                    and (b0, b1).
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == float

    # Intersection point between lines
    out = np.zeros(2)
    yout = False
    xout = False
    if(a_1[0] - a_0[0] == 0): #slope is infinity
        out[0] = a_1[0]
        xout = True
    elif (a_1[1] - a_0[1] == 0): #slope is 0
        a_slope = 0
        out[1] = a_1[1]
        yout = True
    else: 
        a_slope = (a_1[1]-a_0[1])/(a_1[0]-a_0[0])

    if(b_1[0] - b_0[0] == 0):
        out[0] = b_1[0]
        out[1] = a_slope*out[0] - a_slope*a_1[0]+a_1[1]
        xout = True
        yout = True
    elif (b_1[1] - b_0[1] == 0): #slope is 0
        out[1] = b_1[1]
        yout = True 
    else:       
        b_slope = (b_1[1]-b_0[1])/(b_1[0]-b_0[0])

    if(yout and xout):
        return out
    
    if(yout):
        out[0] = (a_slope*a_1[0]-a_1[1]-b_slope*b_1[0]+b_1[1])/(a_slope-b_slope)
        return out
    
    if(xout):
        out[1] = b_slope*out[0] - b_slope*b_1[0] + b_1[1]
        return out

    x_out = (a_slope*a_1[0]-a_1[1]-b_slope*b_1[0]+b_1[1])/(a_slope-b_slope)
    y_out = a_slope*x_out-a_slope*a_1[0]+a_1[1]
    pass
    # END YOUR CODE
    out[0] = x_out
    out[1] = y_out
    assert out.shape == (2,)
    assert out.dtype == float

    return out


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)
    #set the vanishing points that are opposite of the triangle sides
    a_perp_pt = np.zeros(2)
    b_perp_pt = np.zeros(2)
    #find the slope
    #if the slope is infinity, the optical center x is just the value of the other point
    if(v1[0]==v0[0]):
        a_slope = np.inf
    else:
        a_slope = (v1[1]-v0[1])/(v1[0]-v0[0]) 
    a_perp_pt = v2    
    if(v1[0]==v2[0]):
        b_slope = np.inf
    else:
        b_slope = (v2[1]-v1[1])/(v2[0]-v1[0])
    b_perp_pt = v0

    #get the perpendicular slope
    if(a_slope != 0):
        a_perp = -1/a_slope
    if(b_slope != 0):
        b_perp = -1/b_slope

    a0= a1= b0= b1 = np.zeros(2)
    a0 = a_perp_pt
    if(a_slope == 0):
        a1 = a_perp_pt
        a1[1] += 1
    else:
        a1 = a0 + np.array([1, a_perp])
    
    b0 = b_perp_pt
    if(b_slope == 0):
        b1 = b_perp_pt
        b1[1] += 1
    else:
        b1 = b0 + np.array([1, b_perp])
    print(a0, a1, b0, b1)
    optical_center = intersection_from_lines(a0,a1,b0,b1)
    # END YOUR CODE

    assert optical_center.shape == (2,)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    xs = (v0[0]-optical_center[0])*(v1[0]-optical_center[0])
    ys = (v0[1]-optical_center[1])*(v1[1]-optical_center[1])
  #  print(xs, ys)
    f2 = xs + ys
   # print(f2)
    f = np.sqrt(-1*f2)

    # YOUR CODE HERE
    pass
    # END YOUR CODE

    return float(f)


def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    f_mm = f*sensor_diagonal_mm/image_diagonal_pixels

    # YOUR CODE HERE
    pass
    # END YOUR CODE

    return f_mm
