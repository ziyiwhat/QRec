import numpy as np
import cv2
from PIL import Image

from sklearn.cluster import DBSCAN
from scipy.optimize import linprog

def merge_points(points, max_distance=20):
    clustering = DBSCAN(eps=max_distance, min_samples=1).fit(points)
    labels = clustering.labels_
    merged_points = []
    for label in np.unique(labels):
        group_points = points[labels == label]
        center_point = np.mean(group_points, axis=0)
        merged_points.append(center_point)
    return np.array(merged_points)

def warp_and_prod(width, height, homo_ls):
    mask_ls = []
    for i in range(len(homo_ls)):
       ones = np.ones((height, width))
       mask_ = cv2.warpPerspective(ones, homo_ls[i], (width, height))
       mask_ls.append(mask_)
    return np.prod(mask_ls, axis=0)

def compute_equations(vertices):
    """
    Computes the general line equations Ax + By + C = 0 for the edges of a convex polygon.
    
    Parameters:
        vertices (np.ndarray): A (N, 2) array of polygon vertices, assumed to be sorted.
        
    Returns:
        np.ndarray: A (N, 3) array of coefficients [A, B, C] for each edge.
    """
    num_vertices = len(vertices)
    coefficients = []
    
    for i in range(num_vertices):
        # Get the current and next vertex (wrap around at the end)
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % num_vertices]
        
        # Compute coefficients of the general line equation
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        coefficients.append([A, B, C])
    
    return np.array(coefficients)


def sort_polygon_vertices(vertices):
    """
    Sorts the vertices of a convex polygon in counterclockwise order.
    
    Parameters:
        vertices (np.ndarray): A (N, 2) array of polygon vertices.
        
    Returns:
        np.ndarray: The sorted vertices in counterclockwise order.
    """
    # Compute the centroid of the vertices
    centroid = np.mean(vertices, axis=0)
    
    # Compute angles from the centroid to each vertex
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    
    # Sort vertices based on the angles
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]
    
    return sorted_vertices

def find_max_inscribed_rectangle(vertices, ratio):
    """
    Find the largest axis-aligned rectangle inscribed in a convex polygon.
    
    Parameters:
        vertices (np.ndarray): A (N, 2) array of polygon vertices, sorted counterclockwise.
        ratio (float): The desired aspect ratio (width / height) of the rectangle.
    
    Returns:
        list: The coordinates of the largest rectangle in the form [x1, y1, x2, y2].
    """
    # Compute general line equations for polygon edges
    line_equations = compute_equations(vertices)
    
    # Define optimization variables: x1, y1, x2, y2
    # We maximize x2 - x1
    c = [-1, 0, 1, 0]  # Coefficients for the objective function: maximize x2 - x1
    
    # Inequality constraints: Ax + By + C >= 0 for all rectangle vertices
    A_ineq = []
    b_ineq = []
    
    for A, B, C in line_equations:
        # Four vertices of the rectangle: (x1, y1), (x1, y2), (x2, y1), (x2, y2)
        A_ineq.append([A, B, 0, 0])  # A*x1 + B*y1 + C >= 0
        A_ineq.append([A, 0, 0, B])  # A*x1 + B*y2 + C >= 0
        A_ineq.append([0, B, A, 0])  # A*x2 + B*y1 + C >= 0
        A_ineq.append([0, 0, A, B])  # A*x2 + B*y2 + C >= 0
        
        b_ineq.extend([-C] * 4)  # Convert to Ax + By + C <= 0 by flipping sign
    
    # Equality constraint: (x2 - x1) - ratio * (y2 - y1) = 0
    A_eq = [[-1, ratio, 1, -ratio]]
    b_eq = [0]
    
    # Bounds for the variables: x1, y1, x2, y2 must be within the polygon's bounding box
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)
    bounds = [(x_min, x_max), (y_min, y_max), (x_min, x_max), (y_min, y_max)]
    
    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        x1, y1, x2, y2 = result.x
        return np.array([x1, y1, x2, y2])
    else:
        raise ValueError("Linear programming failed to find a solution.")

def opt(mask, width, height):
    if mask.max() == 1:
        mask *= 255.
    mask = np.float32(mask)
    ratio = height / width
    dst = cv2.cornerHarris(mask,4,3,0.04)
    points = np.argwhere(dst>0.01*dst.max())
    points = merge_points(points)
    points = sort_polygon_vertices(points)
    rect = find_max_inscribed_rectangle(points, ratio)
    return rect

if __name__ == '__main__':
    padding = 20
    height, width = 960 * 2, 1080 * 2
    theta = np.radians(15)
    H1 = np.load('./homo/0_homo.npy')
    H2 = np.load('./homo/1_homo.npy')
    H3 = np.load('./homo/2_homo.npy')

    print(H1.shape)

    mask = warp_and_prod(width, height, [H1, H2, H3])
    mask = np.pad(mask, ((padding, padding),(padding, padding)))
    rect = opt(mask, width=width, height=height)
    mask = mask[padding:height+padding, padding:width+padding]
    rect -= padding
    rect = rect.astype(int)
    print(f"rect :{rect}")

    # Calculate the area of the rectangle
    rect_area = abs((rect[3]-rect[1]) * (rect[2]-rect[0]))
    original_area = width * height
    area_ratio = rect_area / original_area
    print(f"Area of the rectangle: {rect_area}")
    print(f"Area of the original image: {original_area}")
    print(f"Area ratio: {area_ratio:.4f}")

    # Convert mask to color
    mask_color = cv2.cvtColor((mask).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw rectangle
    start_point = (rect[1], rect[0])
    end_point = (rect[3], rect[2])
    color = (0, 255, 0)  # Green color
    thickness = 2
    mask_color = cv2.rectangle(mask_color, start_point, end_point, color, thickness)

    # Highlight corner points
    corners = [
        (rect[1], rect[0]),  # Top-left
        (rect[3], rect[0]),  # Top-right
        (rect[1], rect[2]),  # Bottom-left
        (rect[3], rect[2])  # Bottom-right
    ]
    corner_colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (0, 0, 255)]  # Different colors

    for corner, color in zip(corners, corner_colors):
        mask_color = cv2.circle(mask_color, corner, radius=5, color=color, thickness=-1)

    # Save the image with the rectangle and corners
    im = Image.fromarray(mask_color)
    im.save("warped_with_rect_and_corners.png")
