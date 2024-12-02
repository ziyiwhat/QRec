import numpy as np
import cv2
from PIL import Image

def warp_and_prod(width, height, homo_ls):
    ones = np.ones((width, height))
    mask_ls = []
    for i in range(len(homo_ls)):
       mask_ = cv2.warpPerspective(ones, homo_ls[i], (width, height))
       mask_ls.append(mask_)
    return np.prod(mask_ls, axis=0)

def grow(mask, width, height):
    rect = []
    ratio = height / width
    mask_ = np.uint8(mask)
    edge = cv2.Canny(mask_, 0.5, 0.5)
    idx = np.argwhere(edge!=0)
    np.random.shuffle(idx)
    area_width = idx[:,1].max() - idx[:,1].min()
    # print('Max cul:', len(idx), '*', area_width)
    for h_ in range(1,area_width,1):
        flag = 0
        w_ = int(h_ * ratio)
        for x, y in idx:
            if (x + w_ < height) and (y + h_ < width):
                mask_rec = np.zeros_like(mask)            
                mask_rec[x:x+w_, y:y+h_] = 1
                judger = np.zeros_like(mask)
                judger[(mask_rec == 1) & (mask == 0)] = 1
                if judger.sum() == 0:
                    rect = [x, y, w_, h_]
                    flag = 1
                    # print('rect update:', rect)
                    break
            if (x - w_ >= 0) and (y + h_ < width):
                mask_rec = np.zeros_like(mask)            
                mask_rec[x-w_:x, y:y+h_] = 1
                judger = np.zeros_like(mask)
                judger[(mask_rec == 1) & (mask == 0)] = 1
                if judger.sum() == 0:
                    rect = [x, y, -w_, h_]
                    flag = 1
                    # print('rect update:', rect)
                    break
            if  (x + w_ < height) and (y - h_ >= 0):
                mask_rec = np.zeros_like(mask)            
                mask_rec[x:x+w_, y-h_:y] = 1
                judger = np.zeros_like(mask)
                judger[(mask_rec == 1) & (mask == 0)] = 1
                if judger.sum() == 0:
                    rect = [x, y, w_, -h_]
                    flag = 1
                    # print('rect update:', rect)
                    break
            if (x - w_ >= 0) and (y - h_ >= 0):
                mask_rec = np.zeros_like(mask)            
                mask_rec[x-w_:x, y-h_:y] = 1
                judger = np.zeros_like(mask)
                judger[(mask_rec == 1) & (mask == 0)] = 1
                if judger.sum() == 0:
                    rect = [x, y, -w_, -h_]
                    flag = 1
                    # print('rect update:', rect)
                    break
        if not flag:
            break
    return np.array(rect)

if __name__ == '__main__':
    height, width = 384, 512
    theta = np.radians(15)
    H1 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    theta = np.radians(-15)
    H2 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    H3 = np.array([
        [1, 0.3, 0],
        [0.2, 1, 0],
        [0, 0, 1]
    ])

    mask = warp_and_prod(width, height, [H1, H2, H3])
    rect = grow(mask, width=width, height=height)
    print(f"rect :{rect}")

    rect = rect.astype(int)

    # Calculate the area of the rectangle
    rect_area = abs(rect[2] * rect[3])
    original_area = width * height
    area_ratio = rect_area / original_area
    print(f"Area of the rectangle: {rect_area}")
    print(f"Area of the original image: {original_area}")
    print(f"Area ratio: {area_ratio:.4f}")

    # Convert mask to color
    mask_color = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw rectangle
    start_point = (rect[1], rect[0])
    end_point = (rect[1] + rect[3], rect[0] + rect[2])
    color = (0, 255, 0)  # Green color
    thickness = 2
    mask_color = cv2.rectangle(mask_color, start_point, end_point, color, thickness)

    # Highlight corner points
    corners = [
        (rect[1], rect[0]),  # Top-left
        (rect[1] + rect[3], rect[0]),  # Top-right
        (rect[1], rect[0] + rect[2]),  # Bottom-left
        (rect[1] + rect[3], rect[0] + rect[2])  # Bottom-right
    ]
    corner_colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (0, 0, 255)]  # Different colors

    for corner, color in zip(corners, corner_colors):
        mask_color = cv2.circle(mask_color, corner, radius=5, color=color, thickness=-1)

    # Save the image with the rectangle and corners
    im = Image.fromarray(mask_color)
    im.save("warped_with_rect_and_corners.png")
