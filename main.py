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
    
    rect = rect.astype(int)
    mask[np.minimum(rect[0], rect[0]+rect[2]):np.maximum(rect[0], rect[0]+rect[2]), 
         np.minimum(rect[1], rect[1]+rect[3]):np.maximum(rect[1], rect[1]+rect[3])]=0.5
    im = Image.fromarray(mask * 255).convert('L')
    im.save("warped.png")
