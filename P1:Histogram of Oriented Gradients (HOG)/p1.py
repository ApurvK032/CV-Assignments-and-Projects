from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

def get_differential_filter():
    filter_x, filter_y = None, None
    # x direction filter
    filter_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    # y direction filter
    filter_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    return filter_x, filter_y


def filter_image(image, filter):
    image_filtered = None
    # image size
    m, n = image.shape
    k = filter.shape[0]  # assume square filter
    pad = k // 2
    
    #Make a new image with padding
    padded = np.zeros((m + 2*pad, n + 2*pad))
    padded[pad:pad+m, pad:pad+n] = image
    
    # empty output
    image_filtered = np.zeros((m, n))
    
    # convolution: slide filter over image
    for i in range(m):
        for j in range(n):
            region = padded[i:i+k, j:j+k]
            value = np.sum(region * filter)
            image_filtered[i, j] = value
    
    return image_filtered


def get_gradient(image_dx, image_dy):
    grad_mag, grad_angle = None, None

    # magnitude#
    grad_mag = np.sqrt(image_dx**2 + image_dy**2)
    
    # angle(radians)
    grad_angle = np.arctan2(image_dy, image_dx)  # range: [-pi, pi]
    
    # making it unsigned: shifting negatives by adding pi
    grad_angle[grad_angle < 0] += np.pi

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    ori_histo = None

    # image size
    m, n = grad_mag.shape
    M = m // cell_size
    N = n // cell_size
    
    ###6unsigned orientation bins over zero to pi[0,pi)
    num_bins = 6
    bin_width = np.pi / num_bins

    # output : M x N x 6
    ori_histo = np.zeros((M, N, num_bins))

    # loop over cells
    for i in range(M):
        for j in range(N):
            # take the cell
            mag_cell = grad_mag[i*cell_size:(i+1)*cell_size,
                                j*cell_size:(j+1)*cell_size]
            ang_cell = grad_angle[i*cell_size:(i+1)*cell_size,
                                  j*cell_size:(j+1)*cell_size]

            # loop over pixels in the cell
            for u in range(cell_size):
                for v in range(cell_size):
                    mag = mag_cell[u, v]
                    ang = ang_cell[u, v]  

                    # which two neighboring bins? (linear interpolation)
                    
                    bin_float = ang / bin_width          # e.g., 2.3 means between bins 2 and 3
                    b0 = int(np.floor(bin_float))        # left bin (0..5)
                    if b0 >= num_bins:
                        b0 = num_bins - 1                # safety for ang==π (shouldn't happen)
                    b1 = (b0 + 1) % num_bins            # right bin 

                    # weights based on distance to bin edges
                    frac = bin_float - b0                
                    w1 = frac                            # weight to right bin
                    w0 = 1.0 - frac                      # weight to left bin

                    #votes
                    ori_histo[i, j, b0] += mag * w0
                    ori_histo[i, j, b1] += mag * w1

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    ori_histo_normalized = None
    
    M, N, B = ori_histo.shape   #B is no. of bins - (6)
    
    # output array
    out_M = M - block_size + 1
    out_N = N - block_size + 1
    out_D = B * block_size * block_size
    
    ori_histo_normalized = np.zeros((out_M, out_N, out_D))
    
    epsilon = 1e-3 #approx. zero
    
    # loop over all possible block positions
    for i in range(out_M):
        for j in range(out_N):

            # getting block (block_size x block_size x bins)
            block = ori_histo[i:i+block_size, j:j+block_size, :]
            
            # flatten into 1D
            vec = block.flatten()
            
            # compute L2 norm
            norm = np.sqrt(np.sum(vec**2) + epsilon**2)
            
            # normalize#
            vec_normalized = vec / norm
            
            # store result
            ori_histo_normalized[i, j, :] = vec_normalized
    
    return ori_histo_normalized


def extract_hog(image, cell_size=8, block_size=2):
    # convert grey-scale image to double format
    image = image.astype('float') / 255.0
    hog = None
    
    # get filters
    filter_x, filter_y = get_differential_filter()
    
    # filter image in x and y
    image_dx = filter_image(image, filter_x)
    image_dy = filter_image(image, filter_y)
    
    # compute gradient magnitude and angle
    grad_mag, grad_angle = get_gradient(image_dx, image_dy)
    
    # build histogram per cell
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    
    # block normalization
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size)
    
    # flatten to 1D vector
    hog = ori_histo_normalized.flatten()
    
    return hog

def face_detection(I_target, I_template):
    bounding_boxes = None
    
    
    # get template hog
    hog_template = extract_hog(I_template)
    
    # template size
    h, w = I_template.shape
    
    # slide window
    boxes = []
    for y in range(0, I_target.shape[0] - h, 3):
        for x in range(0, I_target.shape[1] - w, 3):
            patch = I_target[y:y+h, x:x+w]
            
            # compute hog of patch
            hog_patch = extract_hog(patch)
            
            # normalize descriptors
            a = hog_patch - np.mean(hog_patch)
            b = hog_template - np.mean(hog_template)
            
            # compute NCC
            score = np.sum(a*b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)) + 1e-5)
            
            # threshold
            if score > 0.5:
                boxes.append([x, y, score])
    
    ##convert to numpy
    bounding_boxes = np.array(boxes)
    
    # Non-Maximum Suppression
    final_boxes = []
    if bounding_boxes.shape[0] > 0:
        scores = bounding_boxes[:, 2]

        # sort by score
        idxs = scores.argsort()[::-1]  
        
        while len(idxs) > 0:
            i = idxs[0]
            final_boxes.append(bounding_boxes[i])
            keep = []
            for j in idxs[1:]:

                # IoU calculation
                xx1 = max(bounding_boxes[i,0], bounding_boxes[j,0])
                yy1 = max(bounding_boxes[i,1], bounding_boxes[j,1])
                xx2 = min(bounding_boxes[i,0]+w, bounding_boxes[j,0]+w)
                yy2 = min(bounding_boxes[i,1]+h, bounding_boxes[j,1]+h)
                
                inter_w = max(0, xx2-xx1)
                inter_h = max(0, yy2-yy1)
                inter_area = inter_w * inter_h
                
                area_i = w*h
                area_j = w*h
                iou = inter_area / float(area_i + area_j - inter_area)
                
                if iou < 0.5:
                    keep.append(j)
            idxs = np.array(keep)
    
    if len(final_boxes) == 0:
        return np.zeros((0,3))
    
    bounding_boxes = np.array(final_boxes) 

    return  bounding_boxes


# def face_detection_bonus(I_target, I_template):
#     bounding_boxes = None
#     # To do
# #   return  bounding_boxes


# ----- Visualization Functions -----
def visualize_hog(image, hog, cell_size=8, block_size=2, num_bins=6):
    image = image.astype('float') / 255.0
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = image.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def visualize_face_detection(I_target, bounding_boxes, box_size):

    I_target = I_target.convert("RGB")
    ww, hh = I_target.size

    draw = ImageDraw.Draw(I_target)

    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size[1]
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size[0]

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)
        bbox_text = f'{bounding_boxes[ii, 2]:.2f}'
        draw.text((x1 + 1, y1 + 2), bbox_text, fill=(0, 255, 0))

    plt.imshow(np.array(I_target), vmin=0, vmax=1)
    plt.axis("off")
    plt.show()
# ----- Visualization Functions -----


if __name__=='__main__':

    # ----- HOG -----
    image = Image.open('cameraman.tif')
    image_array = np.array(image)
    hog = extract_hog(image_array)
    visualize_hog(image_array, hog, 8, 2)

    # ----- Face Detection -----
    I_target = Image.open('target.png')
    I_target_array = np.array(I_target.convert('L'))

    I_template = Image.open('template.png')
    I_template_array = np.array(I_template.convert('L'))

    bounding_boxes=face_detection(I_target_array, I_template_array)
    visualize_face_detection(I_target, bounding_boxes, I_template_array.shape)