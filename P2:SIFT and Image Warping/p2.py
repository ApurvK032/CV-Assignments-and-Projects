from PIL import Image
import numpy as np
from cv2 import resize
import matplotlib.pyplot as plt

from cv2 import SIFT_create, KeyPoint_convert, filter2D
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

def find_match(img1, img2):
    dis_thr = 0.7
    sift = SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    nn1 = NearestNeighbors(n_neighbors=2).fit(des2)
    d12, idx12 = nn1.kneighbors(des1)

    nn2 = NearestNeighbors(n_neighbors=2).fit(des1)
    d21, idx21 = nn2.kneighbors(des2)

    forward_mask = d12[:, 0] < dis_thr * d12[:, 1]
    backward_mask = d21[:, 0] < dis_thr * d21[:, 1]

    forward_map = {i: idx12[i, 0] for i in np.where(forward_mask)[0]}
    backward_map = {j: idx21[j, 0] for j in np.where(backward_mask)[0]}

    mutual_matches = [(i, j) for i, j in forward_map.items() if j in backward_map and backward_map[j] == i]

    x1 = np.array([kp1[i].pt for i, _ in mutual_matches], dtype=np.float64)
    x2 = np.array([kp2[j].pt for _, j in mutual_matches], dtype=np.float64)

    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    A = None

    if x1 is None or x2 is None or x1.shape[0] < 3 or x2.shape[0] < 3:
        return np.eye(3, dtype=np.float64)

    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)

    def fit_affine(p1, p2):
        n = p1.shape[0]
        X = np.zeros((2 * n, 6), dtype=np.float64)
        b = np.zeros((2 * n,), dtype=np.float64)
        X[0::2, 0] = p1[:, 0]
        X[0::2, 1] = p1[:, 1]
        X[0::2, 2] = 1.0
        X[1::2, 3] = p1[:, 0]
        X[1::2, 4] = p1[:, 1]
        X[1::2, 5] = 1.0
        b[0::2] = p2[:, 0]
        b[1::2] = p2[:, 1]
        params, _, _, _ = np.linalg.lstsq(X, b, rcond=None)
        A2x3 = np.array([[params[0], params[1], params[2]],
                         [params[3], params[4], params[5]]], dtype=np.float64)
        A3x3 = np.vstack([A2x3, np.array([0.0, 0.0, 1.0], dtype=np.float64)])
        return A3x3

    best_A = np.eye(3, dtype=np.float64)
    best_inliers = None
    num_pts = x1.shape[0]

    for _ in range(int(ransac_iter)):
        idx = np.random.choice(num_pts, 3, replace=False)
        A_try = fit_affine(x1[idx], x2[idx])
        x1_h = np.hstack([x1, np.ones((num_pts, 1), dtype=np.float64)])
        x2_hat = (x1_h @ A_try.T)[:, :2]
        err = np.sum((x2_hat - x2) ** 2, axis=1)
        inliers = err < ransac_thr
        if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
            best_inliers = inliers
            best_A = A_try

    if best_inliers is not None and np.count_nonzero(best_inliers) >= 3:
        A = fit_affine(x1[best_inliers], x2[best_inliers])
    else:
        A = best_A

    return A


def warp_image(img, A, output_size):
    h, w = map(int, output_size)
    yy, xx = np.indices((h, w), dtype=np.float64)
    homog_coords = np.vstack((xx.ravel(), yy.ravel(), np.ones(xx.size)))
    mapped = A @ homog_coords
    x_mapped, y_mapped = mapped[0], mapped[1]
    src_h, src_w = img.shape
    grid_y = np.arange(src_h, dtype=np.float64)
    grid_x = np.arange(src_w, dtype=np.float64)
    sample_pts = np.column_stack((y_mapped, x_mapped))
    img_warped = interpolate.interpn((grid_y, grid_x), img.astype(np.float64),
                                     sample_pts, method='linear',
                                     bounds_error=False, fill_value=0.0).reshape(h, w)
    return img_warped


def align_image(template, target, A):
    A_refined = None
    errors = None

    tpl = template.astype(np.float64) / 255.0
    tgt = target.astype(np.float64) / 255.0
    H, W = tpl.shape

    kx = np.array([[-1, 0, 1]], dtype=np.float64) / 2.0
    ky = np.array([[-1], [0], [1]], dtype=np.float64) / 2.0

    Ix = filter2D(tpl.astype(np.float32), -1, kx).astype(np.float64)
    Iy = filter2D(tpl.astype(np.float32), -1, ky).astype(np.float64)

    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)

    SDI = np.stack([Ix * X, Ix * Y, Ix, Iy * X, Iy * Y, Iy], axis=-1).reshape(-1, 6)
    Hs = SDI.T @ SDI + 1e-6 * np.eye(6, dtype=np.float64)
    Hs_inv = np.linalg.inv(Hs)

    A_cur = A.astype(np.float64).copy()
    errs = []

    for _ in range(100):
        Iw = warp_image(tgt, A_cur, tpl.shape)           
        e = (Iw - tpl).reshape(-1)                          
        errs.append(np.linalg.norm(e))

        F = SDI.T @ e                                      
        dp = Hs_inv @ F                                    

        if not np.all(np.isfinite(dp)) or np.linalg.norm(dp) < 1e-4:
            break

        dA = np.array([[1.0 + dp[0], dp[1], dp[2]],
                       [dp[3], 1.0 + dp[4], dp[5]],
                       [0.0, 0.0, 1.0]], dtype=np.float64)

        A_cur = A_cur @ np.linalg.inv(dA)

    A_refined = A_cur
    errors = np.array(errs, dtype=np.float64)
    return A_refined, errors

def track_multi_frames(template, img_list):
    A_list = []
    errors_list = []

    tpl = template.astype(np.uint8)
    ransac_thr = 25.0
    ransac_iter = 1000

    for i, img in enumerate(img_list):
        x1, x2 = find_match(tpl, img)

        if x1.shape[0] >= 3:
            A_init = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
        else:
            A_init = np.eye(3, dtype=np.float64)

        A_refined, errs = align_image(tpl, img, A_init)

        A_list.append(A_refined.copy())
        errors_list.append(np.asarray(errs, dtype=np.float64))

        tpl = warp_image(img, A_refined, tpl.shape)
        tpl = np.clip(tpl, 0, 255).astype(np.uint8)

    return A_list, errors_list

# ----- Visualization Functions -----
def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    plt.show()


def visualize_align_image(template, target, A, A_refined, errors=None):
    import cv2
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list, errors_list=None):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()

    if errors_list is not None:
        for i, errors in enumerate(errors_list):
            plt.plot(errors * 255)
            plt.title(f'Frame {i}')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.show()
# ----- Visualization Functions -----

if __name__=='__main__':

    template = Image.open('template.jpg')
    template = np.array(template.convert('L'))
    
    target_list = []
    for i in range(4):
        target = Image.open(f'target{i+1}.jpg')
        target = np.array(target.convert('L'))
        target_list.append(target)
    
    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    # To do
    ransac_thr = 25.0
    ransac_iter = 1000

    # ----------
    
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[1], A)
    visualize_align_image(template, target_list[1], A, A_refined, errors)

    A_list, errors_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list, errors_list)