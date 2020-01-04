import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndi

def generate_target(img, pt, sigma, label_type='Gaussian'):
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def random_shift(img, max_pixel):
    """
    img = (h, w), ndarray
    lr:  >0 right; <0 left [-mp, mp]
    ud:  >0 up;    <0 down [-mp, mp]
    """

    lr = np.random.randint(-max_pixel,max_pixel) # 5
    ud = np.random.randint(-max_pixel,max_pixel) # 6
    # lr=20
    # ud=20
    h, w = img.shape[:2]
    padding_img = np.zeros((h+np.abs(ud), w+np.abs(lr)))
    if ud>=0 and lr>=0:
        padding_img[ud:,lr:] = img
    elif ud>=0 and lr<0:
        padding_img[ud:,:lr] = img
    elif ud<0 and lr<0:
        padding_img[:ud,:lr] = img
    else:
        padding_img[:ud,lr:] = img

    return padding_img[:h,:w]



# COLORMAP_AUTUMN = 0,
# COLORMAP_BONE = 1,
# COLORMAP_JET = 2,
# COLORMAP_WINTER = 3,
# COLORMAP_RAINBOW = 4,
# COLORMAP_OCEAN = 5,
# COLORMAP_SUMMER = 6,
# COLORMAP_SPRING = 7,
# COLORMAP_COOL = 8,
# COLORMAP_HSV = 9,
# COLORMAP_PINK = 10,
# COLORMAP_HOT = 11

img = np.zeros((64,64))
pt = [32,32]
sigma = 1.5
img = (generate_target(img, pt, sigma)*255.).astype(np.uint8)
img_d = ndi.grey_dilation(img, size=(3,3))
# colormap = cv2.applyColorMap(img, 2)
# colormap_d = cv2.applyColorMap(img_d, 2)
# print(colormap_d[colormap_d>254])
# plt.imshow(colormap)
# plt.show()
# plt.imshow(colormap_d)
# plt.show()


img_d = random_shift(img_d, 20).astype(np.uint8)
print(img_d.shape)
colormap = cv2.applyColorMap(img_d, 2)
plt.imshow(colormap)
plt.show()
plt.close()