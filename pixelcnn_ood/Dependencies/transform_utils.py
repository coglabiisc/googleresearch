import numpy as np
import cv2
import itertools

def rotate(img, angle):
    # given an image batch and angle, rotates the images
    
    rotated_img = np.zeros(img.shape)
    if angle == 90:
        rot_func = cv2.cv2.ROTATE_90_CLOCKWISE
    elif angle == 180:
        rot_func = cv2.cv2.ROTATE_180
    elif angle == 270:
        rot_func = cv2.ROTATE_90_COUNTERCLOCKWISE
    for i in range (img.shape[0]):
        temp = cv2.rotate(img[i,:,:,:], rot_func)
        if len(temp.shape) == 2:
            temp = np.expand_dims(temp, axis=-1)
        rotated_img[i,:,:,:] = temp
    return np.round(rotated_img)

def rotate_inv(img, angle):
    # given an image batch and angles, laterally inverts and rotates the images
    rotated_img = np.zeros(img.shape)
    if angle == 90:
        rot_func = cv2.cv2.ROTATE_90_CLOCKWISE
    elif angle == 180:
        rot_func = cv2.cv2.ROTATE_180
    elif angle == 270:
        rot_func = cv2.ROTATE_90_COUNTERCLOCKWISE
    for i in range (img.shape[0]):
        temp = cv2.rotate(cv2.flip(img[i,:,:,:],1), rot_func)
        if len(temp.shape) == 2:
            temp = np.expand_dims(temp, axis=-1)
        rotated_img[i,:,:,:] = temp
    return np.round(rotated_img)

def inv(img):
    # laterally inverts a batch of images
    
    inverted_img = np.zeros(img.shape)
    for i in range (img.shape[0]):
        temp = cv2.flip(img[i,:,:,:],1)
        if len(temp.shape) == 2:
            temp = np.expand_dims(temp, axis=-1)
        inverted_img[i,:,:,:] = temp
    return np.round(inverted_img)

def transform(imgs, angle):
    """
    
    Parameters
    ----------
    imgs : array float32 (batch, 32, 32, channels)
        Batch of images
    angle : int32
        Integer from 0 to 6
            0 - 90 degree rotatiom
            1 - 180 degree rotation
            2 - 270 degree rotation
            3 - lateral inversion
            4 - lateral inversion and 90 degree rotation
            5 - lateral inversion and 180 degree rotation
            6 - lateral inversion and 270 degree rotation

    Returns
    -------
    imgs : array float32 (batch, 32, 32, channels)
        transformed images defined by angle parameter

    """
    if angle == 0:
        imgs = rotate(imgs, 90)
    elif angle == 1:
        imgs = rotate(imgs, 180)
    elif angle == 2:
        imgs = rotate(imgs, 270)
    elif angle == 3:
        imgs = inv(imgs)
    elif angle == 4:
        imgs = rotate_inv(imgs, 90)
    elif angle == 5:
        imgs = rotate_inv(imgs, 180)
    elif angle == 6:
        imgs = rotate_inv(imgs, 270)
    return imgs
    
def derangement(n):
    # returns all derangements for n<=2, and random 20 derangements for n>2
    
    if n <= 2:
        orders = np.array(list(itertools.permutations(np.arange(n*n))))
        tmp = []
        for i in range (orders.shape[0]):
            count = 0
            for j in range (orders.shape[1]):
                if orders[i,j] == j:
                    count += 1
            if count == 0:
                tmp.append(orders[i,:])
        return np.array(tmp), np.array(tmp).shape[0]
    else:
        shuffles = 20
        orders = np.zeros((shuffles, n*n))
        tmp = np.arange(n*n)
        for i in range (shuffles):
            p = np.random.permutation(n*n)
            orders[i,:] = tmp[p]
        return orders.astype('int32'), orders.shape[0]

def patch_shuffle(imgs, n, orders):
    """

    Parameters
    ----------
    imgs : array float32 (batch, 32, 32, channels)
        batch of images to be shaken
    n : int32
        number of patches in each axis (n=2 gives 2x2 patches)
    orders : vector int
        derangements of patches to be returned

    Returns
    -------
    shuffled : array float32 (batch, 32, 32, channels, derangements)
        shaken batch of images

    """
    shap = imgs.shape
    patches = np.zeros((shap[0], int(shap[1]/n), int(shap[2]/n), shap[3], int(n*n)))
    patch_shap = patches.shape
    shuffled = np.zeros((shap[0], shap[1], shap[2], shap[3], orders.shape[0]))
    for i in range (n):
        for j in range (n):
            patches[:,:,:,:,i*n+j] = imgs[:, i*(patch_shap[1]):(i+1)*(patch_shap[1]), j*(patch_shap[2]):(j+1)*(patch_shap[2]), :]
    for k in range (orders.shape[0]):
        for i in range (n):
            for j in range (n):
                shuffled[:, i*(patch_shap[1]):(i+1)*(patch_shap[1]), j*(patch_shap[2]):(j+1)*(patch_shap[2])
                         , :, k] = patches[:,:,:,:,np.squeeze(orders[k,i*n+j])]
    return shuffled