import cv2
import numpy as np


def sift_kp(image):
    """[sift_kp] function to do SIFT and get key points

    Args:
        image: input image

    Return:
        kp_image: image with key points
        kp: keypoints
        des: keypoint descriptor
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_pairs(des1, des2):
    """[get_pairs] function to do SIFT and get pairs

    Args:
        des1, des2: input image

    Return:
        pairs: key points pairs
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    pairs = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pairs.append(m)
    return pairs


def siftimg_rightlignment(img_right, img_left):
    """[siftimg_rightlignment] function to stitch twp images

    Args:
        img_right, img_left: image to stitch

    Return:
        result: vis result
    """
    _, kp1, des1 = sift_kp(img_right)
    _, kp2, des2 = sift_kp(img_left)
    goodMatch = get_pairs(des1, des2)
    if len(goodMatch) > 4:
        ptsA = np.float32(
            [kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32(
            [kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(
            ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        result = cv2.warpPerspective(
            img_right, H, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        return result


img_right = cv2.imread(r'1.jpg')
img_left = cv2.imread(r'2.jpg')

img_right = cv2.resize(
    img_right, (img_right.shape[1]//4, img_right.shape[0]//4))
img_left = cv2.resize(img_left, (img_right.shape[1], img_right.shape[0]))


kpimg_right, kp1, des1 = sift_kp(img_right)
kpimg_left, kp2, des2 = sift_kp(img_left)


cv2.imshow('kp', np.hstack((kpimg_left, kpimg_right)))
cv2.imwrite('output/keypoints.jpg', np.hstack((kpimg_left, kpimg_right)))
goodMatch = get_pairs(des1, des2)

pairs = cv2.drawMatches(
    img_right, kp1, img_left, kp2, goodMatch, None, flags=2)

cv2.imshow('mt', pairs)
cv2.imwrite('output/pairs.jpg', pairs)

result = siftimg_rightlignment(img_right, img_left)
cv2.imshow('result', result)
cv2.imwrite('output/result.jpg', result)
cv2.waitKey(0)
