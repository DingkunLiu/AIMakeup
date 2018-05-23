# import the necessary packages
import argparse
from time import time
import os

import cv2
import dlib
import imutils
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils

left_eye_index = list(range(36, 42))
right_eye_index = list(range(42, 48))
left_face_index = list(range(9))
right_face_index = list(range(8, 17))
nose_index = list(range(27, 31))

__output_shape = (2835, 2835)
blue_background = cv2.imread('%s/blue.jpg' % os.path.dirname(__file__))
white_background = 255 * np.ones((*__output_shape, 3), dtype=np.uint8)


class Detector:
    def __init__(self, model_path='./data/'):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat'))

    def detect(self, image, resize=True):
        if resize:
            size = image.shape
            image = imutils.resize(image, width=500)
            size_after = image.shape
            ratio = size[0] / size_after[0]
        rects = self.detector(image, 1)
        assert len(rects) == 1
        shape = self.predictor(image, rects[0])
        rect = rects[0]

        # dlib to python
        rect = face_utils.rect_to_bb(rect)
        shape = face_utils.shape_to_np(shape, dtype=np.float32)

        if resize:
            rect = list(map(lambda x: x * ratio, rect))
            shape *= ratio
        return rect, shape


def draw(image, rect, shape):
    if not isinstance(rect, tuple) and not isinstance(rect, list):
        rect = face_utils.rect_to_bb(rect)
    if not isinstance(shape, np.ndarray):
        shape = face_utils.shape_to_np(shape)
    if not isinstance(rect[0], int):
        rect = list(map(lambda x: int(np.round(x)), rect))
    image = image.copy()
    (x, y, w, h) = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y) in shape:
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), int(np.round(image.shape[0] / 500)), (0, 0, 255), -1)
    return image


def _transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def image_transform(image, shape):
    """
    Need two transformation in total:
    scaling (fixed x y ratio)
    and translation
    Need two points: jaw and upper bound of head
    :param image: image to be transformed
    :param shape: facial landmarks
    :return: transformed image and its matrix
    """

    # fixed this according to template
    # upper 273
    # bottom 1773
    assert image.dtype == np.uint8
    left_eye_center = (1218, 1084)
    right_eye_center = (1616, 1084)
    jaw = (1417.5, 1770)
    if not isinstance(shape, np.ndarray):
        shape = face_utils.shape_to_np(shape)
    # eye center and jaw
    p_left = np.mean(shape[left_eye_index], axis=0)
    p_right = np.mean(shape[right_eye_index], axis=0)
    p_jaw = shape[8]

    p_source = np.array([p_left, p_right, p_jaw], dtype=np.float32)
    p_target = np.array([left_eye_center, right_eye_center, jaw], dtype=np.float32)
    # affine transform
    transform_matrix = cv2.estimateRigidTransform(p_source, p_target, fullAffine=False)
    image = cv2.warpAffine(image, transform_matrix, (__output_shape[1], __output_shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
    # pad transform matrix to (3, 3)
    pad = np.zeros((1, 3))
    pad[0, -1] = 1
    transform_matrix = np.concatenate((transform_matrix, pad), axis=0)
    return image.astype(np.uint8), transform_matrix


def _character_segm_white(image, background):
    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(image, mask, (100, 100), (255, 255, 255), loDiff=(2, 2, 2), upDiff=(2, 2, 2),
                  flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    mask = mask[1:-1, 1:-1]
    # morphological transform
    kernel = np.ones((50, 50), dtype=np.uint8)
    sure_fg = cv2.erode(255 - mask, kernel, iterations=1)
    sure_bg = cv2.erode(mask, kernel, iterations=1)
    marker = np.zeros_like(mask, dtype=np.int32)
    # unknown region 0
    # foreground
    marker[sure_fg == 255] = 2
    # background
    marker[sure_bg == 255] = 1
    # apply watershed to better segment hair boarder
    marker = cv2.watershed(image, marker)
    final_mask = np.zeros_like(marker, dtype=np.uint8)
    final_mask[marker == 1] = 255
    # smooth edge
    final_mask = cv2.GaussianBlur(final_mask, (15, 15), 9)


    # mask to float
    mask_f = np.tile(final_mask[:, :, None], [1, 1, 3]).astype(np.float32) / 255
    # add together
    im2 = background * mask_f + image * (1 - mask_f)
    return im2.astype(np.uint8), final_mask


def _character_segm_blue(image, background):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H = hsv_image[..., 0]
    S = hsv_image[..., 1]
    # select patch to calculate
    H_mean = H[:20, :20].mean()
    S_mean = S[:20, :20].mean()
    # threshold
    seg1 = cv2.inRange(H, H_mean - 5, H_mean + 5)
    seg2 = cv2.inRange(S, S_mean - 30, S_mean + 30)
    mask = np.logical_and(seg1 == 255, seg2 == 255).astype(np.uint8) * 255
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get index of connected component with max area
    area = [cv2.contourArea(contour) for contour in contours]
    index = np.argmax(area)
    # build mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv2.drawContours(mask, contours, index, 255, cv2.FILLED)

    # morphological transform
    kernel = np.ones((50, 50), dtype=np.uint8)
    sure_fg = cv2.erode(255 - mask, kernel, iterations=1)
    sure_bg = cv2.erode(mask, kernel, iterations=1)
    marker = np.zeros(mask.shape[:2], dtype=np.int32)
    # unknown region 0
    # foreground
    marker[sure_fg == 255] = 2
    # background
    marker[sure_bg == 255] = 1
    # apply watershed to better segment hair boarder
    marker = cv2.watershed(image, marker)
    final_mask = np.zeros_like(marker, dtype=np.uint8)
    final_mask[marker == 1] = 255
    final_mask = cv2.GaussianBlur(final_mask, (15, 15), 9)

    # mask to float
    mask_f = np.tile(final_mask[:, :, None], [1, 1, 3]).astype(np.float32) / 255
    # add together
    im2 = background * mask_f + image * (1 - mask_f)
    return im2.astype(np.uint8), final_mask


def segm_bg(image, mode='post-process'):
    """
    Fill background with
    :param image:
    :param mode:
    :return:
    """
    assert mode in ['post-process', 'training']
    if np.all(image[100, 100, 1:] < 150):
        # transform blue background
        if mode == 'post-process':
            background = cv2.resize(blue_background, dsize=(image.shape[1], image.shape[0]))
        else:
            background = np.zeros(image.shape[:2], dtype=np.uint8)

        background = cv2.warpPerspective(background, transform_matrix, (__output_shape[1], __output_shape[0]))
        im2, mask = _character_segm_blue(image, background)

    else:
        if mode == 'post-process':
            background = white_background
        else:
            background = np.zeros(image.shape[:2], dtype=np.uint8)
        im2, mask = _character_segm_white(image, background)
    return im2, mask


# Check if a point is inside a rectangle
def _rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw delaunay triangles
def _draw_delaunay(img, triangleList, delaunay_color):
    img = img.copy()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if _rect_contains(r, pt1) and _rect_contains(r, pt2) and _rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 2, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 2, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 2, cv2.LINE_AA, 0)
    return img


def _mesh_initializer(landmarks):
    """
    Get triangulation mesh:
        Call subdiv.getEdgelist() to obtain mesh edges
    :param landmarks: facial landmark numpy array
    :return:
    """
    if not isinstance(landmarks, np.ndarray):
        landmarks = face_utils.shape_to_np(landmarks)
    landmarks = landmarks.astype(np.int32)
    size = __output_shape
    landmarks_added = np.array(
        [[0, 0], [size[0] // 2, 0], [size[0] - 1, 0], [0, size[1] // 2], [0, size[1] - 1], [size[0] - 1, size[1] // 2],
         [size[0] // 2, size[1] - 1], [size[0] - 1, size[1] - 1]], dtype=np.int32)
    landmarks = np.append(landmarks, landmarks_added, axis=0).astype(np.int32)
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    for i in range(landmarks.shape[0]):
        subdiv.insert((landmarks[i][0], landmarks[i][1]))
    return subdiv


def subdivlist2edgemat(landmarks, edge_list):
    """
    Convert subdiv to edge matrix
    :param landmarks:
    :param edge_list:
    :return:
    """
    r = (0, 0, __output_shape[1], __output_shape[0])
    # build landmark hash
    landmarks = landmarks.astype(np.int64)
    edge_mat = np.zeros([landmarks.shape[0]] * 2)
    land_dict = {}
    for i in range(landmarks.shape[0]):
        land_dict[(landmarks[i][0], landmarks[i][1])] = i
    for edge in edge_list:
        if _rect_contains(r, edge[:2]) and _rect_contains(r, edge[2:]):
            i = land_dict[(int(edge[0]), int(edge[1]))]
            j = land_dict[(int(edge[2]), int(edge[3]))]
            edge_mat[i][j] = np.sqrt((edge[2] - edge[0]) ** 2 + (edge[3] - edge[1]) ** 2)
    return edge_mat


def subdivlist2triangleindex(landmarks, trianglelists):
    r = (0, 0, __output_shape[1], __output_shape[0])
    # add some points
    size = __output_shape
    landmarks_added = np.array(
        [[0, 0], [size[0] // 2, 0], [size[0] - 1, 0], [0, size[1] // 2], [0, size[1] - 1], [size[0] - 1, size[1] // 2],
         [size[0] // 2, size[1] - 1], [size[0] - 1, size[1] - 1]], dtype=np.int32)
    landmarks = np.append(landmarks, landmarks_added, axis=0).astype(np.int32)
    # hash table for landmarks position
    land_dict = {}
    for i in range(landmarks.shape[0]):
        land_dict[(landmarks[i][0], landmarks[i][1])] = i
    trilist = []
    for t in trianglelists:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if _rect_contains(r, pt1) and _rect_contains(r, pt2) and _rect_contains(r, pt3):
            ind1, ind2, ind3 = list(map(lambda x: land_dict[x], [pt1, pt2, pt3]))
            trilist.append((ind1, ind2, ind3))
    return trilist


def _transform_rect(rect, transform_matrix):
    rect_p1 = np.array(rect[:2])
    rect_p2 = np.array([rect[0] + rect[2], rect[1] + rect[3]])
    rect_p1 = cv2.perspectiveTransform(rect_p1[None, None, :], transform_matrix).squeeze()
    rect_p2 = cv2.perspectiveTransform(rect_p2[None, None, :], transform_matrix).squeeze()
    rect = [rect_p1[0], rect_p1[1], rect_p2[0] - rect_p1[0], rect_p2[1] - rect_p1[0]]
    return rect


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    model = Detector()
    t = time()
    image = cv2.imread(args['image'])
    filename = args['image'].split('/')[-1].split('.')[0]
    print(filename)

    # detect landmarks
    rect, shape = model.detect(image, resize=True)

    image_transformed, transform_matrix = image_transform(image, shape)

    shape = cv2.perspectiveTransform(shape[None, ...], transform_matrix).squeeze()
    rect = _transform_rect(rect, transform_matrix)

    im2, mask = segm_bg(image_transformed)

    # im2 = image_transformed
    cv2.imwrite('./processed/%s_mask.jpg' % filename, mask)
    cv2.imwrite('./processed/%s_cut.jpg' % filename, im2)
    print(time() - t)

    # # subdiv
    # # im2 = cv2.imread('../data/processed/example processed.jpg')
    # subdiv = _mesh_initializer(shape)
    # im3 = _draw_delaunay(im2, subdiv.getTriangleList(), (255, 0, 0))
    # cv2.imwrite('../processed/%s_tri.jpg' % filename, im3)
    #
    # # draw landmarks
    # img = draw(im2, rect, shape)
    # cv2.imwrite('../processed/%s_lm.jpg' % filename, img)

