import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_*.npy' exists and
    contains an vocab size x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """

    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.
    d = 0
    for l in range(max_level + 1):
        d += (vocab_size * (4 ** l))
    pyramid_features = np.zeros((len(image_paths), d))

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        current_feature = []
        for l in range(max_level + 1):
            sub_h, sub_w = h // (2 ** l), w // (2 ** l)
            for y in range(2 ** l):
                for x in range(2 ** l):
                    sub_img = img[y*sub_h:(y+1)*sub_h, x*sub_w:(x+1)*sub_w]
                    sub_features = feature_extraction(sub_img, feature)

                    distances = pdist(sub_features, vocab)
                    closest_vocab_indices = np.argmin(distances, axis=1)

                    hist = np.zeros(vocab_size)
                    for idx in closest_vocab_indices:
                        hist[idx] += 1

                    weight = 2 ** (-max_level+l-1) if l > 0 else -max_level
                    current_feature.extend(weight * hist)

        normalized_feature = np.array(current_feature) / linalg.norm(current_feature)

        pyramid_features[i, :len(normalized_feature)] = normalized_feature

    return pyramid_features
