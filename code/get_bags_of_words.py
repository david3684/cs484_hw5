import cv2
import numpy as np
from numpy import linalg

from distance import pdist
from feature_extraction import feature_extraction


def get_bags_of_words(image_paths, feature):
    """
    This function assumes that 'vocab_*.npy' exists and contains an vocab size x feature vector
    length matrix 'vocab' where each row is a kmeans centroid or visual word. This
    matrix is saved to disk rather than passed in a parameter to avoid recomputing
    the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size') below.
    """
    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.
    bags_of_words = np.zeros((len(image_paths), vocab_size))

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        features = feature_extraction(img, feature)

        # 각 특징에 대한 가장 가까운 시각 단어 찾기
        distances = pdist(features, vocab)
        closest_vocab_indices = np.argmin(distances, axis=1)

        # 히스토그램 구성
        for idx in closest_vocab_indices:
            bags_of_words[i, idx] += 1

        # L2-노름으로 히스토그램 정규화
        bags_of_words[i, :] /= linalg.norm(bags_of_words[i, :])

    return bags_of_words
    #return np.zeros((1500, 36))
