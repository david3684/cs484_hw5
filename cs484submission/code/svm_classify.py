import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats:
        an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels:
        an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats:
        an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type:
        the name of a kernel type. 'linear' or 'RBF'.

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)

    # Your code here. You should also change the return value.
    n_categories = len(categories)

    svms = {category: svm.SVC(kernel=kernel_type.lower(), C=10) for category in categories}

    for category in categories:
        labels = (train_labels == category).astype(int)
        svms[category].fit(train_image_feats, labels)

    predictions = np.zeros((len(test_image_feats), n_categories))
    for i, category in enumerate(categories):
        predictions[:, i] = svms[category].decision_function(test_image_feats)

    predicted_categories = categories[np.argmax(predictions, axis=1)]

    return predicted_categories
