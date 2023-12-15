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

    # 각 카테고리에 대한 SVM 모델 초기화
    svms = {category: svm.SVC(kernel=kernel_type.lower(), C=10) for category in categories}

    # 각 카테고리별로 SVM 학습
    for category in categories:
        binary_labels = (train_labels == category).astype(int)
        svms[category].fit(train_image_feats, binary_labels)

    # 테스트 데이터에 대한 예측 수행
    predictions = np.zeros((len(test_image_feats), n_categories))
    for i, category in enumerate(categories):
        predictions[:, i] = svms[category].decision_function(test_image_feats)

    # 가장 높은 점수를 받은 카테고리 선택
    predicted_categories = categories[np.argmax(predictions, axis=1)]

    return predicted_categories
    #return np.array([categories[0]] * 1500)