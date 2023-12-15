import numpy as np

def get_features_from_pca(feat_num, feature):

    """
    This function loads 'vocab_*.npy' file and
	returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
	:param feature: 'HoG' or 'SIFT'

    :return: an N x feat_num matrix
    """

    vocab = np.load(f'vocab_{feature}.npy')

    # Your code here. You should also change the return value.
    
    vocab_mean = np.mean(vocab, axis=0)
    vocab_centered = vocab - vocab_mean

    covariance_matrix = np.cov(vocab_centered, rowvar=False)

    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

    idx = np.argsort(eigen_values)[::-1]
    principal_components = eigen_vectors[:, idx[:feat_num]]

    reduced_vocab = np.dot(vocab_centered, principal_components)

    return reduced_vocab


