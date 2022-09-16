import numpy as np



def PCA( X,k):
    """
    Assumes observations in X are passed as rows of a numpy array.
    """

    # Translate the dataset so it's centered around 0
    translated_X = X - np.mean(X, axis=0)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    e_values, e_vectors = np.linalg.eigh(np.cov(translated_X.T))

    # Sort eigenvalues and their eigenvectors in descending order
    e_ind_order = np.flip(e_values.argsort())
    e_values = e_values[e_ind_order]
    values_total=np.sum(e_values)
    values_sum=0
    for i in range(len(e_values)):
        values_sum+=e_values[i]
        if values_sum/values_total >0.8:
            break
    e_vectors = e_vectors[e_ind_order]

    # Save the first n_components eigenvectors as principal components
    principal_components = np.take(e_vectors, np.arange(k), axis=0)

    return np.matmul(translated_X, principal_components.T),i