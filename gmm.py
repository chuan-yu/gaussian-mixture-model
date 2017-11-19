import numpy as np
from scipy.stats import multivariate_normal
from functions import io_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GMM:
    """
    This class define a Gaussian Mixture Model object.

    Args:
        data (array): input data, N x D array, N is the no. of observations, D is the no. of features
        ncomp (int): no. of Gaussian components

    Attributes:
        mu (array): means, dimension: ncomp x D
        sigma (array): covariance matrix, dimension: ncomp x D x D
        pi (array): mixing coefficients, dimension: ncomp
        gamma (array): responsibilities, dimension: N x ncomp
        likelihood (float): log likelihood

    """

    def __init__(self, data, ncomp):

        self.__data = data
        self.__K = ncomp
        self.__N, self.__D = data.shape[0], data.shape[1]

        # Initialize mu, sigma, normal distributions and  pi
        self.mu = np.vstack((self.__data[0], self.__data[self.__N // 2]))
        self.sigma = np.ndarray((self.__K, self.__D, self.__D), np.float32)

        cov = np.cov(data, rowvar=False)
        for i in range(self.__K):
            self.sigma[i, :, :] = cov
        self.__update_distributions()
        np.random.seed(100)
        self.pi = np.random.rand(self.__K)

        self.likelihood = 0

    def fit(self, max_steps=500):
        """
        Fit the multivariate normal distributions
        :param max_steps: maximum number of iterations
        """
        for _ in range(max_steps):
            # E step
            self.__update_gamma()

            # M step
            self.__update_mu()
            self.__update_sigma()
            self.__update_pi()

            # Evaluate log likelihood
            self.__update_distributions()
            self.__compute_likelihood()
            print(self.likelihood)

            # check convergence
            if abs(self.likelihood - self.__likelihood_old) <= 1e-7:
                return

    def __update_gamma(self):
        """
        Update responsibilities
        """

        probs = np.ndarray((self.__N, self.__K))

        for k in range(self.__K):
            probs[:, k] = self.pi[k] * self.distributions[k].pdf(self.__data)

        l1_norm = np.sum(probs, axis=1)
        l1_norm = l1_norm.reshape(l1_norm.shape[0], 1)
        self.gamma = probs / l1_norm

        N_k = np.sum(self.gamma, axis=0)
        self.__N_k = N_k.reshape(N_k.shape[0], -1)

    def __update_mu(self):
        """
        Update means
        """
        self.mu = np.divide(np.matmul(np.transpose(self.gamma), self.__data), self.__N_k)

    def __update_sigma(self):
        """
        Update covariance matrices
        """
        for i in range(self.__K):
            gamma = self.gamma[:, i]
            gamma = gamma.reshape(gamma.shape[0], -1)
            mean = self.mu[i, :]
            mean = mean.reshape(-1, mean.shape[0])
            data_shifted = np.subtract(self.__data, mean)
            cov = np.divide(np.matmul(np.transpose(np.multiply(gamma, data_shifted)), data_shifted), self.__N_k[i])
            self.sigma[i, :, :] = cov

    def __update_pi(self):
        """
        Update mixing coefficients
        """
        self.pi = np.divide(self.__N_k, self.__N)

    def __update_distributions(self):
        """
        Update the normal distributions based on the new parameters
        """
        self.distributions = []
        for i in range(self.__K):
            self.distributions.append(multivariate_normal(self.mu[i], self.sigma[i]))

    def __compute_likelihood(self):
        """
        Compute the log likelihood
        """
        # save the old likelihood
        self.__likelihood_old = self.likelihood

        probs = np.ndarray((self.__N, self.__K))
        for k in range(self.__K):
            probs[:, k] = self.pi[k] * self.distributions[k].pdf(self.__data)
        self.likelihood = np.sum(np.log(np.sum(probs, axis=1)), axis=0)


if __name__ == "__main__":

    input_folder = "a2/"
    file_name = "zebra"
    output_folder = "results/"
    max_steps = 100

    # Load data
    data, image = io_data.read_data(input_folder + file_name + ".txt", False)
    x, y = image.shape[0], image.shape[1]
    N = x * y
    image = image.reshape(N, image.shape[2])

    # Fit GMM model
    gmm = GMM(image, 2)
    gmm.fit(max_steps)

    # Generate mask
    mask = np.argmax(gmm.gamma, axis=1)
    mask = mask.reshape(mask.shape[0], 1)

    # Separate background and foreground
    foreground = np.multiply(mask, image)
    background = image - foreground
    mask = mask.reshape((x, y))
    background = background.reshape((x, y, 3))
    foreground = foreground.reshape((x, y, 3))

    # Save images
    plt.imsave(output_folder + file_name + "_mask.png", mask, cmap=cm.gray)
    plt.imsave(output_folder + file_name + "_background.png", background, cmap=cm.gray)
    plt.imsave(output_folder + file_name + "_foreground.png", foreground, cmap=cm.gray)







