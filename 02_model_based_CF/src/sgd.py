from tqdm import tqdm
import numpy as np


class MF():

    def __init__(self, r_mat, n_latent, lr, reg_param, n_epoch, verbose=False):
        
        """
        * param r_mat: 평점 행렬
        * param k: user latent vector, item latent vector 차원 수
        * param lr: learning rate
        * param reg_param: weight regularization
        * param epoch: 최대 epoch 수
        * param verbose: 10번 마다 cost 출력
        """

        self._r_mat = r_mat
        self._n_user, self._n_item = r_mat.shape
        self._n_latent = n_latent
        self._lr = lr
        self._reg_param = reg_param
        self._n_epoch = n_epoch
        self._verbose = verbose


    def fit(self):
        """
        * Update matrix latent weight and bias
        * r = mat_u x mat_i
        """
        
        # * 1. u, i matrix 초기화
        self._u_mat = np.random.normal(size=(self._n_user, self._n_latent))
        self._i_mat = np.random.normal(size=(self._n_item, self._n_latent))
        
        # * u, i bias 초기화
        self._b_u = np.zeros(self._n_user)
        self._b_i = np.zeros(self._n_item)
        self._b = np.mean(self._r_mat[np.where(self._r_mat != 0)]) # * 전체 평점의 평균

        # * 학습
        self._training_process = list()
        
        for epoch in range(self._n_epoch):
            
            # * 평점이 0 이 아닌 index
            # * 사용자가 평점을 기록한 부분에 대해서만 진행
            xi, yi = self._r_mat.nonzero() 
            
            for i, j in zip(xi, yi):

                self.gradient_descent(i, j, self._r_mat[i, j])

            cost = self.get_cost()
            self._training_process.append([epoch, cost])

            if ((epoch + 1) % self._verbose == 0):

                print(f"Epoch {epoch}: {cost}")
    
        return None


    def gradient_descent(self, i, j, rating):

        """
        * param i: user index of matrix
        * param j: item index of matrix
        * param rating: rating of (i, j) 
        """

        # * Get error
        rating_pred = self.get_prediction(i, j)
        error = rating - rating_pred

        # * Update bias
        self._b_u[i] += self._lr * (error - self._reg_param * self._b_u[i])
        self._b_i[j] += self._lr * (error - self._reg_param * self._b_i[j])

        # * Update latent
        du, di = self.get_gradient(error, i, j)
        self._u_mat[i, :] += self._lr * du
        self._i_mat[j, :] += self._lr * di

        return None


    def get_gradient(self, error, i, j):
        
        """
        * param error: rating - rating_pred
        * param i: user index
        * param j: item index
        * return: gradient latent du, di
        """

        du = (error * self._i_mat[j, :]) - (self._reg_param * self._u_mat[i, :])
        di = (error * self._u_mat[i, :]) - (self._reg_param * self._i_mat[j, :])

        return du, di

    def get_prediction(self, i, j):
        
        """
        * param i: user index of matrix
        * param j: item index of matrix
        * :return: predicted rating of r_ij
        """

        rating_pred =  self._u_mat[i, :].dot(self._i_mat[j, :].T) + self._b_u[i] + self._b_i[j] + self._b

        return rating_pred


    def get_cost(self):
        
        """
        * :return: rmse
        """

        xi, yi = self._r_mat.nonzero()
        cost = 0

        for i, j in zip(xi, yi):

            cost += pow(self._r_mat[i, j] - self.get_prediction(i, j), 2)

        cost = np.sqrt(cost/len(xi))

        return cost


    def get_completed_matrix(self):

        completed_mat = self._u_mat.dot(self._i_mat.T) + self._b_u[:, np.newaxis] + self._b_i[np.newaxis, :] + self._b

        return completed_mat


if __name__ == '__main__':

    np.random.seed(42)

    r_mat = np.array([
        [1, 0, 0, 1, 3],
        [2, 0, 3, 1, 1],
        [1, 2, 0, 5, 0],
        [1, 0, 0, 4, 4],
        [2, 1, 5, 4, 0],
        [5, 1, 5, 4, 0],
        [0, 0, 0, 1, 0],
    ])

    mf = MF(r_mat=r_mat,
            n_latent=3,
            lr=.01,
            reg_param=.01,
            n_epoch=1000,
            verbose=100)
            
    mf.fit()
    completed_mat = mf.get_completed_matrix()
