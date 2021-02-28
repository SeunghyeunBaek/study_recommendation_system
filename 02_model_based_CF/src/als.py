import numpy as np
from scipy.sparse import csr_matrix

class ALS():


    def __init__(self, r_mat, n_latent, reg_param, n_epoch, verbose=100):
        
        self.r_mat = r_mat
        self.n_user, self.n_item = r_mat.shape
        self.n_latent = n_latent
        self.reg_param = reg_param
        self.n_epoch = n_epoch
        self.verbose = verbose

    
    def fit(self):

        # * Initiate latent matrix
        self.user_latent_mat = np.random.normal(size=(self.n_user, self.n_latent))
        self.item_latent_mat = np.random.normal(size=(self.n_item, self.n_latent))

        self.training_process = list()
        self.user_error, self.item_error = 0, 0

        for epoch in range(self.n_epoch):
            
            # * For all User vectors
            for i, user_vector in enumerate(self.r_mat):

                self.user_latent_mat[i] = self.get_user_latent_vector(i, user_vector)
                self.user_error = self.cost()

            # * For all item vectors
            for i, item_vector in enumerate(self.r_mat.T):

                self.item_latent_mat[i] = self.get_item_latent_vector(i, item_vector)
                self.item_error = self.cost()

            cost = self.cost()

            if ((epoch + 1)%verbose == 0):

                print(f"Epoch {epoch}: {cost}")

    
    def get_user_latent_vector(self, i, user_vector):

        du = np.linalg.solve(np.dot(self.item_latent_mat.T, np.dot(np.diag(user_vector), self.item_latent_mat)) +\
            self.reg_param * np.eye(self.n_latent), np.dot(self.item_latent_mat.T, np.dot(np.diag(user_vector), self.user_vector[i].T))).T

        return du

    
    def get_item_latent_vector(self, i, item_vector):

        di = np.linalg.solve(np.dot(self.user_latent_vector.T, np.dot(np.diag(item_vector), self.user_latent_vector)) +\
            self.reg_param * np.eye(self.n_latent), np.dot(self.user_latent_vector.T, np.dot(np.diag(item_vector), self.r_mat[:, i])))

        return di


    def cost(self):

        xi, yi = self.r_mat.nonzero()
        cost = 0

        for x, y in zip(xi, y1):

            cost += pow((self.r_mat[x, y] - self.get_prediction(x, y)), 2)

        cost = np.sqrt(cost/len(xi))

        return cost


    def get_prediction(self, x, y):

        rating_pred = self.user_latent_mat[x, :].dot(self.item_latent_mat[y, :].T)

        return rating_pred            

if __name__ == '__main__':

    r_mat = np.array([
            [1, 0, 0, 1, 3],
            [2, 0, 3, 1, 1],
            [1, 2, 0, 5, 0],
            [1, 0, 0, 4, 4],
            [2, 1, 5, 4, 0],
            [5, 1, 5, 4, 0],
            [0, 0, 0, 1, 0],
    ])
    
    r_mat = csr_matrix(r_mat)
    als = ALS(r_mat=r_mat, n_latent=3, reg_param=.01, n_epoch=100, verbose=100)
    als.fit()