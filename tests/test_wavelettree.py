        # debug only, make sure that X[idxs,:] is the same as pca reconstruct
        # x_orig = np.atleast_2d(X[idxs,:])
        # multiply with sqrt of size because for somereason the scale it in pca
        # x_hat = np.sqrt(self.size) *  self.basis.T @ np.diag(self.sigmas[:len(self.sigmas)-1])[:self.basis.T.shape[1]] @ self.Z + self.center


        # print('difference:', np.linalg.norm(x_orig.T - x_hat))  
        # assert np.linalg.norm(x_orig.T - x_hat) < .1, 'reconstruct fail'
        # print('number of basis', self.basis.shape[0])
        # print('reconstruction error:', np.linalg.norm(X[idxs,:] - x_hat))

        # self.center, self.size, self.radius, self.basis, self.sigmas, self.Z = no(
        #     np.atleast_2d(X[idxs,:]),
        #     manifold_dim,
        #     max_dim,
        #     is_leaf,
        #     shelf=shelf,
        #     threshold=threshold,
        #     precision=precision
        # )

