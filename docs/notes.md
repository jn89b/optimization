# Notes from Data Driven MPC paper
https://github.com/uzh-rpg/data_driven_mpc/blob/main/ros_gp_mpc/nodes/gp_mpc_node.py

## Important Files 

### gp.py
#### Class CustomKernelFunctions
#### Methods
- has methods for kernel functions one which uses symbolic expressions and the other uses numeric values via numpy

#### Class CustomGPRegression 


#### Class GPEnsemble
You basically can have single or multiple GPs, you're using ensemble machine learning practices refer to this: https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/


### gp_mpc_node.py
- Class Quad3DOptimizer
    - Method checks for gp ensembles, that is it combines multiple GPS to 


## Important Notes
- Checkout the in quad_3d_optimzer.py
```python
        # Build full model. Will have 13 variables. self.dyn_x contains the symbolic variable that
        # should be used to evaluate the dynamics function. It corresponds to self.x if there are no GP's, or
        # self.x_with_gp otherwise
        acados_models, nominal_with_gp = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u)['x_dot'], model_name)

dynamics_equations[i] = nominal + cs.mtimes(self.B_x, gp_means)
```


## Updating the kernels
```python

## Equation referernces from Learning-based Model Predictive Control for Autonomous Racing
    self.K = self.kernel(x_train, x_train) + self.sigma_n ** 2 * np.eye(len(x_train))
    self.K_inv = inv(self.K)
    self.K_inv_y = self.K_inv.dot(y_train) 

    # Ensure at least n=1
    x_test = np.atleast_2d(x_test) if isinstance(x_test, np.ndarray) else x_test

    if isinstance(x_test, cs.MX):
        return self._predict_sym(x_test=x_test, return_std=return_std, return_cov=return_cov)

    if isinstance(x_test, cs.DM):
        x_test = np.array(x_test).T

    k_s = self.kernel(x_test, self.x_train)
    k_ss = self.kernel(x_test, x_test) + 1e-8 * np.eye(len(x_test))

    # Posterior mean value
    mu_s = k_s.dot(self.K_inv_y) + self.y_mean #Equation 3a from paper

    # Posterior covariance #Equation 3b
    cov_s = k_ss - k_s.dot(self.K_inv).dot(k_s.T)
    std_s = np.sqrt(np.diag(cov_s))

    if not return_std and not return_cov:
        return mu_s

    # Return covariance
    if return_cov:
        return mu_s, std_s ** 2

```


