# Import necessary libraries
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor

# Create a class for Gaussian Stationary analysis
class Gaussian_Stationary:
    
    def __init__(self, df_comp, data_df_res):
        # Initialize the Gaussian_Stationary class with data
        
        # Split the data into training and test sets
        self.split_data(data_df_res, df_comp)

        # Set up the Gaussian Process kernel
        self.set_Kernel()

        # Create and configure the Gaussian Process model
        self.get_model()

        # Train the model on the data
        self.train()

    def train(self):
        """
        Train the Gaussian Process model.
        """
        # Fit the GP1 model on the training data
        self.gp1.fit(self.x_train_res_1, self.y_train_res_1)
        
        # Generate predictions and standard deviations for the training data
        y_pred, y_std = self.gp1.predict(self.x_train_res_1, return_std=True)
        self.df_train_res['y_pred'] = y_pred
        self.df_train_res['y_std'] = y_std
        self.df_train_res['y_pred_lwr'] = self.df_train_res['y_pred'] - 2 * self.df_train_res['y_std']
        self.df_train_res['y_pred_upr'] = self.df_train_res['y_pred'] + 2 * self.df_train_res['y_std']
        
        # Plot the predictions and actual data for training
        plt.figure(figsize=(20, 5))
        plt.plot(self.df_train_res["y_pred"], color='red')
        plt.plot(self.df_train_res["delta_1_Healthcare"])
        plt.savefig("Output/" + "pred_delta_train.png")
        
        # Generate predictions and standard deviations for the test data
        y_pred, y_std = self.gp1.predict(self.x_test_res_1, return_std=True)
        self.df_test_res['y_pred'] = y_pred
        self.df_test_res['y_std'] = y_std
        self.df_test_res['y_pred_lwr'] = self.df_test_res['y_pred'] - 2 * self.df_test_res['y_std']
        self.df_test_res['y_pred_upr'] = self.df_test_res['y_pred'] + 2 * self.df_test_res['y_std']
        
        # Plot the predictions and actual data for the test data
        plt.figure(figsize=(20, 5))
        plt.plot(self.df_test_res["y_pred"])
        plt.plot(self.df_test_res["delta_1_Healthcare"], color='red')
        plt.savefig("Output/" + "pred_stationary.png")

    def get_model(self):
        """
        Set the hyperparameters and create a Gaussian Process model.
        """
        # Configure the Gaussian Process model
        self.gp1 = GaussianProcessRegressor(
            kernel=self.kernel_1,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=0.004
        )

    def set_Kernel(self):
        """
        Set up the kernel for Gaussian Process.
        """
        # Define the components of the kernel
        k0 = WhiteKernel(noise_level=0.3 ** 2, noise_level_bounds=(0.1 ** 2, 0.5 ** 2))
        k1 = ConstantKernel(constant_value=2) * \
             ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45))
        
        # Combine the kernel components
        self.kernel_1 = k0 + k1

    def split_data(self, data_df_res, df_comp):
        """
        Split the data into training and test sets.
        """
        # Define the test set size
        test_size = 12
        
        # Extract features and target variable
        X = df_comp["timestamp"]
        y = df_comp["delta_1_Healthcare"]
        
        # Split the data into training and test sets
        x_train_res = X[:-test_size]
        y_train_res = y[:-test_size]
        x_test_res = X[-test_size:]
        y_test_res = y[-test_size:]
        
        # Create DataFrames for training and test data
        self.df_train_res = data_df_res[:-test_size][1:]
        self.df_test_res = data_df_res[-test_size:][1:]
        
        # Plot the training and test data
        plt.figure(figsize=(20, 5))
        plt.title('Train and Test Sets', size=20)
        plt.plot(y_train_res, label='Training set')
        plt.plot(y_test_res, label='Test set', color='orange')
        plt.legend()
        plt.savefig("Output/" + "split.png")
        
        # Reshape the data for modeling
        self.x_train_res_1 = x_train_res.values.reshape(-1, 1)[1:]
        self.y_train_res_1 = y_train_res.values.reshape(-1, 1)[1:]
        self.x_test_res_1 = x_test_res.values.reshape(-1, 1)[1:]
        self.y_test_res_1 = y_test_res.values.reshape(-1, 1)[1:]
