import numpy as np
from nptyping import NDArray, Float, Shape

class KalmanFilterClass2D(object):
    """A Kalman filter to track a single object in 2D.
    
    dt: float, time step between images.
    sd_acceleration: float, standard deviation of the mean 0 acceleration
    mes_sd: float, measurement error."""
    def __init__(self, dt: float, sd_acceleration: float, mes_sd: float)-> None:
        self.dt = dt
        #  The state transition matrix 
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # The matrix that maps state vector to measurement 
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        # Processs Covariance that for our case depends solely on the acceleration  
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * sd_acceleration ** 2
        # Measurement Covariance
        self.R = np.matrix([[mes_sd ** 2, 0],
                           [0, mes_sd ** 2]])
        # The error covariance matrix that is Identity for now. It gets updated based on Q, A and R.
        self.P = np.eye(self.A.shape[1])
        #  Finally the vector in consideration ; it's [ x position ;  y position ; x velocity ; y velocity ; ]
        self.x = np.matrix([[0], [0], [0], [0]])

    def predict(self) -> NDArray[Shape["2", Float]]:
        """"Predict the new state based on the previous one."""
        # The state update : X_t = A*X_t-1 + B*u 
        # here u is acceleration,a 
        self.x = np.dot(self.A, self.x) 
        
        # Update of the error covariance matrix 
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z: NDArray[Shape["2", Float]])-> NDArray[Shape["2", Float]]:
        """Update the position based both on the measured position and the predicted one."""
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 
        self.x += np.dot(K, (z - np.dot(self.H, self.x)))
        I = np.eye(self.H.shape[1])
        self.P = (I -(K*self.H))*self.P  
        return self.x[0:2]