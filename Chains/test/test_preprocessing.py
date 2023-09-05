"""Test for preprocessing"""

import numpy as np
import preprocessing

class TestPreprocessing:
    def setup_method(self):
        self.im_shape = (1024, 1024)
        self.im_list = [np.zeros(self.im_shape) for i in range(3)]