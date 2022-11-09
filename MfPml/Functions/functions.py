import numpy as np


class functions:

    def __init__(self, x: np.ndarray, fidelity: str = 'high') -> np.ndarray:
        """This function if used to compute the values of forrester function"""
        self.x = x
        self.fidelity = fidelity
        self.y = None
        #########

    def Forrester(self):

        if self.fidelity == 'high':
            self.y = (6 * self.x) ** 2 * np.sin(12 * self.x - 4)
        elif self.fidelity == 'low1a':
            self.y = (6 * self.x) ** 2 * np.sin(12 * self.x - 4) - 5
        elif self.fidelity == 'low1b':
            self.y = 0.5 * ((6 * self.x) ** 2 * np.sin(12 * self.x - 4)) + 10 * (self.x - 0.5) - 5
        elif self.fidelity == 'low1c':
            self.y = (6 * (self.x + 0.2) - 2) ** 2 * np.sin(12 * (self.x + 0.2) - 4)
        else:
            print("Error!!! Please input the right!!! \n")

        return self.y
