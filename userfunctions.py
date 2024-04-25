from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

class Gaussian3D(IFunction1D):

    def init(self):
        # Tell Mantid about the 2 parameters
        self.declareParameter("y0", 0.0)
        self.declareParameter("A", 1.0)
        self.declareParameter("sig_x", 5.0)
        self.declareParameter("sig_y", 5.0)
        self.declareParameter("sig_z", 5.0)

    def function1D(self, xvals):
        # xvals is a 1D numpy array that contains the X values for the defined fitting range.
        y0 = self.getParameterValue("y0")
        A = self.getParameterValue("A")
        sig_x = self.getParameterValue("sig_x")
        sig_y = self.getParameterValue("sig_y")
        sig_z = self.getParameterValue("sig_z")

        y = xvals[:, np.newaxis, np.newaxis]
        n_steps = 50  # Low number of integration steps because otherwise too slow
        theta = np.linspace(0, np.pi / 2, n_steps)[np.newaxis, :, np.newaxis]
        phi = np.linspace(0, np.pi / 2, n_steps)[np.newaxis, np.newaxis, :]

        S2_inv = (
            np.sin(theta) ** 2 * np.cos(phi) ** 2 / sig_x**2
            + np.sin(theta) ** 2 * np.sin(phi) ** 2 / sig_y**2
            + np.cos(theta) ** 2 / sig_z**2
        )

        J = np.sin(theta) / S2_inv * np.exp(-(y**2) / 2 * S2_inv)

        J = np.trapz(J, x=phi, axis=2)[:, :, np.newaxis]  # Keep shape
        J = np.trapz(J, x=theta, axis=1)

        J *= (
            A * 2 / np.pi * 1 / np.sqrt(2 * np.pi) * 1 / (sig_x * sig_y * sig_z)
        )  # Normalisation
        J = J.squeeze()
        return y0 + J


class Gaussian2D(IFunction1D):

    def init(self):
        self.declareParameter("y0", 0.0)
        self.declareParameter("A", 1.0)
        self.declareParameter("sig_x", 5.0)
        self.declareParameter("sig_y", 5.0)

    def function1D(self, xvals):
        # xvals is a 1D numpy array that contains the X values for the defined fitting range.
        y0 = self.getParameterValue("y0")
        A = self.getParameterValue("A")
        sig1 = self.getParameterValue("sig_x")
        sig2 = self.getParameterValue("sig_y")

        theta = np.linspace(0, np.pi, 300)[:, np.newaxis]
        y = xvals[np.newaxis, :]

        sigTH = np.sqrt(
            sig1**2 * np.cos(theta) ** 2 + sig2**2 * np.sin(theta) ** 2
        )
        jp = np.exp(-(y**2) / (2 * sigTH**2)) / (2.506628 * sigTH)
        jp *= np.sin(theta)

        JBest = np.trapz(jp, x=theta, axis=0)
        JBest /= np.abs(np.trapz(JBest, x=y))
        JBest *= A
        return y0 + JBest


class DoubleWell(IFunction1D):
    def init(self):
        self.declareParameter("y0", 0.0)
        self.declareParameter("A", 1.0)
        self.declareParameter("d", 1.0)
        self.declareParameter("R", 1.0)
        self.declareParameter("sig_x", 5.0)
        self.declareParameter("sig_y", 5.0)

    def function1D(self, xvals):
        y0 = self.getParameterValue("y0") 
        A = self.getParameterValue("A") 
        d = self.getParameterValue("d") 
        R = self.getParameterValue("R") 
        sig1 = self.getParameterValue("sig_x") 
        sig2 = self.getParameterValue("sig_y") 


        theta = np.linspace(0, np.pi, 300)[
            :, np.newaxis
        ]  # 300 points seem like a good estimate for ~10 examples
        y = xvals[np.newaxis, :]

        sigTH = np.sqrt(
            sig1**2 * np.cos(theta) ** 2 + sig2**2 * np.sin(theta) ** 2
        )
        alpha = 2 * (d * sig2 * sig1 * np.sin(theta) / sigTH) ** 2
        beta = (2 * sig1**2 * d * np.cos(theta) / sigTH**2) * y
        denom = (
            2.506628
            * sigTH
            * (1 + R**2 + 2 * R * np.exp(-2 * d**2 * sig1**2))
        )
        jp = (
            np.exp(-(y**2) / (2 * sigTH**2))
            * (1 + R**2 + 2 * R * np.exp(-alpha) * np.cos(beta))
            / denom
        )
        jp *= np.sin(theta)

        JBest = np.trapz(jp, x=theta, axis=0)
        JBest /= np.abs(np.trapz(JBest, x=y))
        JBest *= A
        return y0 + JBest

# Register with Mantid
FunctionFactory.subscribe(Gaussian3D)
FunctionFactory.subscribe(Gaussian2D)
FunctionFactory.subscribe(DoubleWell)
