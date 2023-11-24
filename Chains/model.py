"""Model of chain swimming."""

from typing import Dict, Type
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from numpy import pi, sin, cos, log, arctan
import pandas as pd

THICKNESS = 0.02
PITCH_NORMAL = 3.37
RADIUS_NORMAL = 0.47
LENGTH = 8.5
ETA = 12e-3 #Pa.s Ficoll not dialized
ASPECT_RATIO = 2 #0.18 or 2 : 0.18 Lighthill, 2 Gray and Hancock
LONG_AXIS = 1.4  #semi long axis of ellipse
SMALL_AXIS = 0.5

def get_A0(n: int,
           eta: float=ETA,
           long_axis: float=LONG_AXIS,
           small_axis: float=SMALL_AXIS) -> float:
    """Calculate the coefficient between force and velocity for a body chain of length n.
    
    F0 = - A0 v"""
    return (4 * pi * eta * long_axis * n) / (log(2 * n * long_axis / small_axis) - 0.5)

def get_D0(n: int,
           eta: float=ETA,
           long_axis: float=LONG_AXIS,
           small_axis: float=SMALL_AXIS) -> float:
    """Calculate the coefficient between torque and rotation velocity for a body chain of length n.
    
    T0 = - D0 Omega"""
    return (16 * pi * eta * long_axis * n * small_axis ** 2 ) / 3

def get_psi(radius: float=RADIUS_NORMAL, pitch: float=PITCH_NORMAL) -> float:
    """Calculate the angle of the ellipse"""
    return arctan(2 * pi * radius / pitch)

def get_kn(eta: float=ETA,
           aspect_ratio:float = ASPECT_RATIO,
           pitch: float= PITCH_NORMAL,
           thickness: float=THICKNESS) -> float:
    """Calculate the Kn coefficient."""
    return 8 * pi * eta / (2 * log(aspect_ratio * pitch / thickness) + 1)

def get_kt(eta: float=ETA,
           aspect_ratio:float = ASPECT_RATIO,
           pitch: float= PITCH_NORMAL,
           thickness: float=THICKNESS) -> float:
    """Calculate the Kt coefficient."""
    return 4 * pi * eta / (2 * log(aspect_ratio * pitch / thickness) - 1)

def get_A_flagella(length: float,
                   kn: float=get_kn(),
                   kt: float=get_kt(),
                   psi: float=get_psi())-> float:
    """Calculate the A coefficient for the flagella of length length.
    
    F = -A v + B w
    T = -B v + D w"""
    return length * (kn * sin(psi) ** 2 + kt * cos(psi) ** 2) / cos(psi)

def get_B_flagella(length: float,
                   pitch: float=PITCH_NORMAL,
                   kn: float=get_kn(),
                   kt: float=get_kt(),
                   psi: float=get_psi())-> float:
    """Calculate the B coefficient for the flagella of a given length."""
    return length * pitch * sin(psi) ** 2 * (kn - kt) / (2 * pi * cos(psi))

def get_D_flagella(length: float,
                   pitch: float=PITCH_NORMAL,
                   kn: float=get_kn(),
                   kt: float=get_kt(),
                   psi: float=get_psi())-> float:
    """Calculate the D coefficient for the flagella of a given length."""
    return length * (pitch / (2 * pi)) ** 2 * (sin(psi) ** 2 / cos(psi) ** 3) * (kn * cos(psi) ** 2 + kt * sin(psi) ** 2)


class Model(ABC):
    """Abstract for model."""
    def __init__(self, n: int) -> None:
        self.n: int = n

    @abstractmethod
    def calculate_body_rotation(self, flagella_rot: float=1) -> float:
        """Get the body rotation velocity from the flagella rotation velocity."""
        pass
    
    @abstractmethod
    def calculate_velocity(self, flagella_rot: float=1) -> float:
        """Get the swimming velocity from the flagella rotation velocity."""
        pass
    
    @abstractmethod
    def calculate_force_body(self, flagella_rot: float=1) -> float:
        """Calculate the force on the body."""
        pass
    
    @abstractmethod
    def calculate_torque_body(self, flagella_rot: float=1) -> float:
        """Calculate the torque on the body."""
        pass

    @abstractmethod
    def calculate_force_flagella(self, flagella_rot: float=1) -> float:
        """Calculate the force on the flagella."""
        pass

    
    @abstractmethod
    def calculate_torque_flagella(self, flagella_rot: float=1) -> float:
        """Calculate the torque on the flagella."""
        pass

    def process(self, flagella_rot: float=1) -> Dict[str, float]:
        """Perform all the calculations."""
        return {"n": self.n,
                "flagella_rot": flagella_rot,
                "body_rot": self.calculate_body_rotation(flagella_rot),
                "velocity": self.calculate_velocity(flagella_rot),
                "F0": self.calculate_force_body(flagella_rot),
                "T0": self.calculate_torque_body(flagella_rot),
                "F": self.calculate_force_flagella(flagella_rot),
                "T": self.calculate_torque_flagella(flagella_rot)}


class NaiveModel(Model):
    def __init__(self, n: int, alpha: float=0) -> None:
        super().__init__(n)
        self.alpha = alpha
        self.A0 = get_A0(n)
        self.D0 = get_D0(n)
        self.L = LENGTH + 2 * alpha * (n - 1) * LONG_AXIS
        self.Af = get_A_flagella(self.L)
        self.Bf = get_B_flagella(self.L)
        self.Df = get_D_flagella(self.L)

    def calculate_body_rotation(self, flagella_rot: float=1) -> float:
        return (self.Df * (self.A0 + self.Af) - self.Bf ** 2) * flagella_rot / (self.D0 * (self.A0 + self.Af))
    
    def calculate_velocity(self, flagella_rot: float = 1) -> float:
        return self.Bf * flagella_rot / (self.A0 + self.Af)

    def calculate_force_body(self, flagella_rot: float = 1) -> float:
        v = self.calculate_velocity(flagella_rot)
        return - self.A0 * v
    
    def calculate_force_flagella(self, flagella_rot: float = 1) -> float:
        v = self.calculate_velocity(flagella_rot)
        return - self.Af * v + self.Bf * flagella_rot
    
    def calculate_torque_body(self, flagella_rot: float = 1) -> float:
        body_rot = self.calculate_body_rotation(flagella_rot)
        return - self.D0 * body_rot
    
    def calculate_torque_flagella(self, flagella_rot: float = 1) -> float:
        v = self.calculate_velocity(flagella_rot)
        return - self.Bf * v + self.Df * flagella_rot


class SimpleInteractionModel(Model):
    """Simple interaction model where we considerer that the flagella around the body will experience a rotation frequency of w + W.
    We do not consider the fluid motion create by the flagella around the body to determine W."""
    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.A0 = get_A0(n)
        self.D0  = get_D0(n)

        around_body_length = 2 * (self.n - 1) * LONG_AXIS
        full_length = LENGTH + around_body_length
        self.At = get_A_flagella(full_length)
        self.Bt = get_B_flagella(full_length)
        self.Bbf = get_B_flagella(around_body_length)
        self.Dt = get_D_flagella(full_length)
        self.Dbf = get_D_flagella(around_body_length)

    def calculate_body_rotation(self, flagella_rot: float=1) -> float:
        """Get the body rotation velocity from the flagella rotation velocity."""
        Asum = self.At + self.A0
        beta = (self.Dt * Asum - self.Bt ** 2) / (Asum * (self.D0 + ((self.Bt * self.Bbf) / Asum) - self.Dbf))
        return beta * flagella_rot
    
    def calculate_velocity(self, flagella_rot: float=1) -> float:
        """Get the swimming velocity from the flagella rotation velocity."""
        body_rot = self.calculate_body_rotation(flagella_rot)
        return (self.Bt * flagella_rot + self.Bbf * body_rot) / (self.At + self.A0)
    
    def calculate_force_body(self, flagella_rot: float=1) -> float:
        """Calculate the force on the body."""
        return - self.A0 * self.calculate_velocity(flagella_rot)

    def calculate_torque_body(self, flagella_rot: float=1) -> float:
        """Calculate the torque on the body."""
        return - self.D0 * self.calculate_body_rotation(flagella_rot)
    
    def calculate_force_flagella(self, flagella_rot: float=1) -> float:
        """Calculate the force on the flagella."""
        v = self.calculate_velocity(flagella_rot)
        body_rot = self.calculate_body_rotation(flagella_rot)
        return - self.At * v + self.Bt * flagella_rot + self.Bbf * body_rot
    
    def calculate_torque_flagella(self, flagella_rot: float=1) -> float:
        """Calculate the torque on the flagella."""
        v = self.calculate_velocity(flagella_rot)
        body_rot = self.calculate_body_rotation(flagella_rot)
        return - self.Bt * v + self.Dbf * body_rot + self.Dt * flagella_rot
    

class InteractionModel2(Model):
    """Simple interaction model where we considerer that the flagella around the body will experience a rotation frequency of w + W."""
    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.A0 = get_A0(n)
        self.D0n  = get_D0(n)
        self.D0n1 = get_D0(n - 1)

        around_body_length = 2 * (self.n - 1) * LONG_AXIS
        full_length = LENGTH + around_body_length
        self.At = get_A_flagella(full_length)
        self.Bt = get_B_flagella(full_length)
        self.Bbf = get_B_flagella(around_body_length)
        self.Dt = get_D_flagella(full_length)
        self.Dbf = get_D_flagella(around_body_length)

    def calculate_body_rotation(self, flagella_rot: float=1) -> float:
        """Get the body rotation velocity from the flagella rotation velocity."""
        Asum = self.At + self.A0
        beta = (self.Dt * Asum - self.Bt ** 2 - Asum * self.D0n1) / (Asum * (self.D0n + ((self.Bt * self.Bbf) / Asum) - self.Dbf)) 
        return beta * flagella_rot
    
    def calculate_velocity(self, flagella_rot: float=1) -> float:
        """Get the swimming velocity from the flagella rotation velocity."""
        body_rot = self.calculate_body_rotation(flagella_rot)
        return (self.Bt * flagella_rot + self.Bbf * body_rot) / (self.At + self.A0)
    
    def calculate_force_body(self, flagella_rot: float=1) -> float:
        """Calculate the force on the body."""
        return - self.A0 * self.calculate_velocity(flagella_rot)

    def calculate_torque_body(self, flagella_rot: float=1) -> float:
        """Calculate the torque on the body."""
        return - self.D0n * self.calculate_body_rotation(flagella_rot) - self.D0n1 * flagella_rot
    
    def calculate_force_flagella(self, flagella_rot: float=1) -> float:
        """Calculate the force on the flagella."""
        v = self.calculate_velocity(flagella_rot)
        body_rot = self.calculate_body_rotation(flagella_rot)
        return - self.At * v + self.Bt * flagella_rot + self.Bbf * body_rot
    
    def calculate_torque_flagella(self, flagella_rot: float=1) -> float:
        """Calculate the torque on the flagella."""
        v = self.calculate_velocity(flagella_rot)
        body_rot = self.calculate_body_rotation(flagella_rot)
        return - self.Bt * v + self.Dbf * body_rot + self.Dt * flagella_rot


class RunnerModel:
    """Run the model for different chain lengths and plots."""
    def __init__(self, model: Type[Model], max_length: int) -> None:
        self.model = model
        self.max_length = max_length
        self.range = range(1, max_length)

    def run_serie(self):
        """Run a serie of calculation from the model for different chain lengths."""
        data = pd.DataFrame()
        for n in self.range:
            modeln = self.model(n)
            res = pd.DataFrame(modeln.process(), index=[n])
            if len(data) == 0:
                data = pd.DataFrame(res)
            else:
                data = pd.concat([data, res])
        data["Normalized_vel"] = data["velocity"] / data.loc[1, "velocity"]
        self.data = data
        return data

    def plot(self, fig_folder: str):
        """Plot the data from the model as a function of chain length."""
        plt.figure()
        plt.plot(self.data["n"], self.data["Normalized_vel"] , "bo")
        plt.ylabel("$V / V_{single}$")
        plt.xlabel("Chain length")
        plt.savefig(os.path.join(fig_folder, "norm_vel.png"))
        plt.close()

        plt.figure()
        plt.plot(self.data["n"], self.data["F"], "o", label="$F_f$")
        plt.plot(self.data["n"], self.data["F0"], "o", label="$F_0$")
        plt.plot(self.data["n"], self.data["F0"] + self.data["F"], "o", label="$F_0 + F_f$")
        plt.xlabel("Chain length")
        plt.ylabel("Force")
        plt.legend()
        plt.savefig(os.path.join(fig_folder, "force.png"))
        plt.close()

        plt.figure()
        plt.plot(self.data["n"], self.data["T0"], "o", label="$T_0$")
        plt.plot(self.data["n"], self.data["T"], "o", label="$T_f$")
        plt.plot(self.data["n"], self.data["T0"] + self.data["T"], "o", label="$T_0 + T_f$")
        plt.xlabel("Chain length")
        plt.ylabel("Torque") 
        plt.legend() 
        plt.savefig(os.path.join(fig_folder, "torque.png"))
        plt.close()   

        plt.figure()
        plt.plot(self.data["n"], self.data["body_rot"], "o")
        plt.xlabel("Chain length")
        plt.ylabel("Body rotation")
        plt.savefig(os.path.join(fig_folder, "body_rot.png"))
        plt.close()


if __name__=="__main__":
    runner = RunnerModel(NaiveModel, 8)
    data = runner.run_serie()
    runner.plot("/Users/sintes/Desktop/NASGuillaume/Chains/Figures/Models/NaiveAlpha0")

    runner = RunnerModel(SimpleInteractionModel, 8)
    data = runner.run_serie()
    runner.plot("/Users/sintes/Desktop/NASGuillaume/Chains/Figures/Models/Interaction1")

    runner = RunnerModel(InteractionModel2, 8)
    data2 = runner.run_serie()
    runner.plot("/Users/sintes/Desktop/NASGuillaume/Chains/Figures/Models/Interaction2")    
