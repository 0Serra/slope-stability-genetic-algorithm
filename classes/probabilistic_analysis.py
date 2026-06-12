import numpy as np
from scipy.stats import norm


class ProbabilisticSlopeAnalysis:
    def __init__(self, mean, standard_deviation, tolerance):
        self.m = np.array(mean)
        self.sd = np.array(standard_deviation)
        self.tolerance = tolerance

    def limit_state_function(
        self, x, slice_base_length, slice_area, slice_center_angle
    ):
        self.soil_cohesion = x[0]
        self.soil_friction_angle = x[1]
        self.soil_specific_weight = x[2]

        self.slice_base_length = np.array(slice_base_length)
        self.slice_area = np.array(slice_area)
        self.slice_center_angle = np.array(slice_center_angle)

        self.t1 = np.sum(self.soil_cohesion * self.slice_base_length)
        self.t2 = np.sum(
            self.slice_area
            * self.soil_specific_weight
            * np.cos(np.radians(self.slice_center_angle))
            * np.tan(np.radians(self.soil_friction_angle))
        )
        self.t3 = np.sum(
            self.slice_area
            * self.soil_specific_weight
            * np.sin(np.radians(self.slice_center_angle))
        )

        lef = ((self.t1 + self.t2) / self.t3) - 1

        return lef

    def gradient_x(self):
        grad_1 = np.sum(self.slice_base_length) / self.t3
        grad_2 = (
            (
                np.sum(
                    self.slice_area
                    * self.soil_specific_weight
                    * np.cos(np.radians(self.slice_center_angle))
                )
            )
            * (1 + (np.tan(np.radians(self.soil_friction_angle))) ** 2)
            * (np.pi / 180)
            / self.t3
        )
        grad_3 = (
            (
                self.t3
                * np.sum(
                    self.slice_area
                    * np.cos(np.radians(self.slice_center_angle))
                    * np.tan(np.radians(self.soil_friction_angle))
                )
            )
            - (
                (self.t1 + self.t2)
                * np.sum(self.slice_area * np.sin(np.radians(self.slice_center_angle)))
            )
        ) / (self.t3**2)

        return [grad_1, grad_2, grad_3]

    def gradient_y(self):
        jacobian_xy = np.diag(self.sd)

        return jacobian_xy.T @ self.gradient_x()

    def new_y(self, y, x):
        return (
            (
                self.gradient_y() @ y
                - self.limit_state_function(
                    x, self.slice_base_length, self.slice_area, self.slice_center_angle
                )
            )
            / (np.linalg.norm(self.gradient_y()) ** 2)
        ) * self.gradient_y()

    def beta(self, x):
        return np.linalg.norm(x)

    def find_probability_failure(self):
        # x = self.m
        x = [3, 33, 22]
        y = (x - self.m) / self.sd

        beta = self.beta(y)
        stop_condition = self.tolerance + 1

        while stop_condition > self.tolerance:
            g_x = self.limit_state_function(
                x, self.slice_base_length, self.slice_area, self.slice_center_angle
            )
            # g_y = g_x

            # grad_x = self.gradient_x()
            grad_y = self.gradient_y()

            # direction_cosines = grad_y / np.linalg.norm(grad_y)
            # sensitivity = direction_cosines ** 2

            previous_y = y
            y = self.new_y(y, x)
            beta = self.beta(y)

            x = y * self.sd + self.m
            # print(f"X = {x}")
            # print(f"Y = {y}")
            # print(f"Beta = {beta}")
            stop_condition = abs(
                (np.linalg.norm(y) - np.linalg.norm(previous_y)) / np.linalg.norm(y)
            )

        pf = norm.cdf(-beta)

        return pf
