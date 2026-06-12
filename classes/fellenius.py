import math
import sys

import numpy as np
import sympy as sp

"""📌
   - Na parte do imput verificar erro caso o usuário coloque 2 pontos iguais ou xi < xi-1 em slope_points;
"""

x, y = sp.symbols("x, y")


class Fellenius:

    def __init__(
        self,
        slope_points,
        number_slices,
        soil_specific_weight,
        soil_cohesion,
        soil_friction_angle,
    ):

        self.slope_points = slope_points
        self.number_slices = number_slices
        self.soil_specific_weight = soil_specific_weight
        self.soil_cohesion = soil_cohesion
        self.soil_friction_angle = soil_friction_angle
        self.slope_segments = self.define_slope_surface()
        self.inicializado = False

    def __getattr__(self, atribute):

        return f"\nO atributo '{atribute}' ainda não foi gerado.\nUse '.set_circle_surface()' para gerá-lo\n"

    def define_slope_surface(self):
        slope_segments = []

        for i in range(len(self.slope_points) - 1):
            x0, y0 = self.slope_points[i]
            x1, y1 = self.slope_points[i + 1]

            dx = x1 - x0
            dy = y1 - y0

            segment = {}

            if abs(dx) < 1e-9:
                segment["type"] = "vertical"
                segment["x_const"] = x0
            elif abs(dy) < 1e-9:
                segment["type"] = "horizontal"
                segment["y_const"] = y0

            else:
                m = dy / dx
                b = y0 - m * x0

                segment["type"] = "regular"
                segment["m"] = m
                segment["b"] = b

            segment["x0"] = x0
            segment["y0"] = y0
            segment["x1"] = x1
            segment["y1"] = y1

            segment["x_min"] = min(x0, x1) - 1e-9
            segment["x_max"] = max(x0, x1) + 1e-9
            segment["y_min"] = min(y0, y1) - 1e-9
            segment["y_max"] = max(y0, y1) + 1e-9

            slope_segments.append(segment)

        return slope_segments

    def line_circle_intersections(self, segment, cx, cy, r):
        intersections = []
        eps = 1e-9

        if segment["type"] == "vertical":
            x = segment["x_const"]
            inside = r**2 - (x - cx) ** 2  # MODIFICADO;

            if inside < -eps:  # FORA DO CÍRCULO;
                return []

            if abs(inside) <= eps:  # TANGENTE AO CÍRCULO;
                y = cy

                if segment["y_min"] <= y <= segment["y_max"]:
                    intersections.append((x, y))

                    return intersections

            root = math.sqrt(inside)
            y1 = cy + root
            y2 = cy - root

            if segment["y_min"] <= y1 <= segment["y_max"]:
                intersections.append((x, y1))
            if segment["y_min"] <= y2 <= segment["y_max"]:
                intersections.append((x, y2))

            return intersections

        if segment["type"] == "horizontal":
            y = segment["y_const"]
            inside = r**2 - (y - cy) ** 2

            if inside < -eps:
                return []

            if abs(inside) <= eps:
                x = cx

                if segment["x_min"] <= x <= segment["x_max"]:
                    intersections.append((x, y))

                return intersections

            root = math.sqrt(inside)
            x1 = cx + root
            x2 = cx - root

            if segment["x_min"] <= x1 <= segment["x_max"]:
                intersections.append((x1, y))
            if segment["x_min"] <= x2 <= segment["x_max"]:
                intersections.append((x2, y))

            return intersections

        m = segment["m"]
        b = segment["b"]

        #  Ax^2 + Bc*x + C = 0

        a = 1 + m**2
        bc = -2 * cx + 2 * m * (b - cy)
        c = cx**2 + (b - cy) ** 2 - r**2

        delta = bc * bc - 4 * a * c

        if delta < -eps:
            return []

        if abs(delta) <= eps:
            x = -bc / (2 * a)
            if segment["x_min"] <= x <= segment["x_max"]:
                y = m * x + b
                intersections.append((x, y))

            return intersections

        root_delta = math.sqrt(delta)
        x1 = (-bc + root_delta) / (2 * a)
        x2 = (-bc - root_delta) / (2 * a)

        if segment["x_min"] <= x1 <= segment["x_max"]:
            y1 = m * x1 + b
            intersections.append((x1, y1))

        if segment["x_min"] <= x2 <= segment["x_max"]:
            y2 = m * x2 + b
            intersections.append((x2, y2))

        return intersections

    def set_circle_surface(self, circle_center, circle_radius):

        self.circle_center = circle_center
        self.circle_radius = circle_radius

        # self.circle_equation = (x - self.circle_center[0]) ** 2 + \
        #     (y - self.circle_center[1]) ** 2 - self.circle_radius ** 2

        self.intersections = self.find_intersections_slope_and_circle(
            self.slope_segments
        )

        self.inicializado = True

        if self.intersections == 0:
            return

        v1, v2, v3, v4, v5, v6, v7 = self.define_slice_properties(
            self.intersections, self.slope_segments
        )
        self.slice_width = v1
        self.slice_x = v2
        self.slice_center_x = v3
        self.slice_height = v4
        self.slice_area = v5
        self.slice_center_angle = v6
        self.slice_base_length = v7
        self.slice_total_area = sum(self.slice_area)
        self.safety_factor = self.fellenius_safety_factor(
            self.slice_area, self.slice_center_angle, self.slice_base_length
        )

    def find_intersections_slope_and_circle(self, slope_segments):
        cx = self.circle_center[0]
        cy = self.circle_center[1]
        r = self.circle_radius

        all_intersections = []
        eps = 1e-7

        for segment in slope_segments:
            intersections = self.line_circle_intersections(segment, cx, cy, r)

            for intersection in intersections:
                already = False

                for i in all_intersections:
                    if (
                        abs(intersection[0] - i[0]) < eps
                        and abs(intersection[1] - i[1]) < eps
                    ):
                        already = True
                        break

                if not already:
                    all_intersections.append(intersection)

        if len(all_intersections) != 2:
            return 0

        all_intersections.sort(key=lambda t: t[0])

        return all_intersections

    def define_slice_properties(self, intersections, slope_segments):

        if intersections == 0:
            return

        x_initial, x_final = intersections[0][0], intersections[1][0]

        slice_width = (x_final - x_initial) / self.number_slices
        slice_x = np.linspace(x_initial, x_final, self.number_slices + 1)
        slice_center_x = (slice_x[:-1] + slice_x[1:]) / 2

        cx, cy = self.circle_center
        r = self.circle_radius

        inside = (r**2) - (slice_x - cx) ** 2
        inside[inside < 0] = 0
        circle_y = cy - np.sqrt(inside)
        circle_center_y = cy - np.sqrt((r * r) - (slice_center_x - cx) ** 2)

        slope_y = np.zeros_like(slice_x)

        for segment in slope_segments:
            m = segment.get("m", None)
            b = segment.get("b", None)

            mask = (slice_x >= segment["x_min"]) & (slice_x <= segment["x_max"])

            if segment["type"] == "vertical":
                x0 = segment["x_const"]
                slope_y[mask] = segment["y0"]
            elif segment["type"] == "horizontal":
                slope_y[mask] = segment["y_const"]
            else:
                slope_y[mask] = m * slice_x[mask] + b

        slice_height = slope_y - circle_y
        # slice_height = np.concatenate(([0], slice_height, [0]))

        slice_area = (slice_height[:-1] + slice_height[1:]) * slice_width * 0.5

        m_ang = (cy - circle_center_y) / (cx - slice_center_x)

        slice_center_angle = 90 - abs(np.degrees(np.arctan(m_ang)))

        slice_base_length = slice_width / np.cos(np.deg2rad(slice_center_angle))

        return (
            slice_width,
            slice_x.tolist(),
            slice_center_x.tolist(),
            slice_height.tolist(),
            slice_area.tolist(),
            slice_center_angle.tolist(),
            slice_base_length.tolist(),
        )

    def fellenius_safety_factor(
        self, slice_area, slice_center_angle, slice_base_length
    ):
        # Esse método servirá como função de aptidão para o algoritimo genético;

        slice_area = np.array(slice_area)
        slice_center_angle = np.array(slice_center_angle)
        slice_base_length = np.array(slice_base_length)

        t1 = np.sum(self.soil_cohesion * slice_base_length)
        t2 = np.sum(
            slice_area
            * self.soil_specific_weight
            * np.cos(np.radians(slice_center_angle))
            * np.tan(np.radians(self.soil_friction_angle))
        )
        t3 = np.sum(
            slice_area
            * self.soil_specific_weight
            * np.sin(np.radians(slice_center_angle))
        )

        safety_factor = (t1 + t2) / t3

        return safety_factor
