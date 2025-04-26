import math
import sys

import numpy as np
import sympy as sp

'''📌
   - Na parte do imput verificar erro caso o usuário coloque 2 pontos iguais ou xi < xi-1 em slope_points;
'''

x, y = sp.symbols("x, y")


class Fellenius():

    def __init__(self, slope_points, number_slices, soil_specific_weight, soil_cohesion, soil_friction_angle):

        self.slope_points = slope_points
        self.number_slices = number_slices
        self.soil_specific_weight = soil_specific_weight
        self.soil_cohesion = soil_cohesion
        self.soil_friction_angle = soil_friction_angle

    def define_slope_surface(self):
        slope_segments = []

        for i in range(len(self.slope_points) - 1):
            segment = [self.slope_points[i]] + [self.slope_points[i + 1]]
            slope_segments.append([j for i in segment for j in i])

        return slope_segments

    def define_circle_equation(self, circle_center, circle_radius):

        self.circle_center = circle_center
        self.circle_radius = circle_radius

        circle_equation = (x - self.circle_center[0]) ** 2 + \
            (y - self.circle_center[1]) ** 2 - self.circle_radius ** 2

        return circle_equation

    def find_intersections_slope_and_circle(self, slope_segments, circle_equation):
        intersections = []

        for i in slope_segments:
            nm, dm = i[3] - i[1], i[2] - i[0]

            # ⚠ Evitar os erros de atribuição dos _slope_points;
            if nm == 0:
                i.append(i[1])
                roots = sp.solve(circle_equation.subs(y, i[1]))
            elif dm == 0:
                i.append(min(i[1], i[3]))
                roots = sp.solve(circle_equation.subs(x, i[0]))
            else:
                m = nm / dm
                segment_equation = m * (x - i[0]) + i[1]
                i.append(segment_equation)
                roots = sp.solve(circle_equation.subs(y, segment_equation))

            real_roots = [i for i in roots if i.is_real]

            if len(real_roots) >= 1:
                for j in real_roots:
                    if nm == 0 and (i[0] <= j <= i[2] or i[0] >= j >= i[2]):
                        intersection_x = j
                        intersection_y = i[1]
                        intersections.append((intersection_x, intersection_y))
                    elif dm == 0 and (i[1] <= j <= i[3] or i[1] >= j >= i[3]):
                        intersection_x = i[0]
                        intersection_y = j
                        intersections.append((intersection_x, intersection_y))
                    elif nm != 0 and dm != 0 and (i[0] <= j <= i[2] or i[0] >= j >= i[2]):
                        intersection_x = j
                        intersection_y = segment_equation.subs(x, j)
                        intersections.append((intersection_x, intersection_y))

        intersections = list(dict.fromkeys(
            [(float(i), float(j)) for i, j in intersections]))

        # Possível restrição;
        if len(intersections) != 2:
            print("A quantidade de interseções encontradas é inválida!")
            sys.exit()
        else:
            return intersections

    def define_slice_properties(self, intersections, slope_segments, circle_equation):
        x_initial, x_final = intersections[0][0], intersections[1][0]
        slice_height = []
        slice_area = []
        slice_center_angle = []
        slice_base_length = []

        slice_width = (x_final - x_initial) / self.number_slices
        slice_x = np.linspace(x_initial, x_final, self.number_slices + 1)
        slice_center_x = (slice_x[:-1] + slice_x[1:]) / 2

        for i in slice_x[1:-1]:
            circle_y = min(sp.solve(circle_equation.subs(x, i)))

            for j in slope_segments:
                if j[0] <= i <= j[2] or j[0] >= i >= j[2]:
                    if j[1] == j[3] or j[0] == j[2]:
                        slope_y = j[4]
                    else:
                        slope_y = j[4].subs(x, i)

            # ⚠ Evitar os erros de atribuição dos _slope_points;
            slice_height.append(slope_y - circle_y)

        slice_height = [0] + slice_height + [0]

        for i in range(len(slice_height) - 1):
            slice_area.append(
                (slice_height[i] + slice_height[i + 1]) * slice_width / 2)

        m_angular = (self.circle_center[1] - np.array([np.float64(min(sp.solve(circle_equation.subs(x, i))))
                                                       for i in slice_center_x])) / (self.circle_center[0] - slice_center_x)

        for i in m_angular:
            if i > 0:
                slice_center_angle.append(
                    90 - math.degrees(math.atan(float(i))))
            elif i < 0:
                slice_center_angle.append(
                    (90 + math.degrees(math.atan(float(i)))) * -1)
            elif i == 0:
                slice_center_angle.append(0)
            else:
                slice_center_angle.append(90)

            slice_base_length.append(
                slice_width / math.cos(math.radians(slice_center_angle[-1])))

        slice_x = [float(i) for i in slice_x]
        slice_center_x = [float(i) for i in slice_center_x]

        # slice_base_length também pode ser calculado diretamente no fator de segurança;
        return slice_width, slice_x, slice_center_x, slice_height, slice_area, slice_center_angle, slice_base_length

    def fellenius_safety_factor(self, slice_area, slice_center_angle, base_length):
        # Esse método servirá como função de aptidão para o algoritimo genético;

        slice_area = np.array(slice_area)
        slice_center_angle = np.array(slice_center_angle)
        base_length = np.array(base_length)

        t1 = np.sum(self.soil_cohesion * base_length)
        t2 = np.sum(slice_area * self.soil_specific_weight *
                    np.cos(np.radians(slice_center_angle)) * np.tan(np.radians(self.soil_friction_angle)))
        t3 = np.sum(slice_area * self.soil_specific_weight *
                    np.sin(np.radians(slice_center_angle)))

        safety_factor = (t1 + t2) / t3

        return safety_factor
