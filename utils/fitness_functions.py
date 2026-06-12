import math
import time as tm


def fitness_fellenius(chromosome, ga_parameters, soil_parameters):
    normalized_individual = ga_parameters.normalize_chromosome(chromosome)

    soil_parameters.set_circle_surface(
        normalized_individual[:2], normalized_individual[2]
    )

    # 🚩 Essa parte ainda precisa ser alterada no código para cada novos parâmetros de solo.
    d_center_foot = math.sqrt(
        (soil_parameters.circle_center[0] - 13) ** 2
        + (soil_parameters.circle_center[1] - 0) ** 2
    )

    if soil_parameters.intersections == 0:
        viability = False

        return viability, 1000 + d_center_foot, d_center_foot
    else:
        viability = True

    d_inter_foot = math.sqrt(
        (soil_parameters.intersections[-1][0] - 13) ** 2
        + (soil_parameters.intersections[-1][1] - 0) ** 2
    )

    return viability, soil_parameters.safety_factor + d_inter_foot, d_inter_foot


def fitness_rastrigin(chromosome, ga_parameters):
    x = ga_parameters.normalize_chromosome(chromosome)
    a = 10
    n = len(x)

    fitness = (a * n) + sum(
        [(x[i] ** 2 - a * math.cos(2 * math.pi * x[i])) for i in range(n)]
    )
    viable = True

    return viable, fitness
