import math
import time as tm


def fitness_fellenius(chromosome, ga_parameters, soil_parameters):
    normalized_individual = ga_parameters.normalize_chromosome(
        chromosome)

    soil_parameters.set_circle_surface(
        normalized_individual[:2], normalized_individual[2])

    if soil_parameters.intersections == 0:
        viability = False

        return viability, 1000
    else:
        viability = True

        return viability, soil_parameters.safety_factor


def fitness_rastrigin(chromosome, ga_parameters):
    x = ga_parameters.normalize_chromosome(
        chromosome)
    a = 10
    n = len(x)

    fitness = (
        (a * n) + sum([(x[i] ** 2 - a * math.cos(2 * math.pi * x[i])) for i in range(n)]))
    viable = True

    return viable, fitness
