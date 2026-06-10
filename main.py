from classes import BinaryGeneticOperators, Fellenius
from utils import fitness_fellenius, fitness_rastrigin, progress_bar
import math as mt


# slope_points = [[0, 5], [10, 5], [15, 10], [20, 10]]
slope_points = [[0, 9], [9, 5.5], [10, 3.5], [13, 0]]
number_slices = 8
soil_specific_weight = 20.71
soil_cohesion = 4.15
soil_friction_angle = 35.05

variables_number = 3
variables_length = 20
# variables_range = [((5, 20), 1), ((5, 15), 1), ((2.5, 10), 1)]
variables_range = [((8, 14), 1), ((0, 10), 1), ((1, 15), 1)]
population_length = 1000
generations = 300
mutation_rate = 1
elitism_rate = 1

soil_parameters = Fellenius(
    slope_points, number_slices,
    soil_specific_weight,
    soil_cohesion,
    soil_friction_angle
)

ga_parameters = BinaryGeneticOperators(
    variables_number,
    variables_length,
    variables_range,
    population_length,
    generations,
    mutation_rate,
    elitism_rate
)

initial_population = ga_parameters.create_population()
current_generation = 1
new_population = initial_population

progress_bar(current_generation, ga_parameters.generations)

while current_generation < ga_parameters.generations:
    evaluated_population, elite, selected = ga_parameters.evaluation_and_selection(
        new_population, fitness_fellenius, ga_parameters, soil_parameters)

    new_population = ga_parameters.generate_new_population(
        selected, 1)

    new_population += elite

    current_generation += 1

    # best_norm = ga_parameters.normalize_chromosome(evaluated_population[0][0])
    # soil_parameters.set_circle_surface([best_norm[0], best_norm[1]], best_norm[2])
    # penalty = mt.sqrt((soil_parameters.intersections[0][0] - 10) ** 2 + (soil_parameters.intersections[0][1] - 5) ** 2)

    # print()
    # print('gen: ', current_generation)
    # print(
    #     f"\nVar: {best_norm}")
    # print(f"Viavel: {evaluated_population[0][1]}")
    # print(
    #     f"Fitness: {(evaluated_population[0][2]) - penalty}")

    progress_bar(current_generation, ga_parameters.generations)

last_population = new_population

best_norm = ga_parameters.normalize_chromosome(evaluated_population[0][0])
soil_parameters.set_circle_surface([best_norm[0], best_norm[1]], best_norm[2])
# penalty = mt.sqrt((soil_parameters.intersections[0][0] - 10) ** 2 + (soil_parameters.intersections[0][1] - 5) ** 2)
penalty = evaluated_population[0][3]

print()
print(
    f"\nVar: {best_norm}")
print(f"Viavel: {evaluated_population[0][1]}")
print(
    f"Fitness: {(evaluated_population[0][2]) - penalty}")
# print(
#     f"Fitness: {(evaluated_population[0][2])}")
print(f"Larguras:{soil_parameters.slice_base_length}")
print(f"Áreas:{soil_parameters.slice_area}")
print(f"Ângulos:{soil_parameters.slice_center_angle}")