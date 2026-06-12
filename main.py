from classes import BinaryGeneticOperators, Fellenius, ProbabilisticSlopeAnalysis
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

m = [4.15, 35.05, 20.71]
sd = [0.581, 3.505, 0.4142]
t = 1e-4

soil_parameters = Fellenius(
    slope_points,
    number_slices,
    soil_specific_weight,
    soil_cohesion,
    soil_friction_angle,
)

ga_parameters = BinaryGeneticOperators(
    variables_number,
    variables_length,
    variables_range,
    population_length,
    generations,
    mutation_rate,
    elitism_rate,
)

fosm_parameters = ProbabilisticSlopeAnalysis(m, sd, t)

initial_population = ga_parameters.create_population()
current_generation = 1
new_population = initial_population
data = []

progress_bar(current_generation, ga_parameters.generations)

while current_generation <= ga_parameters.generations:
    evaluated_population, elite, selected = ga_parameters.evaluation_and_selection(
        new_population, fitness_fellenius, ga_parameters, soil_parameters
    )

    new_population = ga_parameters.generate_new_population(selected, 1)

    new_population += elite

    best_norm = ga_parameters.normalize_chromosome(evaluated_population[0][0])

    soil_parameters.set_circle_surface(best_norm[:2], best_norm[2])
    bases = soil_parameters.slice_base_length
    areas = soil_parameters.slice_area
    angles = soil_parameters.slice_center_angle

    fosm_parameters.limit_state_function(m, bases, areas, angles)
    pf = fosm_parameters.find_probability_failure()

    data.append(
        [current_generation]
        + best_norm
        + [evaluated_population[0][2] - evaluated_population[0][3]]
        + [pf]
    )

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

with open(
    r"C:\Users\Lenovo\Desktop\ARTIGO\Resultados.txt", "w", encoding="utf-8"
) as file:
    # Cabeçalho
    file.write("GERAÇÃO X Y R FITNESS PF\n")

    # Dados
    for row in data:
        formatted_row = " ".join(map(str, row))
        file.write(formatted_row + "\n")

# penalty = mt.sqrt((soil_parameters.intersections[0][0] - 10) ** 2 + (soil_parameters.intersections[0][1] - 5) ** 2)
penalty = evaluated_population[0][3]

print()
print(f"\nVar: {best_norm}")
print(f"Viavel: {evaluated_population[0][1]}")
print(f"Fitness: {(evaluated_population[0][2]) - penalty}")
print(f"Penalidade: {penalty}")
print(f"Pf: {pf}")
