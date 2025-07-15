from classes import BinaryGeneticOperators, Fellenius
from utils import fitness_fellenius, fitness_rastrigin, progress_bar

slope_points = [[0, 20], [2, 18], [8, 16], [10, 12], [16, 8], [26, 6], [32, 0]]
number_slices = 4
soil_specific_weight = 19
soil_cohesion = 25
soil_friction_angle = 18

# variables_number = 2
# variables_length = 100
# variables_range = [((-5.12, 5.12), variables_number)]
# population_length = 80
# generations = 100
# mutation_rate = 1
# elitism_rate = 10

variables_number = 3
variables_length = 8
variables_range = [((0, 32), 1), ((0, 20), 1), ((0, 32), 1)]
population_length = 20
generations = 30
mutation_rate = 1
elitism_rate = 10

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
    # print(
    #     f"\nVar: {ga_parameters.normalize_chromosome(evaluated_population[0][0])}")
    # print(f"Viavel: {evaluated_population[0][1]}")
    # print(f"Fitness: {evaluated_population[0][2]}")

    new_population = elite + ga_parameters.generate_new_population(selected, 1)

    current_generation += 1

    progress_bar(current_generation, ga_parameters.generations)

last_population = new_population

print(
    f"\nVar: {ga_parameters.normalize_chromosome(evaluated_population[0][0])}")
print(f"Viavel: {evaluated_population[0][1]}")
print(f"Fitness: {evaluated_population[0][2]}")


# Var: [8.664711632453567, 9.000977517106548, 10.3069403714565]
# Fitness: 7.70386032802755
