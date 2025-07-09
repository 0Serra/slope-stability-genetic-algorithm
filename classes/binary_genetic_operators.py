import random


class BinaryGeneticOperators():  # Objeto criado será o conjunto dos parâmetros referentes às variáveis de projeto;

    def __init__(self, variables_number, variables_length, variables_range, population_length, elitism_rate):
        self.variables_number = variables_number
        self.variables_length = variables_length
        self.genes_number = variables_number * variables_length
        self.variables_range = variables_range
        self.population_length = population_length
        self.elite_number = (elitism_rate / 100) * population_length

    def create_chromosome(self):

        return ''.join(str(random.choice([0, 1])) for _ in range(self.genes_number))

    def normalize_chromosome(self, chromosome):
        normalized_chromosome = []
        normalized_variables = 0
        separate_chromosome = [chromosome[i:i+self.variables_length]
                               for i in range(0, len(chromosome), self.variables_length)]

        for (initial_range, final_range), variable in self.variables_range:

            for _ in range(variable):
                int_value = int(
                    separate_chromosome[normalized_variables - 1], 2)
                normalized_value = initial_range + ((final_range - initial_range) /
                                                    (2 ** self.variables_length - 1)) * int_value
                normalized_chromosome.append(normalized_value)
                normalized_variables += 1

        return normalized_chromosome

    def create_population(self):

        return [self.create_chromosome() for _ in range(self.population_length)]

    def evaluate_chromosome(self, chromosome, fitness_function, *args):
        viable, fitness = fitness_function(chromosome, *args)
        return viable, fitness

    def tournament(self, chromosome1, chromosome2, fitness_function, *args):
        viable1, fitness1 = self.evaluate_chromosome(
            chromosome1, fitness_function, *args)
        viable2, fitness2 = self.evaluate_chromosome(
            chromosome2, fitness_function, *args)

        if viable1 and not viable2:
            return chromosome1
        elif viable2 and not viable1:
            return chromosome2
        elif fitness1 <= fitness2:
            return chromosome1
        else:
            return chromosome2

    # def select_reproduction(self, population, fitness_function, *args):
    #     index = [i for i in range(len(population)) for _ in range(2)]
    #     random.shuffle(index)
    #     selected = []

    #     for i in range(0, len(index) - 2 * self.elite_number, 2):
    #         best = self.tournament(
    #             index[i], index[i + 1], fitness_function, *args)
    #         selected.append(best)

    #     return selected

    def evaluation_and_selection(self, population, fitness_function, *args):
        evaluated_population = []

        for chromosome in population:
            viability, fitness = self.evaluate_chromosome(
                chromosome, fitness_function, *args
            )
            evaluated_population.append([chromosome, viability, fitness])

        evaluated_population.sort(key=lambda item: (not item[1], item[2]))

        elite = [i[0] for i in evaluated_population[:self.elite_number]]
        selected = [i[0] for i in evaluated_population[:len(
            population) - self.elite_number]]

        return evaluated_population, elite, selected

    def crossover(self, parents, cuts):
        cut_points = sorted(random.sample(range(1, len(parents[0])), cuts))
        offspring = ""
        start = 0

        for i in range(len(cut_points) + 1):
            end = cut_points[i] if i < len(cut_points) else len(parents[0])
            donor = 0 if i % 2 == 0 else 1
            offspring += parents[donor][start:end]
            start = end

        return offspring

    # def generate_new_population(self, selected, cuts):
    #     new_population = []

    #     if len(selected) < 2:
    #         return selected

    #     while len(new_population) < len(selected):
    #         parents = random.sample(selected, 2)
    #         offspring = self.crossover(parents, cuts)
    #         new_population.append(offspring)

    #     return new_population

    def mutation(self, chromosome, rate):

        genes = list(chromosome)

        for i in range(len(genes)):
            if random.random() < rate:
                genes[i] = '1' if genes[i] == '0' else '0'

        mutated_chromosome = ''.join(genes)

        return mutated_chromosome
