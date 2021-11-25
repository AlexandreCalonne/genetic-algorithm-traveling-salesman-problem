import operator
import random
import matplotlib.pyplot as plot
import time

from City import City
from Fitness import Fitness


def create_individual(genes_pool):
    """
    Creates a new individual by shuffling the genes pool.
    """
    route = random.sample(genes_pool, len(genes_pool))

    return route


def get_initial_population(population_size, genes_pool):
    """
    Create a new population of individuals from the genes pool.
    """
    population = []
    for i in range(0, population_size):
        population.append(create_individual(genes_pool))

    return population


def rank_routes(population):
    """
    Rank and sort a population based on the fitness of its individuals.
    """
    fitness_result = {i: Fitness(x).route_fitness() for i, x in enumerate(population)}

    return sorted(fitness_result.items(), key=operator.itemgetter(1), reverse=True)


def selection(ranked_population, elites_size):
    """
    Perform a roulette wheel selection with elitism to retrieve the indices of the individuals for the mutation table.

    Elitism: A defined number of individuals are selected directly based on their fitness.

    Roulette wheel selection: The higher the fitness of the remaining individuals,
    the higher the probability is for them to be selected.
    """
    selection_result = []

    # Elitism selection
    for i in range(elites_size):
        selection_result.append(ranked_population[i][0])

    # Roulette wheel selection
    # Disabled because crossover is not as precise with this selection at the moment.
    """
    data_frame = pandas.DataFrame(numpy.array(ranked_population), columns=["Index", "Fitness"])
    data_frame['cum_sum'] = data_frame.Fitness.cumsum()
    data_frame['cum_perc'] = 100 * data_frame.cum_sum / data_frame.Fitness.sum()

    for i in range(len(ranked_population) - elites_size):
        pick = 100 * random.random()

        for j in range(len(ranked_population)):
            if pick <= data_frame.iat[j, 3]:
                selection_result.append(ranked_population[j][0])
                break
    """

    return selection_result


def get_mating_pool(population, selection_result):
    """
    Gets the mating pool of individuals based on the result of the selection done prior.
    """
    mating_pool = []
    for i in range(len(selection_result)):
        index = selection_result[i]
        mating_pool.append(population[index])

    return mating_pool


def crossover_parents(father, mother):
    """
    Creates a new individual from the genes of its parents with an ordered crossover algorithm,
    since the order of the genes is important for the Traveling Salesman Problem.
    """
    half = len(father) // 2
    child = father[:half] + mother[half:]

    # Slower and useless ?
    """
    gene_a_index = int(random.random() * len(father))
    gene_b_index = int(random.random() * len(father))

    father_part = father[min(gene_a_index, gene_b_index):max(gene_a_index, gene_b_index)]
    mother_part = [item for item in mother if item not in father_part]
    child = father_part + mother_part
    """

    return child


def crossover_population(mating_pool, population_desired_size):
    """
    Crossover the population by creating new individuals from the mating pool to fill the remaining spots
    of the population.
    """
    children = []
    length = len(mating_pool)
    for i in range(population_desired_size - length):
        parents_indices = random.sample(range(0, len(mating_pool)), 2)

        child = crossover_parents(mating_pool[parents_indices[0]], mating_pool[parents_indices[1]])
        children.append(child)

    return children


def mutate_individual(individual, mutation_rate):
    """
    Perform a mutation on an individual based on the mutation rate.
    In this case, a mutation is swapping two genes from the individual.
    """
    for actual_index in range(len(individual)):
        if random.random() < mutation_rate:
            swap_index = int(random.random() * len(individual))

            swap = individual[actual_index]
            individual[actual_index] = individual[swap_index]
            individual[swap_index] = swap

    return individual


def mutate_population(population, mutation_rate):
    """
    Try to mutate every individual from the population.
    """
    mutated_population = []
    for i in range(len(population)):
        mutated_individual = mutate_individual(population[i], mutation_rate)
        mutated_population.append(mutated_individual)

    return mutated_population


def next_generation(current_generation, elites_size, mutation_rate):
    """
    Evolve the generation to a new one.
    """
    ranked_population = rank_routes(current_generation)

    selection_results = selection(ranked_population, elites_size)
    mating_pool = get_mating_pool(current_generation, selection_results)

    mating_pool.extend(crossover_population(mating_pool, len(current_generation)))
    new_generation = mutate_population(mating_pool, mutation_rate)

    return new_generation, ranked_population


def genetic_algorithm(genes_pool, population_size, elites_size, mutation_rate, generations_count):
    """
    Perform the entire genetic algorithm by evolving the population *generations_count* times and
    returning the best individual from the last generation.
    """
    population = get_initial_population(population_size, genes_pool)
    print("Initial distance: " + str(1 / rank_routes(population)[0][1]))

    for i in range(generations_count):
        population = next_generation(population, elites_size, mutation_rate)[0]

    print("Final distance: " + str(1 / rank_routes(population)[0][1]))
    best_route_index = rank_routes(population)[0][0]
    best_route = population[best_route_index]

    return best_route


def genetic_algorithm_plot(genes_pool, population_size, elites_size, mutation_rate, generations_count):
    """
    Perform the entire genetic algorithm by evolving the population *generations_count* times and
    returning the best individual from the last generation.

    Also plots the progress to a graph for better visualization.
    """
    population = get_initial_population(population_size, genes_pool)
    progress = [1 / rank_routes(population)[0][1]]

    for i in range(generations_count):
        evolution = next_generation(population, elites_size, mutation_rate)
        population = evolution[0]

        progress.append(1 / evolution[1][0][1])

        print(i)

    plot.plot(progress)
    plot.ylabel('Distance')
    plot.xlabel('Generation')
    plot.show()


def generate_cities(count):
    """
    Generate a sample of *count* random genes to test the algorithm.
    """
    cities = []
    for i in range(count):
        cities.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

    return cities


genetic_algorithm_plot(genes_pool=generate_cities(25),
                       population_size=200,
                       elites_size=20,
                       mutation_rate=0.02,
                       generations_count=1000)
