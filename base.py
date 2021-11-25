import functools
import math
from random import random, randint
from operator import add


def get_individual(length, min_value, max_value):
    return [randint(min_value, max_value) for x in range(length)]


def get_population(count, length, min_value, max_value):
    return [get_individual(length, min_value, max_value) for x in range(count)]


def get_fitness(individual, target_sum):
    return abs(target_sum - sum(individual))


def get_best_fitness(population, target_sum):
    return min([get_fitness(x, target_sum) for x in population])


def get_average_population_fitness(population, target_sum):
    fitness_sum = sum(get_fitness(x, target_sum) for x in population)
    return fitness_sum / len(population)


def mutate_population(population, mutation_chance, individual_min_value, individual_max_value):
    for i in range(len(population)):
        if mutation_chance > random():
            population[i][randint(0, len(population[i]) - 1)] = randint(individual_min_value, individual_max_value)
            # population[i] = [randint(individual_min_value, individual_max_value) for x in population[i]]

    return population


def evolve_population(population, target_sum, individual_min_value, individual_max_value, retain=.2, random_select=.05, mutate=.01):
    # Sort individuals based on their fitness
    graded = [(get_fitness(x, target_sum), x) for x in population]
    graded = [x[1] for x in sorted(graded)]

    # Only keep the retain% best individuals
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # Randomly add inferior individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # Crossover from parents until the population is the same length as before
    parents_length = len(parents)
    desired_children_length = len(population) - parents_length
    children = []

    while len(children) < desired_children_length:
        father = randint(0, parents_length - 1)
        mother = randint(0, parents_length - 1)

        if father != mother:
            child = [parents[father][i] if .5 > random() else parents[mother][i] for i in range(len(parents[father]))]
            children.append(child)

    # Add newly created children to the population
    parents.extend(children)

    # Mutate the population
    parents = mutate_population(parents, mutate, individual_min_value, individual_max_value)

    return parents


# Prepare data
target_sum = 371000
individual_min_value = 0
individual_max_value = 100000
population = get_population(100, 5, individual_min_value, individual_max_value)
fitness_history = [get_average_population_fitness(population, target_sum)]

current_iteration = 0
current_fitness = math.inf
max_iterations = 1000

# Evolve the population until the average fitness is satisfying or a max number of iterations is reached
while current_fitness != 0 and current_iteration < max_iterations:
    current_iteration += 1

    population = evolve_population(population, target_sum, individual_min_value, individual_max_value)

    current_fitness = get_average_population_fitness(population, target_sum)
    fitness_history.append(current_fitness)

for fitness in fitness_history:
    print(fitness)

print("Max : ", get_best_fitness(population, target_sum))
