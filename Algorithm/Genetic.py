#!/usr/bin/env python3
"""
Genetic algorithm implemented with DEAP solving the one max problem
(maximising number of 1s in a binary string).

"""
import random
import multiprocessing
import statistics

from deap import creator, base, tools, algorithms
from PIL import Image, ImageDraw, ImageChops

toolbox = base.Toolbox()


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])
    return image


def make_polygon():
    R = random.randint(0, 256)
    G = random.randint(0, 256)
    B = random.randint(0, 256)
    A = random.randint(30, 60)

    x1 = random.randint(10, 190)
    x2 = random.randint(10, 190)
    x3 = random.randint(10, 190)
    y1 = random.randint(10, 190)
    y2 = random.randint(10, 190)
    y3 = random.randint(10, 190)

    return [(R, G, B, A), (x1, y1), (x2, y2), (x3, y3)]


def mutate(solution, indpb):
    if random.random() < 0.75:
        # mutate points
        polygon = random.choice(solution)
        coords = [x for point in polygon[1:] for x in point]
        tools.mutGaussian(coords, 0, 10, indpb)
        coords = [max(0, min(int(x), 200)) for x in coords]
        polygon[1:] = list(zip(coords[::2], coords[1::2]))
    if random.random() < 0.1:
        # add a new polygon
        newbron = make_polygon()
        solution.append(newbron)
    if random.random() < 0.15:
        rgb = []
        polygon = random.choice(solution)
        limit = 2
        for index, colour in enumerate(polygon[0]):
            rgb.append(colour)
            if index == limit:
                break
        tools.mutGaussian(rgb, 0, 50, 1)
    else:
        # reorder polygons
        tools.mutShuffleIndexes(solution, indpb)
    return (solution,)


MAX = 255 * 200 * 200
TARGET = Image.open("cd..TargetImages/8a.png")
TARGET.load()


def evaluate(solution):
    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX,


def run(generations=50, population_size=100, seed=41):
    # random.seed(seed)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox.register("Individual", tools.initRepeat, creator.Individual, make_polygon, n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=100)

    population = toolbox.population(n=population_size)

    #for i in range(generations):
    #   offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    #    fitnesses = toolbox.map(toolbox.evaluate, offspring)
    #    for value, individual in zip(fitnesses, offspring):
    #        individual.fitness.values = value
    #    population = toolbox.select(offspring, len(population))

    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda x: x.fitness.values[0])
    stats.register("avg", statistics.mean)
    stats.register("std", statistics.stdev)

    population, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.5,
                                          ngen=generations, stats=stats, halloffame=hof, verbose=False)

    print(log)
    best = tools.selBest(population, k=1)[0]
    draw(best).save('solution.png')


run(generations=500)

# toolbox.register("evaluate", evaluate)
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", mutate, indpb=0.05)
# toolbox.register("select", tools.selTournament, tournsize=3)


# hof = tools.HallOfFame(3)
# stats = tools.Statistics(lambda x: x.fitness.values[0])
# stats.register("avg", statistics.mean)
# stats.register("std", statistics.stdev)

# population, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=50, verbose=False)
# print(log)
