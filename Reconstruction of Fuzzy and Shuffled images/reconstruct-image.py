import blackbox
import random
from deap import base
from deap import creator
from deap import tools

oracle = blackbox.BlackBox("shredded.png", "original.png")

# deap initializer
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual",list, fitness=creator.FitnessMin)

# define functions
def individual(size=128):
    return creator.Individual(random.sample(range(size), size))

def evaluate_ind(ind):
    return oracle.evaluate_solution(ind),

# register functions
toolbox = base.Toolbox()
# register the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, individual)
# register crossover operator
toolbox.register("mate", tools.cxPartialyMatched)
# register mutation operator
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.005)
# register select operator
toolbox.register("select", tools.selTournament, tournsize=3)
# register evaluate
toolbox.register("evaluate", evaluate_ind)


def main():
    # Run 30 times to collect data for the table 1
    NUMBER_OF_RUNS = 30

    fitness_of_30 = []

    for r in range(NUMBER_OF_RUNS):

        # generate randomly the best value for 30 runs 
        #random.seed(30) 


        # input parameters
        N = 500  # generations
        CXPB = 0.9  # crossover probability
        MUTPB = 1   # mutation probability

        # create an initial population of 100 individuals
        pop = toolbox.population(n=100)

        # evaluate each individual in the population
        fitness = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitness):
            ind.fitness.values = fit

        for g in range(N):
            # select the next generation individual
            offspring = toolbox.select(pop, len(pop))
            # clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # apply mutation on the offspring
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitness = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind,fit in zip(invalid_ind, fitness):
                ind.fitness.values = fit

            # replace the entire population with the offspring
            pop[:] = offspring

            # print data for the table 2
            fits = [ind.fitness.values[0] for ind in pop]
            length = len(pop)
            mean = sum(fits) / length
            # calculate standard deviation: https://en.wikipedia.org/wiki/Standard_deviation
            sum_std = sum((x - mean)**2 for x in fits)
            std = abs((sum_std/(length-1))**0.5)
            print(str(g+1)+"\t"+str(min(fits))+"\t"+str(max(fits))+"\t"+str(mean)+"\t"+str(std))

            # collect data to draw the figure 5
            best = tools.selBest(pop, 1)[0]
            print(best.fitness.values[0])

                       
        # select and show the best individual
        final_best_ind = tools.selBest(pop, 1)[0]
        if (NUMBER_OF_RUNS == 1) :
            oracle.show_solution(final_best_ind)
            oracle.show_solution(final_best_ind, "result.png")

        # collect data for table 1
        fitness_of_best = final_best_ind.fitness.values[0]
        fitness_of_30.append((fitness_of_best)) # add values
        print(str(r + 1) + "\t" + str(fitness_of_best))

    # print data for the table 1
    if NUMBER_OF_RUNS > 1:
        for r in range(NUMBER_OF_RUNS) : print(str(r + 1) + "\t" + str(fitness_of_30[r]))
        mean = sum(fitness_of_30) / NUMBER_OF_RUNS
        # calculate standard deviation: 
        sum_std = sum((x - mean)**2 for x in fitness_of_30)
        std = abs((sum_std/(NUMBER_OF_RUNS-1))**0.5)
        print("min \t max \t avg \t std")
        print(str(min(fitness_of_30)) + "\t" + str(max(fitness_of_30)) + "\t" + str(mean) + "\t" + str(std))

if __name__ == "__main__":
    main()