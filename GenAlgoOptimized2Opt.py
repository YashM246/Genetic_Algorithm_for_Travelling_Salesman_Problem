import random
import math
import numpy as np
import time

class GeneticAlgoTSP:
    
    def __init__(self, filepath):
        # Where the file is stored
        self.filepath = filepath
        # Coordinates of cities
        self.cities = []
        # Population size
        self.population_size = 100
        # Number of Elite Parents
        self.elite_size = 20
        # Mutation Rate
        self.mutation_rate = 0.01
        # Number of iterations
        self.iterations = 5
        # Tournament Size  
        self.tournament_size = 5
        # Expensive to apply 2 opt everytime so Apply 2-opt every 2 generations       
        self.two_opt_frequency = 2
        # Limit 2-opt iterations
        self.max_two_opt_iterations = 300 
        # Window size for partial 2-opt 
        self.two_opt_window = 50  

        # Take input from file
        self.read_input()
        # Compute Distance Matrix for fast lookup
        self.distance_matrix = self.create_dist_matrix()

        # Create Initial Population
        population = self.initial_population()

        # Run for 5 iterations
        for i in range(self.iterations):
            population = self.next_generation(population)
            if i % self.two_opt_frequency == 0:
                best_route = min(population, key=lambda x: self.path_length(x))
                # Apply 2-opt optimization to the best individual in each generation
                optimized_route = self.two_opt(best_route)
                population[population.index(best_route)] = optimized_route

        best_route = min(population, key=lambda x: self.path_length(x))
        best_length = self.path_length(best_route)
        self.write_output(best_route, best_length)

    def read_input(self):
        with open(self.filepath, "r") as f:
            # First Line denotes the number of cities
            self.num_cities = int(f.readline().strip())
            # x, y, z coordinates
            self.cities = [tuple(map(int, line.strip().split())) for line in f]
        
    def create_dist_matrix(self):
        matrice = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i+1, self.num_cities):
                dist = self.euc_distance(self.cities[i], self.cities[j])
                matrice[i][j] = matrice[j][i] = dist
        return matrice
    
    def euc_distance(self, city1, city2):
        # Euclidean Distance between two cities
        return math.sqrt(sum((c1-c2)**2 for c1, c2 in zip(city1, city2)))

    def initial_population(self):
        # Create random population for first iteration
        return [self.generate_path() for _ in range(self.population_size)]
        
    def generate_path(self):
        # Get random path from list of cities
        return random.sample(range(self.num_cities), self.num_cities)
    
    def path_length(self, path):
        length = sum(self.distance_matrix[path[i]][path[i+1]] for i in range(self.num_cities-1))
        length += self.distance_matrix[path[-1]][path[0]]
        return length

    def tournament_selection(self, population):
        # Select 5 parents from population at random
        tournament = random.sample(population, self.tournament_size)
        # Return the best parent from those 5
        return min(tournament, key=lambda x: self.path_length(x))
        
    def crossover(self, parent1, parent2):
        size = len(parent1)
        # Choose two random points
        # This path will be copied into the child from parent1
        start, end = sorted(random.sample(range(size), 2))
        child = [-1]*size

        child[start:end] = parent1[start:end]

        # Add the remaining cities in the same order as parent2
        remaining = [item for item in parent2 if item not in child]
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining.pop(0)
        return child
        
    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            # Mutate by swapping two random cities
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
        
    def next_generation(self, population):
        new = []
        # Retain the best genetic parents from old population
        sorted_pop = sorted(population, key=lambda x: self.path_length(x))
        new.extend(sorted_pop[:self.elite_size])

        while len(new) < self.population_size:
            # Random Parent Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            # Crossover
            child = self.crossover(parent1, parent2)
            # Mutate the child
            child = self.mutate(child)
            # Add it to new population
            new.append(child)
        return new

    def two_opt(self, route):
        improved = True
        best_route = route
        best_length = self.path_length(route)
        iterations = 0
        
        # Limit the times 2 opt is being used
        while improved and iterations < self.max_two_opt_iterations:
            improved = False
            iterations += 1
            for i in range(1, len(route) - 1):
                for j in range(i + 1, min(i + self.two_opt_window, len(route))):
                    # If cost is less after inverting the edge, keep the change                        
                    if j - i == 1:
                        continue
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1]
                    new_length = self.path_length(new_route)
                    if new_length < best_length:
                        best_route = new_route
                        best_length = new_length
                        improved = True
                        break
                if improved:
                    break
            route = best_route
        
        return best_route
        
    def write_output(self, best, cost):
        with open("output_opt_2OPT.txt", "w") as file:
            file.write(f"{cost}\n")
            for city_index in best:
                city = self.cities[city_index]
                file.write(f"{city[0]} {city[1]} {city[2]}\n")
            first_city = self.cities[best[0]]
            file.write(f"{first_city[0]} {first_city[1]} {first_city[2]}\n")
        
if __name__ == "__main__":
    start_time = time.time()
    GeneticAlgoTSP("input8.txt")
    print(f"Execution time: {time.time() - start_time} seconds")