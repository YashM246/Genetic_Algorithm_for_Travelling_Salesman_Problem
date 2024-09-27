import random
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
        self.iterations = 30
        # Tournament Size  
        self.tournament_size = 5
        # Apply LKH every 2 generations
        self.lkh_frequency = 2
        # Maximum number of moves in a single LKH iteration
        self.max_lkh_moves = 100

        # Take input from file
        self.read_input()
        # Compute Distance Matrix for fast lookup
        self.distance_matrix = self.create_dist_matrix()
        # Cache for storing calculated path lengths
        self.path_cache = {}

        # Create Initial Population
        population = self.initial_population()

        # Run for specified number of iterations
        for i in range(self.iterations):
            population = self.next_generation(population)
            if i % self.lkh_frequency == 0:
                # Apply LKH optimization to the best individual in each generation
                best_route = min(population, key=self.path_length)
                optimized_route = self.lin_kernighan(best_route)
                # Find the index of the best route and update it
                best_index = np.argmin([self.path_length(route) for route in population])
                population[best_index] = optimized_route

        best_route = min(population, key=self.path_length)
        best_length = self.path_length(best_route)
        self.write_output(best_route, best_length)

    def lin_kernighan(self, route):

        best_length = self.path_length(route)
        improved = True
        moves = 0
        # Limit the LKH moves to reduce time complexity
        while improved and moves < self.max_lkh_moves:
            improved = False
            moves += 1

            for i in range(len(route)):
                # Try to find an improving move
                new_route, improvement = self.find_lkh_move(route, i)
                if improvement:
                    route = new_route
                    best_length -= improvement
                    improved = True
                    break

        return route

    def find_lkh_move(self, route, start):

        best_gain = 0
        best_route = None
        n = len(route)

        for i in range(2, min(n-1, 5)):  # Limit the search to 2-opt, 3-opt, and 4-opt moves
            for j in range(start+i, n):
                # Calculate the gain of reversing the segment [start+1:j]
                gain = (self.distance_matrix[route[start], route[start+1]] +
                        self.distance_matrix[route[j-1], route[j]]) - \
                       (self.distance_matrix[route[start], route[j-1]] +
                        self.distance_matrix[route[start+1], route[j]])
                # We want the best gain 
                if gain > best_gain:
                    new_route = np.concatenate([route[:start+1],
                                                route[start+1:j][::-1],
                                                route[j:]])
                    best_gain = gain
                    best_route = new_route

        return best_route, best_gain

    def read_input(self):
        with open(self.filepath, "r") as f:
            # First Line denotes the number of cities
            self.num_cities = int(f.readline().strip())
            # x, y, z coordinates
            self.cities = np.array([list(map(int, line.strip().split())) for line in f])
        
    def create_dist_matrix(self):
        # Calculate distance using more efficient np.linalg.norm
        return np.linalg.norm(self.cities[:, np.newaxis] - self.cities, axis=2)
    
    def initial_population(self):
        # Create random population for first iteration
        # np.random is apparently faster than random.random()
        return [np.random.permutation(self.num_cities).astype(np.int32) for _ in range(self.population_size)]
    
    def path_length(self, path):
        key = path.tobytes()
        if key not in self.path_cache:
            length = np.sum(self.distance_matrix[path[:-1], path[1:]]) + self.distance_matrix[path[-1], path[0]]
            self.path_cache[key] = length
        return self.path_cache[key]

    def tournament_selection(self, population):
        # Select 5 parents from population at random
        tournament = random.sample(population, self.tournament_size)
        # Return best parents from those 5
        return min(tournament, key=self.path_length)
        
    def crossover(self, parent1, parent2):
        size = len(parent1)
        # Choose two random points
        # This path will be copied into the child from parent1
        start, end = sorted(random.sample(range(size), 2))
        # np.full creates a np array of given shape and type filled with specific value
        child = np.full(size, -1, dtype=np.int32)
        child[start:end] = parent1[start:end]
        remaining = [item for item in parent2 if item not in child[start:end]]
        child[:start] = remaining[:start]
        child[end:] = remaining[start:]
        return child
        
    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            # Mutate by swapping two random cities
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
        
    def next_generation(self, population):
        sorted_pop = sorted(population, key=self.path_length)
        # Retain the best genetic parents from old population
        new_pop = sorted_pop[:self.elite_size]
        
        for _ in range(self.population_size - self.elite_size):
            # Random Parent Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            # Crossover
            child = self.crossover(parent1, parent2)
            # Mutation
            child = self.mutate(child)
            # Add it to new population
            new_pop.append(child)
        
        return new_pop
        
    def write_output(self, best, cost):
        with open("outputLKH.txt", "w") as file:
            file.write(f"{cost}\n")
            for city_index in best:
                city = self.cities[city_index]
                file.write(f"{city[0]} {city[1]} {city[2]}\n")
            first_city = self.cities[best[0]]
            file.write(f"{first_city[0]} {first_city[1]} {first_city[2]}\n")
        
if __name__ == "__main__":
    start_time = time.time()
    GeneticAlgoTSP("input.txt")
    print(f"Execution time: {time.time() - start_time} seconds")