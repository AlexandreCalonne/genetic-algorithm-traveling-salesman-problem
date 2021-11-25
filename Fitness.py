class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            self.distance = sum([self.route[i].distance(self.route[i + 1]) for i in range(len(self.route) - 1)])
            self.distance += self.route[len(self.route) - 1].distance(self.route[0])

            if self.distance == 0:
                print("NTM")

        return self.distance

    def route_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())

        return self.fitness
