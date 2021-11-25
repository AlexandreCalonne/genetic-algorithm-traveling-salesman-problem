import math


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return math.dist([self.x, self.y], [city.x, city.y])

    def __repr__(self):
        return f'({self.x}, {self.y})'
