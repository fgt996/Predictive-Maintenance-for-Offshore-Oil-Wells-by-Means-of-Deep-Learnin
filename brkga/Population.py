#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniele Ferone
"""
class Population:
    def __init__(self, n, p, maximize):
        """
        Population of the BRKGA
        :param n: number of genes in each chromosome
        :param p: number of elements in each population
        :param maximize: maximization problem
        """
        self.n = n
        self.p = p
        self.population = []
        self.fitness = []
        for i in range(0, p):
            self.population.append([0] * n)
            self.fitness.append([0, i])

        self.maximize = maximize

    def get_chromosome(self, i):
        return self.population[self.fitness[i][1]]

    def set_fitness(self, i, f):
        self.fitness[i] = [f, i]

    def sort_fitness(self):
        self.fitness.sort(reverse=self.maximize)

    def get_fitness(self, i):
        return self.fitness[i][0]

    def get_best_fitness(self):
        return self.get_fitness(0)
