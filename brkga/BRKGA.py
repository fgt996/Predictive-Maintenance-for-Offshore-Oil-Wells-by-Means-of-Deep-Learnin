#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniele Ferone
"""

from random import Random
from brkga import Population
import logging
import datetime


class BRKGA:
    def __init__(self, n, p, pe, pm, rhoe, seed, decoder, max_time=0, max_time_wi=0, maximize=False):
        """
        Parameters
        ----------
        :param n: number of genes in each chromosome
        :param p: number of elements in each population
        :param pe: percentage of elite items into each population
        :param pm: percentage of mutants introduced at each generation into the population
        :param rhoe: probability that an offspring inherits the allele of its elite parent
        :param seed: seed for random numbers
        :param decoder: decoder object
        :param max_time: max time
        :param max_time_wi: max time without improvement
        :param maximize: Maximization problem
        """
        self.n = n
        self.p = p
        self.pe = int(p * pe)
        self.pm = int(pm * p)
        self.rhoe = rhoe
        self.rand = Random()
        self.rand.seed(seed)
        self.decoder = decoder

        self.max_time = max_time
        self.max_time_wi = max_time_wi

        self.previous = Population.Population(n, p, maximize)
        self.current = Population.Population(n, p, maximize)

        self.start = datetime.datetime.now()
        self.last_improvement = self.start

        for i in range(0, p):
            created = False
            while not created:
                for j in range(0, n):
                    self.current.population[i][j] = self.rand.random()

                fit = decoder.decode(self.current.get_chromosome(i))
                self.current.set_fitness(i, fit)
                created = True

        self.current.sort_fitness()

    def evolution(self, curr, next_population):
        i = 0  # Iterate chromosome by chromosome unsigned

        # The 'pe' best chromosomes are maintained
        while i < self.pe:
            for j in range(0, self.n):
                next_population.population[i][j] = curr.population[curr.fitness[i][1]][j]

            next_population.fitness[i][0] = curr.fitness[i][0]
            next_population.fitness[i][1] = i

            i += 1

        # We'll mate 'p - pe - pm' pairs; initially, i = pe, so we need to iterate until i < p - pm:
        while i < self.p - self.pm:
            created = False
            while not created:
                elite_parent = self.rand.randint(0, self.pe - 1)
                nonelite_parent = self.pe + (self.rand.randint(0, self.p - self.pe - 1))

                for j in range(0, self.n):
                    source_parent = nonelite_parent
                    if self.rand.random() < self.rhoe:
                        source_parent = elite_parent

                    next_population.population[i][j] = curr.population[curr.fitness[source_parent][1]][j]

                fit = self.decoder.decode(next_population.population[i])
                next_population.set_fitness(i, fit)
                created = True

            i += 1

        # We'll introduce 'pm' mutants
        while i < self.p:
            created = False
            while not created:
                for j in range(0, self.n):
                    next_population.population[i][j] = self.rand.random()

                fit = self.decoder.decode(next_population.population[i])
                next_population.set_fitness(i, fit)
                created = True

            i += 1

        # i = int(self.pe)
        # while i < int(self.p):
        #     logging.getLogger().debug("{} <? {}".format(i, int(self.p)))
        #     next_population.set_fitness(i, self.decoder.decode(next_population.population[i]))
        #     i += 1

        next_population.sort_fitness()

    def evolve(self, generations=1):
        for i in range(0, generations):
            self.evolution(self.current, self.previous)
            self.current, self.previous = self.previous, self.current

    def get_best_fitness(self):
        return self.current.get_best_fitness()

    def get_best_chromosome(self):
        return self.current.get_chromosome(0)

    def optimize(self, generations):
        current_generation = 0
        while generations <= 0 or current_generation <= generations:
            self.evolve()
            seconds = (datetime.datetime.now() - self.start).total_seconds()
            if 0 < self.max_time < seconds:
                break
            seconds_from_last_improvement = (datetime.datetime.now() - self.last_improvement).total_seconds()
            if 0 < self.max_time_wi < seconds_from_last_improvement:
                break
            current_generation += 1
            logging.getLogger().info("Generazione: {}, best fitness: {}".format(current_generation,
                                                                                self.get_best_fitness()))