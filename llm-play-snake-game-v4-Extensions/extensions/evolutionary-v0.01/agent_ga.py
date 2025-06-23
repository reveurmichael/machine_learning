"""GAAgent – Minimal Genetic-Algorithm Planner (v0.01)

The agent evolves a short **sequence of moves** that minimises the Euclidean
(Manhattan) distance from the snake's head to the apple while avoiding *obvious*
collisions against walls.  It is *not* a production-ready planner; its sole
purpose is to showcase how an evolutionary algorithm can be plugged into the
existing BaseClassBlabla architecture without any extra support from core.

Algorithm Outline (very small computational budget):
1. **Encoding**  – A chromosome is a fixed-length list of moves chosen from
   {UP, DOWN, LEFT, RIGHT}.
2. **Population** – 30 chromosomes.
3. **Fitness**    – Simulate the chromosome on a *copy* of the current state
   (ignoring self-collision for simplicity) and return the Manhattan distance
   to the apple after all moves.  Lower is better; crashing into a wall gets a
   large penalty.
4. **Selection**  – Tournament of size 3.
5. **Crossover**  – Single-point crossover with 0.7 probability.
6. **Mutation**   – Each gene mutates to a random move with 0.1 probability.
7. Run for **10 generations** and output the **first gene** of the best
   chromosome as the chosen action.

This tiny GA is intentionally lightweight so that v0.01 remains simple; future
versions can add proper collision checks, dynamic chromosome lengths, etc.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from config.game_constants import DIRECTIONS, VALID_MOVES

Move = str  # Alias for readability
Position = Tuple[int, int]


class GAAgent:
    """Very small GA planner that outputs a single move."""

    # GA hyper-parameters – easy to tweak via constructor
    POP_SIZE: int = 30
    CHROMOSOME_LEN: int = 8
    GENERATIONS: int = 10
    TOURNAMENT: int = 3
    CX_PROB: float = 0.7
    MUT_PROB: float = 0.1

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    # ---------------------
    # Public interface expected by GameLogic
    # ---------------------

    def plan(self, head: Position, apple: Position, grid_size: int) -> List[Move]:
        """Return a *sequence* of planned moves starting from *head*.

        The GameLogic will pop moves from this list one by one.  If something
        invalid happens mid-execution the logic will just ask for another plan
        next round – that is perfectly fine for v0.01 simplicity.
        """
        population = [self._random_chromosome() for _ in range(self.POP_SIZE)]

        for _ in range(self.GENERATIONS):
            fitnesses = [self._fitness(chrom, head, apple, grid_size) for chrom in population]
            new_pop: List[List[Move]] = []
            while len(new_pop) < self.POP_SIZE:
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                self._mutate(child1)
                self._mutate(child2)
                new_pop.extend([child1, child2])
            population = new_pop[: self.POP_SIZE]

        # Choose best chromosome from final population
        best = min(population, key=lambda chrom: self._fitness(chrom, head, apple, grid_size))
        return best

    # ---------------------
    # GA internals
    # ---------------------

    def _random_chromosome(self) -> List[Move]:
        return [self.rng.choice(VALID_MOVES) for _ in range(self.CHROMOSOME_LEN)]

    def _fitness(self, chromosome: List[Move], head: Position, apple: Position, grid_size: int) -> float:
        x, y = head
        for move in chromosome:
            dx, dy = DIRECTIONS[move]
            x += dx
            y += dy
            # Penalise wall collisions heavily
            if not (0 <= x < grid_size and 0 <= y < grid_size):
                return float("inf")  # Worst possible
        return abs(x - apple[0]) + abs(y - apple[1])  # Manhattan distance

    def _tournament_select(self, population: List[List[Move]], fitnesses: List[float]) -> List[Move]:
        contenders = self.rng.sample(list(zip(population, fitnesses)), k=self.TOURNAMENT)
        return min(contenders, key=lambda x: x[1])[0]

    def _crossover(self, p1: List[Move], p2: List[Move]) -> Tuple[List[Move], List[Move]]:
        if self.rng.random() > self.CX_PROB:
            return p1[:], p2[:]
        point = self.rng.randint(1, self.CHROMOSOME_LEN - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2

    def _mutate(self, chrom: List[Move]) -> None:
        for i in range(self.CHROMOSOME_LEN):
            if self.rng.random() < self.MUT_PROB:
                chrom[i] = self.rng.choice(VALID_MOVES) 