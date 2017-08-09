import numpy as np
from mesa import Agent


class LanguageAgent(Agent):
    def __init__(self, model, unique_id, initial_prob_v):
        """
        A LanguageAgent represents a particular place during a language shift simulation.
        :param model: the model that the agent is in
        :param unique_id: Location number of the agent
        :param initial_prob_v: a list of probabilities of speaking particular languages
        """
        super().__init__(unique_id, model)
        self.probability = np.array(initial_prob_v)
        self.next_probability = np.array(self.probability, copy=True)
        self.diffusion = self.model.diffusion
        self.get_population()

    def get_population(self):
        '''
        Updates the population of the LanguageAgent
        Returns: None
        '''
        self.population = self.model.agent_pop[self.unique_id][self.model.schedule.time]

    def calculate_contribution(self, other):
        '''
        Args:
            other: Another agent for which you want to find the impact from.
        Returns: None
        '''
        # this if statement turns the ret_val into 0 if the other agent is to far away
        # if self.model.grid.get_distance(self, other) > np.sqrt(2):
        #    ret_val = 0
        #   print('zero ret_val!!!!' + str(self.unique_id) + ' ' + str(other.unique_id))
        # else:
        ret_val = ((other.population * other.probability) / (4 * np.pi * self.diffusion)) * np.exp(
            -np.square(self.model.grid.get_distance(self, other))) / (4 * self.diffusion)
        return ret_val

    def step(self):
        '''
        Prepare for the next timestep
        Returns: None
        '''
        f = np.zeros(len(self.probability))
        self.get_population()
        for neighbor in self.model.grid.get_neighbors_by_agent(self)[1:self.model.grid.neighborhood_size + 1]:
            f += self.calculate_contribution(neighbor)

        self.next_probability = ((self.population * self.probability) + f) / (np.sum(f) + self.population)

    def advance(self):
        '''
        Advance to the next timestep
        Returns: None
        '''
        self.probability, self.next_probability = self.next_probability, self.probability
