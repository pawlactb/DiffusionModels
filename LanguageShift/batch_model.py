import datetime

import numpy as np
from mesa.batchrunner import BatchRunner

from LanguageShift.LanguageModel import LanguageModel

model_params = {'diffusivity': [0.013, 0.009], 'timestep': 1, 'filename': 'doctoreddata.csv',
                'grid_pickle': 'neighbor.pkl'}

model_reporters = {}
agent_reporters = {"pop": lambda x: x.population,
                   "p_p_german": lambda x: x.p_probability[0],
                   "p_p_slovene": lambda x: x.p_probability[1],
                   "p_german": lambda x: x.probability[0],
                   "p_slovene": lambda x: x.probability[1],
                   "p_diff": lambda x: np.sum(np.abs(x.probability - x.p_probability)),
                   "lat": lambda x: x.pos[0],
                   "long": lambda x: x.pos[1]}

batch = BatchRunner(LanguageModel, parameter_values=model_params, agent_reporters=agent_reporters, max_steps=30,
                    iterations=1)
batch.run_all()

batch.get_agent_vars_dataframe().to_csv('batch_output' + str(datetime.date.today()))
