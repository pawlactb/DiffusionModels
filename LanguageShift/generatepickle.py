import pickle

from LanguageShift.Language import LanguageModel

filename = 'neighbor.pkl'
m = LanguageModel([.005, .005], 'doctoreddata.csv')
pickle.dump(m.grid.agent_neighbors, open(filename, 'wb'))
print('Pickled as ' + filename)
