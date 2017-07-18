from LanguageShift.Language import LanguageModel

m = LanguageModel([.005, .005], 'doctoreddata.csv', grid_pickle='neighbor.pkl')
m.run(30)
print(m.datacollector.get_agent_vars_dataframe().tail())
