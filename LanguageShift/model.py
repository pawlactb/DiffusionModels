from LanguageShift.Language import LanguageModel

m = LanguageModel([.005, .005], 'doctoreddata.csv')
m.run(30)
print(m.datacollector.get_agent_vars_dataframe().tail())
