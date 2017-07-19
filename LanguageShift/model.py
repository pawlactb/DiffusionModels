from LanguageShift.Language import LanguageModel
from LanguageShift.Visualization import Visualization




m = LanguageModel([.013, .009], 'doctoreddata.csv', grid_pickle='neighbor.pkl')
# m.run(30)
# m.datacollector.get_agent_vars_dataframe().to_csv('output.txt')

v = Visualization(shapefile='./shapefiles/AUT_adm2', csv='output.txt')
v.new_figure()
