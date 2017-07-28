import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap


class Visualization(object):
    def __init__(self, shapefile, dataFrame=None, csv=None):
        self.shapefiles = shapefile

        # plot map with albert's equal area projection
        self.draw_map(res='c',
                      proj='aea',
                      lat=46.64, lon=14.26,
                      lllong=12.5, lllat=46, urlong=15.1, urlat=47.15)

        for sf in self.shapefiles:
            self.add_shapefile(sf[0], sf[1])

        if dataFrame is None:
            self.data = pd.read_csv(csv)
        else:
            self.data = dataFrame

        if self.data is None:
            raise ValueError('Must pass data to visualization.')

        self.figure_count = 0

    def add_shapefile(self, file, key):
        self.bmap.readshapefile(file, name=key)

    def draw_map(self, res,
                 proj,
                 lat, lon,
                 lllong, lllat, urlong, urlat):
        self.bmap = Basemap(resolution=res,
                            projection=proj,
                            lat_0=lat, lon_0=lon,
                            llcrnrlon=lllong, llcrnrlat=lllat, urcrnrlon=urlong, urcrnrlat=urlat)

        self.bmap.drawmapboundary(fill_color='#46bcec')
        self.bmap.fillcontinents(color='#f2f2f2', lake_color='#46bcec')

    def show(self):
        plt.colorbar()
        plt.show()

    def new_figure(self, title='Figure'):
        # increment the figure count
        self.figure_count += 1

        plt.figure(self.figure_count)
        plt.title(title)

    def plot_timestep(self, timestep):
        longs = np.array(self.data.loc[self.data['Step'] == timestep]['long'])
        lats = np.array(self.data.loc[self.data['Step'] == timestep]['lat'])

        german = self.data.loc[self.data['Step'] == timestep]['p_german']
        slovene = self.data.loc[self.data['Step'] == timestep]['p_slovene']
        cmap = cm.ScalarMappable(cmap='coolwarm')
        col = cmap.to_rgba(german.tolist(), norm=None)

        # x, y = self.bmap(longs, lats, c=col)
        self.bmap.scatter(longs, lats, latlon=True, marker='s', s=1, c=german, cmap='coolwarm', zorder=10)

        # self.bmap.plot(x, y, 'bo', markersize=1)
