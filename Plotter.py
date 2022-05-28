import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network


class Plotter:
    def __init__(self, isovist=None):
        self.fig, self.ax = plt.subplots()
        if isovist is not None:
            self.add_isovist(isovist)

    def refresh(self, save_iso=True):
        self.close()
        if save_iso:
            self.__init__(self.isovist)
        else:
            self.fig, self.ax = plt.subplots()

    def add_poly(self, space_x, space_y, color='black'):
        plt.plot(space_x, space_y, color)

    def add_holes(self, holes_x, holes_y, color='r'):
        for i in range(0, len(holes_x)):
            plt.plot(holes_x[i], holes_y[i], color)

    def add_points(self, points, label, color='go'):
        plt.plot([d.x() for d in points], [d.y() for d in points],
                 color, label=label)

    def add_points_lines(self, points_list, is_vis=True, color='blue'):
        for twopoints in points_list:
            if is_vis:
                plt.plot([d.x() for d in twopoints], [d.y() for d in twopoints], color=color)
            else:
                plt.plot([d.x for d in twopoints], [d.y for d in twopoints], color=color)

    def add_views(self, path_view, is_labeled=False, label='views'):
        X = []
        Y = []
        U = []
        V = []
        for view in path_view:
            X.append(view[0].x)
            Y.append(view[0].y)
            U.append(view[1].x - view[0].x)
            V.append(view[1].y - view[0].y)
        if is_labeled:
            self.ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, label=label)
        else:
            self.ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)

    def add_isovist(self, isovist_object):
        self.isovist = isovist_object
        self.add_poly(isovist_object.space_x, isovist_object.space_y)
        self.add_holes(isovist_object.holes_x, isovist_object.holes_y)
        self.add_points(isovist_object.door_points, 'gateway')
        self.add_points(isovist_object.landmarks_points, 'landmark', color='ro')

    def show(self, legend=True):
        if legend:
            plt.legend()
        plt.show()

    def close(self):
        plt.close()
        plt.cla()
        plt.clf()

    def plot_graph(self, graph):
        nx.draw(graph)
        plt.show()

    def write_graph(self, file, graph, is_directed=True, notebook=False, width='90%', height='800px'):
        nt2 = Network(width=width, height=height, directed=is_directed, notebook=notebook)
        nt2.from_nx(graph, show_edge_weights=False)
        nt2.options.physics.use_repulsion({'node_distance': 185, 'central_gravity': 0.2, 'spring_length': 200,
                                           'spring_strength': 0.05, 'damping': 0.09})
        nt2.show_buttons(filter_=True)
        nt2.show(file)

    def get_plt(self):
        return plt
