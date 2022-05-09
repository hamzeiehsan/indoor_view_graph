import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def add_poly(self, space_x, space_y, color='black'):
        plt.plot(space_x, space_y, color)

    def add_holes(self, holes_x, holes_y, color='r'):
        for i in range(0, len(holes_x)):
            plt.plot(holes_x[i], holes_y[i], color)

    def add_points(self, points, label, color='go'):
        plt.plot([d.x() for d in points], [d.y() for d in points],
                 color, label=label)

    def add_points_lines(self, points_list, color='blue'):
        for twopoints in points_list:
            plt.plot([d.x() for d in twopoints], [d.y() for d in twopoints], color=color)

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
