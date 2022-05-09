from Container import Container
from Isovist import Isovist
from Plotter import Plotter
from ViewGraph import ViewGraph

class IndoorEnvironment:
    def __init__(self, address, pfiles, hfiles, dfiles, dpfiles, lfiles):  # todo: link between containers
        self.containers = []
        if len(pfiles) == len(hfiles) == len(dfiles) == len(dpfiles) == len(lfiles):
            print('environment files -- count is valid')
            for idx, pfile in enumerate(pfiles):
                self.containers.append(Container(address, pfile, hfiles[idx], dfiles[idx], dpfiles[idx], lfiles[idx]))
        else:
            print('environment files -- count is invalid')

    def cviewgraph(self, cidx):
        container = self.containers[cidx]
        isovist_object = Isovist(container.boundary, container.holes, container.doors,
                                 container.dpoints, container.landmarks)
        vg = ViewGraph(isovist_object)
        return vg


if __name__ == '__main__':
    address = 'envs/hypo/'
    polygon_files = ['hypo_env.geojson']
    holes_files = ['hypo_holes.geojson']
    doors_files = ['hypo_doors.geojson']
    landmarks_files = ['hypo_landmarks.geojson']
    ie = IndoorEnvironment(address, polygon_files, holes_files, doors_files, [None], landmarks_files)
    vg = ie.cviewgraph(0)
