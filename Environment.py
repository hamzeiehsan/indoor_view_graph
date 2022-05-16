from Container import Container
from Isovist import Isovist
from Plotter import Plotter
from ViewGraph import ViewGraph
from Utility import Utility

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
        return vg, isovist_object


if __name__ == '__main__':
    # test environment
    address = 'envs/hypo/'
    polygon_files = ['hypo_env.geojson']
    holes_files = ['hypo_holes.geojson']
    doors_files = ['hypo_doors.geojson']
    dpoints_files = ['hypo_dpoints.geojson']
    landmarks_files = ['hypo_landmarks.geojson']

    # create an indoor environment
    ie = IndoorEnvironment(address, polygon_files, holes_files, doors_files, dpoints_files, landmarks_files)

    # create view graph
    vg, isovist_object = ie.cviewgraph(0)

    # calculate shortest path and generate verbal description
    vp, pv= vg.shortest_path_regions(38, 56)

    # derive door-to-door visibility graph (doors and decision points)
    connected, dtd_graph = vg.generate_door_to_door_graph(isovist_object)

    # derive door-to-door visibility graph (only doors)
    connected2, dtd_graph2 = vg.generate_door_to_door_graph(isovist_object, only_doors=True)

    # derive all shortest path visibility graph and spanning tree
    vps, pvs, st_vps, st_pvs = vg.generate_all_gateway_paths(isovist_object)

    # derive place graph
    place_graph = vg.generate_place_graph(isovist_object)
    pg_l2_2 = Utility.create_subgraph(place_graph, 'landmark 2', 2)

    input('Press Enter: Describe the shortest path')
    plotter = Plotter()
    plotter.add_isovist(isovist_object)
    plotter.add_views(pv)
    plotter.show()
    plotter.close()
    vg.generate_route_description(vp)

    input('Press Enter: Door to door visibility (doors+gateways)')
    plotter = Plotter()
    plotter.add_isovist(isovist_object)
    plotter.add_points_lines(connected)
    plotter.show()
    plotter.close()
    plotter.plot_graph(dtd_graph)

    input('Press Enter: Door to door visibility (only doors)')
    plotter = Plotter()
    plotter.add_poly(isovist_object.space_x, isovist_object.space_y)
    plotter.add_holes(isovist_object.holes_x, isovist_object.holes_y)
    plotter.add_points(isovist_object.door_points[:isovist_object.door_idx], 'doors')
    plotter.add_points_lines(connected2)
    plotter.show()
    plotter.close()
    plotter.plot_graph(dtd_graph2)

    input('Press Enter: Portal-junction navigation graph')
    plotter = Plotter()
    plotter.add_isovist(isovist_object)

    for pv in pvs:
        plotter.add_views(pv)
    plotter.show()

    plotter.refresh()
    for pv in st_pvs:
        plotter.add_views(pv)
    plotter.show()

    plotter.refresh()
    for pv in st_pvs:
        plotter.add_points_lines(pv, is_vis=False)
    plotter.show()

    input('Press Enter: Place graph generation; visualize for all and only for landmark 2')
    plotter.write_graph('placegraph.html', place_graph)
    plotter.write_graph('placegraph_l2_2.html', pg_l2_2)