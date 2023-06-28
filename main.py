import warnings

import shapely.geometry
from geojson import MultiPolygon, Feature, FeatureCollection, dump

from Container import Container
from Isovist import Isovist
from Parameters import Parameters
from Plotter import Plotter
from Environment import IndoorEnvironment

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Basic environment
    if Parameters.basic:
        address = 'envs/basic/'
        pfiles = ['t_bound.geojson']
        hfiles = [None]
        dfiles = ['t_doors.geojson']
        dpfiles = [None]
        lfiles = ['t_landmarks.geojson']
        # create an indoor environment
        ie = IndoorEnvironment(address, pfiles, hfiles, dfiles, dpfiles, lfiles)

    # Hypo environment
    elif Parameters.hypo:
        address = 'envs/hypo/'
        pfiles = ['hypo_env.geojson']
        hfiles = ['hypo_holes.geojson']
        dfiles = ['hypo_doors.geojson']
        dpfiles = ['hypo_dpoints.geojson']
        lfiles = ['hypo_landmarks.geojson']
        # create an indoor environment
        ie = IndoorEnvironment(address, pfiles, hfiles, dfiles, dpfiles, lfiles)

    # MC5 real world environment
    else:
        address = 'envs/mc-floor-5/'
        pfiles, hfiles, dfiles, dpfiles, lfiles = IndoorEnvironment.reformat(
            address, 'containers.geojson', 'doors.geojson', 'landmarks.geojson')
        # create an indoor environment
        ie = IndoorEnvironment('', pfiles, hfiles, dfiles, dpfiles, lfiles)

    # create view graph
    vgs, isovist_objects = ie.construct_view_graph()
    if not Parameters.basic and not Parameters.hypo:
        cidx = ie.containers_names.index('Active Hub')
        vg = vgs[cidx]
        isovist_object = isovist_objects[cidx]

        # calculate shortest path and generate verbal description
        vp, pv = vg.shortest_path_regions(0, len(vg.regions_list) - 1)

        # derive door-to-door visibility graph (doors and decision points)
        connected, dtd_graph = vg.generate_door_to_door_graph(isovist_object)

        # derive door-to-door visibility graph (only doors)
        connected2, dtd_graph2 = vg.generate_door_to_door_graph(isovist_object, only_doors=True)

        # derive all shortest path visibility graph and spanning tree
        vps, pvs, st_vps, st_pvs, nvgraph = \
            vg.generate_navigation_graph(isovist_object, indirect_access=False)

        # derive place graph
        place_graph = vg.generate_place_graph(isovist_object)

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
        plotter.write_graph('d-t-d-all.html', dtd_graph, is_directed=False)

        input('Press Enter: Door to door visibility (only doors)')
        plotter = Plotter()
        plotter.add_poly(isovist_object.space_x, isovist_object.space_y)
        plotter.add_holes(isovist_object.holes_x, isovist_object.holes_y)
        plotter.add_points(isovist_object.door_points[:isovist_object.door_idx], 'doors')
        plotter.add_points_lines(connected2)
        plotter.show()
        plotter.close()
        plotter.write_graph('d-t-d-doors.html', dtd_graph2, is_directed=False)

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
    else:
        vg = vgs[0]
        isovist_object = isovist_objects[0]
