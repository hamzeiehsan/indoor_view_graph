import shapely.geometry
from Container import Container
from Isovist import Isovist
from Plotter import Plotter
from Utility import Utility
from ViewGraph import ViewGraph
from geojson import MultiPolygon, Feature, FeatureCollection, dump


class IndoorEnvironment:
    def __init__(self, address, pfiles, hfiles, dfiles, dpfiles, lfiles):  # todo: link between containers
        self.containers = []
        if len(pfiles) == len(hfiles) == len(dfiles) == len(dpfiles) == len(lfiles):
            print('environment files -- count is valid')
            for idx, pfile in enumerate(pfiles):
                self.containers.append(Container(address, pfile, hfiles[idx], dfiles[idx], dpfiles[idx], lfiles[idx]))
        else:
            print('environment files -- count is invalid')

    @staticmethod
    def reformat(address, containers_file, doors_file, landmarks_file, dpoints_file=None):
        pfiles = []
        hfiles = []
        dfiles = []
        dpfiles = []
        lfiles = []
        containers = Utility.read_geojson(address, containers_file)
        doors = Utility.read_geojson(address, doors_file)
        generate_dpoints = False
        if dpoints_file is None:
            generate_dpoints = True
            dpoints = None
        else:
            dpoints = Utility.read_geojson(address, dpoints_file)
        landmarks = Utility.read_geojson(address, landmarks_file)

        for container in containers['features']:
            name = container['properties']['name']
            doors_features = []
            for d in doors['features']:
                if d['properties']['container1'] == name or d['properties']['container2'] == name:
                    doors_features.append(d)
            doors_features = FeatureCollection(doors_features)
            landmarks_features = []
            for l in landmarks['features']:
                if l['properties']['container'] == name:
                    landmarks_features.append(l)
            landmarks_features = FeatureCollection(landmarks_features)

            c_polygon = MultiPolygon([[container['geometry']['coordinates'][0][0]]])
            polygon_features = FeatureCollection([Feature(geometry=c_polygon, properties=container['properties'])])

            c_holes_features = []
            for i in range(1, len(container['geometry']['coordinates'][0])):
                coords = container['geometry']['coordinates'][0][i]
                # coords.reverse()
                c_holes_features.append(Feature(geometry=MultiPolygon([[coords]]), properties={'id':i}))
            holes_features = FeatureCollection(c_holes_features)
            with open('{0}{1}-pfile.geojson'.format(address,name), 'w', encoding='utf-8') as fp:
                dump(polygon_features, fp)
            pfiles.append('{0}{1}-pfile.geojson'.format(address,name))
            with open('{0}{1}-hfile.geojson'.format(address,name), 'w', encoding='utf-8') as fp:
                dump(holes_features, fp)
            hfiles.append('{0}{1}-hfile.geojson'.format(address,name))
            with open('{0}{1}-lfile.geojson'.format(address,name), 'w', encoding='utf-8') as fp:
                dump(landmarks_features, fp)
            lfiles.append('{0}{1}-lfile.geojson'.format(address,name))
            with open('{0}{1}-dfile.geojson'.format(address,name), 'w', encoding='utf-8') as fp:
                dump(doors_features, fp)
            dfiles.append('{0}{1}-dfile.geojson'.format(address,name))
            dpoint_features = []
            if generate_dpoints:
                container_shape = shapely.geometry.shape(container['geometry'])
                skel = Utility.generate_skeleton(polygon_features,
                                                 holes_features['features'])
                dpoint_features = Utility.extract_decision_points(container_shape, skel)
            else:
                for dp in dpoints['features']:
                    if dp['properties']['container'] == name:
                        dpoint_features.append(dp)
                dpoint_features = FeatureCollection(dpoint_features)
            with open('{0}{1}-dpfile.geojson'.format(address,name), 'w', encoding='utf-8') as fp:
                dump(dpoint_features, fp)
            dpfiles.append('{0}{1}-dpfile.geojson'.format(address,name))
        return pfiles, hfiles, dfiles, dpfiles, lfiles

    def cviewgraph(self, cidx):
        container = self.containers[cidx]
        isovist_object = Isovist(container.boundary, container.holes, container.doors,
                                 container.dpoints, container.landmarks)
        container_info = {'name': container.name, 'door_info': container.door_names,
                          'landmark_info': container.landmark_names}
        vg = ViewGraph(isovist_object, container_info)
        return vg, isovist_object

    def construct_view_graph(self):
        vgs = []
        isovist_objects = []
        for idx, container in enumerate(self.containers):
            print('\n*******************************************\nAnalyzing: {}'.format(container.name))
            vg, isovist_object = self.cviewgraph(idx)
            vgs.append(vg)
            isovist_objects.append(isovist_object)
        return vgs, isovist_objects


if __name__ == '__main__':
    # test environment
    # address = 'envs/hypo/'
    # polygon_files = ['hypo_env.geojson']
    # holes_files = ['hypo_holes.geojson']
    # doors_files = ['hypo_doors.geojson']
    # dpoints_files = ['hypo_dpoints.geojson']
    # landmarks_files = ['hypo_landmarks.geojson']
    address = 'envs/mc5/'
    IndoorEnvironment.reformat(address, 'area.geojson', 'doors.geojson', 'landmarks.geojson')
    containers = ['E_Corridor', 'UX_Lab', 'W_Toilet', 'M_Toilet', 'D_Toilet']
    polygon_files = ['{}-pfile.geojson'.format(container) for container in containers]
    holes_files = ['{}-hfile.geojson'.format(container) for container in containers]
    doors_files = ['{}-dfile.geojson'.format(container) for container in containers]
    dpoints_files = ['{}-dpfile.geojson'.format(container) for container in containers]
    # dpoints_files = [None for container in containers]
    landmarks_files = ['{}-lfile.geojson'.format(container) for container in containers]


    # create an indoor environment
    ie = IndoorEnvironment(address, polygon_files, holes_files, doors_files, dpoints_files, landmarks_files)

    # create view graph
    vgs, isovist_objects = ie.construct_view_graph()
    vg = vgs[0]
    isovist_object = isovist_objects[0]

    # calculate shortest path and generate verbal description
    vp, pv= vg.shortest_path_regions(2, len(vg.regions_list)-1)

    # derive door-to-door visibility graph (doors and decision points)
    connected, dtd_graph = vg.generate_door_to_door_graph(isovist_object)

    # derive door-to-door visibility graph (only doors)
    connected2, dtd_graph2 = vg.generate_door_to_door_graph(isovist_object, only_doors=True)

    # derive all shortest path visibility graph and spanning tree
    vps, pvs, st_vps, st_pvs, nvgraph =\
        vg.generate_navigation_graph(isovist_object, indirect_access=False)

    # derive place graph
    place_graph = vg.generate_place_graph(isovist_object)
    pg_l2_2 = Utility.create_subgraph(place_graph, 'the landmark 2', 2)
    pg_g7_2 = Utility.create_subgraph(place_graph, 'the door to M_Toilet', 2)

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
    plotter.write_graph('placegraph_l2_2.html', pg_l2_2)
    plotter.refresh()
    ls = []
    gs = []
    for n in list(pg_l2_2.nodes):
        if 'landmark ' in n:
            lid = int(n.replace('landmark ', ''))
            ls.append(isovist_object.landmarks_points[lid])
        elif 'gateway ' in n:
            gid = int(n.replace('gateway ', ''))
            gs.append(isovist_object.door_points[gid])
    plotter.add_points(ls, label='graph (landmarks)', color='ko')
    plotter.add_points(gs, label='graph (gateways)', color='mo')
    plotter.show()

    plotter.write_graph('placegraph_g7_2.html', pg_g7_2)
    plotter.refresh()
    ls = []
    gs = []
    for n in list(pg_g7_2.nodes):
        if 'landmark ' in n:
            lid = int(n.replace('landmark ', ''))
            ls.append(isovist_object.landmarks_points[lid])
        elif 'gateway ' in n:
            gid = int(n.replace('gateway ', ''))
            gs.append(isovist_object.door_points[gid])
    plotter.add_points(ls, label='graph (landmarks)', color='ko')
    plotter.add_points(gs, label='graph (gateways)', color='mo')
    plotter.show()
