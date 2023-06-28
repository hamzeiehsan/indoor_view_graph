import warnings

import shapely.geometry
from geojson import MultiPolygon, Feature, FeatureCollection, dump

from Container import Container
from Isovist import Isovist
from Parameters import Parameters
from Utility import Utility
from ViewGraph import ViewGraph

warnings.filterwarnings("ignore")


class IndoorEnvironment:
    def __init__(self, address, pfiles, hfiles, dfiles, dpfiles, lfiles):
        self.containers = []
        self.container_info = {}
        self.containers_names = []
        self.cviewgraphs = []
        self.isovist_objects = []
        self.graph = None
        if len(pfiles) == len(hfiles) == len(dfiles) == len(dpfiles) == len(lfiles):
            print('environment files -- count is valid')
            for idx, pfile in enumerate(pfiles):
                if 'workplace' not in pfile and 'room' not in pfile and 'focus' not in pfile:
                    # if True:
                    container = Container(address, pfile, hfiles[idx], dfiles[idx], dpfiles[idx], lfiles[idx])
                    self.containers.append(container)
                    self.containers_names.append(container.name)
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
            doors_points = []
            for d in doors['features']:
                if d['properties']['container1'] == name or d['properties']['container2'] == name:
                    doors_features.append(d)
                    doors_points.append(
                        shapely.geometry.Point(d['geometry']['coordinates'][0], d['geometry']['coordinates'][1]))
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
                c_holes_features.append(Feature(geometry=MultiPolygon([[coords]]), properties={'id': i}))
            holes_features = FeatureCollection(c_holes_features)
            with open('{0}{1}-pfile.geojson'.format(address, name), 'w', encoding='utf-8') as fp:
                dump(polygon_features, fp)
            pfiles.append('{0}{1}-pfile.geojson'.format(address, name))
            with open('{0}{1}-hfile.geojson'.format(address, name), 'w', encoding='utf-8') as fp:
                dump(holes_features, fp)
            hfiles.append('{0}{1}-hfile.geojson'.format(address, name))
            with open('{0}{1}-lfile.geojson'.format(address, name), 'w', encoding='utf-8') as fp:
                dump(landmarks_features, fp)
            lfiles.append('{0}{1}-lfile.geojson'.format(address, name))
            with open('{0}{1}-dfile.geojson'.format(address, name), 'w', encoding='utf-8') as fp:
                dump(doors_features, fp)
            dfiles.append('{0}{1}-dfile.geojson'.format(address, name))
            dpoint_features = []
            if generate_dpoints:
                container_shape = shapely.geometry.shape(container['geometry'])
                skel = Utility.generate_skeleton(polygon_features,
                                                 holes_features['features'])
                dpoint_features = Utility.extract_decision_points(container_shape, skel, doors=doors_points)
            else:
                for dp in dpoints['features']:
                    if dp['properties']['container'] == name:
                        dpoint_features.append(dp)
                dpoint_features = FeatureCollection(dpoint_features)
            with open('{0}{1}-dpfile.geojson'.format(address, name), 'w', encoding='utf-8') as fp:
                dump(dpoint_features, fp)
            dpfiles.append('{0}{1}-dpfile.geojson'.format(address, name))
        return pfiles, hfiles, dfiles, dpfiles, lfiles

    def cviewgraph(self, cidx):
        container = self.containers[cidx]
        isovist_object = Isovist(container.boundary, container.holes, container.doors,
                                 container.dpoints, container.landmarks)
        container_info = {'name': container.name, 'door_info': container.door_names,
                          'landmark_info': container.landmark_names}
        self.container_info[container.name_id] = container_info
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

        connected_containers = []
        for idx, container in enumerate(self.containers):
            c_name = container.name
            if idx == 0:
                self.graph = vgs[idx].rviewgraph.copy()
                connected_containers.append(container)
            else:
                self.graph = Utility.merge_graphs(self.graph, vgs[idx].rviewgraph)
                for idx2, container2 in enumerate(connected_containers):
                    c_name2 = container2.name
                    for did, d in enumerate(container.doors):
                        if d['properties']['container1'] == container2.name_id or \
                                d['properties']['container2'] == container2.name_id:
                            did2 = -1
                            for tdid2, d2 in enumerate(container2.doors):
                                if d['properties']['id'] == d2['properties']['id']:
                                    did2 = tdid2
                                    break
                            if did2 != -1:  # todo: augment turn or follow action to edges!
                                for vvto1 in vgs[idx].to_door_vids[did]:
                                    for vvfrom2 in vgs[idx2].from_door_vids[did2]:
                                        v1 = vgs[idx].rviews[vvto1]
                                        v2 = vgs[idx2].rviews[vvfrom2]
                                        vto1 = '{0}-V{1}'.format(c_name, vvto1)
                                        vfrom2 = '{0}-V{1}'.format(c_name2, vvfrom2)
                                        self.graph.add_edge(
                                            vto1, vfrom2,
                                            weight=Parameters.door_weight + Utility.calculate_distance(v1[0], v1[1]),
                                            label='Enter {0} and {1}'.format(container.door_names[did],
                                                                             Utility.calculate_turn_follow(v1, v2)),
                                            action='enter')
                                for vvto2 in vgs[idx2].to_door_vids[did2]:
                                    for vvfrom1 in vgs[idx].from_door_vids[did]:
                                        v1 = vgs[idx2].rviews[vvto2]
                                        v2 = vgs[idx].rviews[vvfrom1]
                                        vto2 = '{0}-V{1}'.format(c_name2, vvto2)
                                        vfrom1 = '{0}-V{1}'.format(c_name, vvfrom1)
                                        self.graph.add_edge(
                                            vto2, vfrom1,
                                            weight=Parameters.door_weight + Utility.calculate_distance(v1[0], v1[1]),
                                            label='Enter {0} and {1}'.format(container2.door_names[did2],
                                                                             Utility.calculate_turn_follow(v1, v2)),
                                            action='enter')

                connected_containers.append(container)
        self.cviewgraphs = vgs
        self.isovist_objects = isovist_objects
        return vgs, isovist_objects

    def shortest_path(self, container_name1, region_id1, container_name2, region_id2):  # hierarchical?
        idx1 = self.containers_names.index(container_name1)
        idx2 = self.containers_names.index(container_name2)
        container1 = self.containers[idx1]
        container2 = self.containers[idx2]
        vg1 = self.cviewgraphs[idx1]
        vg2 = self.cviewgraphs[idx2]
        vid1 = '{0}-V{1}'.format(container1.name, vg1.to_region_ids[region_id1])
        vid2 = '{0}-V{1}'.format(container2.name, vg2.from_region_ids[region_id2])
        spath = Utility.shortest_path(self.graph, vid1, vid2)
        path_view = []
        current = None
        for view in spath:
            container_name = view.split('-V')[0]
            cid = self.containers_names.index(container_name)
            vid = int(view.split('-V')[1])
            path_view.append(self.cviewgraphs[cid].rviews[vid])
            if current is None:
                current = container_name
            if current != container_name:
                print('enter: {}'.format(container_name))
                current = container_name
        return spath, path_view
