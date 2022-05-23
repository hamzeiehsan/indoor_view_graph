import math
import statistics

import geopandas as gpd
import networkx as nx
from numpy import tan
from shapely.geometry import Point, LineString, LinearRing
from shapely.geometry import Polygon as Poly
from shapely.ops import unary_union, polygonize, nearest_points

from Parameters import Parameters
from Utility import Utility


class ViewGraph:
    def __init__(self, isovist_object, container_info):
        self.name = container_info['name']
        self.door_info = container_info['door_info']
        self.landmark_info = container_info['landmark_info']
        self.calculate(isovist_object)
        self.container_info = container_info

    def save(self, address):
        nx.write_graphml(self.rviewgraph, address)

    def load(self, address):
        self.rviewgraph = nx.read_graphml(address)

    def calculate(self, isovist_object):
        self.shapes_list = [isovist_object.shapes[i]['shape'] for i in range(0, len(isovist_object.shapes))]
        overlay_regions = list(polygonize(unary_union(list(x.exterior for x in self.shapes_list))))
        gdf = gpd.GeoDataFrame(geometry=overlay_regions)
        regions = list(gdf['geometry'])
        regions_area = [r.area for r in regions]
        regions_area, regions = zip(*sorted(zip(regions_area, regions)))
        self.regions_list = []
        merge_dict = {}
        skip = []
        for idx, a in enumerate(regions_area):
            if a < Parameters.min_area:
                r = regions[idx]
                touches = {}
                max_length = -1
                max_idx = -1
                for idx2, r2 in enumerate(regions):
                    if idx2 > idx and r2.touches(r):
                        touches[idx2] = r2.intersection(r).length
                        if touches[idx2] > max_length:
                            max_length = touches[idx2]
                            max_idx = idx2
                if max_idx > 0 and touches[max_idx] > 0:
                    if max_idx not in merge_dict.keys():
                        merge_dict[max_idx] = []
                    merge_dict[max_idx].append(idx)
                    skip.append(idx)

        for idx, r in enumerate(regions):
            if idx not in skip:
                if idx not in merge_dict.keys():
                    self.regions_list.append(r)
                else:
                    temp = r
                    for idx2 in merge_dict[idx]:
                        temp = temp.union(regions[idx2])
                    self.regions_list.append(temp)

        holes_centroids = []
        for idx, holex in enumerate(isovist_object.holes_x):
            holey = isovist_object.holes_y[idx]
            holes_centroids.append(Point(statistics.mean(holex), statistics.mean(holey)))

        regions_list_no_holes = []
        for r in self.regions_list:
            is_hole = False
            for cent in holes_centroids:
                if r.contains(cent):
                    is_hole = True
                    break
            if not is_hole:
                regions_list_no_holes.append(r)
        self.regions_list = regions_list_no_holes

        # calculate regions signatures
        print('calculating the visibility signatures...')
        self.signatures = []
        for oregion in self.regions_list:
            center = oregion.centroid
            self.signatures.append([self.shapes_list.index(shp) for shp in self.shapes_list if shp.contains(center)])

        # adjacent regions
        print('calculating adjacency matrix for regions')
        self.adjacency_matrix = {}
        for i in range(0, len(self.regions_list) - 1):
            ri = self.regions_list[i]
            if i not in self.adjacency_matrix.keys():
                self.adjacency_matrix[i] = []
            for j in range(i + 1, len(self.regions_list)):
                rj = self.regions_list[j]
                if ri.touches(rj):
                    if isinstance(ri.intersection(rj), Point):
                        continue
                    self.adjacency_matrix[i].append(j)
                    if j not in self.adjacency_matrix.keys():
                        self.adjacency_matrix[j] = [i]
                    else:
                        self.adjacency_matrix[j].append(i)
        if len(self.adjacency_matrix) == 0:
            self.adjacency_matrix[0] = []

        # constructing view graph for decomposed regions
        print('finding regions that contains doors/gateways and decision points')
        self.regions_info = {i: self.regions_list[i].centroid for i in range(len(self.regions_list))}
        self.regions_doors_info = {}
        done = None
        for pid, door in enumerate(isovist_object.door_points):
            dpoint = Point(door.x(), door.y())
            for rid, r in enumerate(self.regions_list):
                if r.contains(dpoint) or dpoint.touches(r):
                    if rid not in self.regions_doors_info.keys():
                        self.regions_doors_info[rid] = []
                    self.regions_doors_info[rid].append(pid)
                    done = pid
            if done != pid:
                min_dr = 1000
                chosen = None
                for rid, r in enumerate(self.regions_list):
                    dr = r.distance(dpoint)
                    if dr < min_dr:
                        min_dr = dr
                        chosen = rid
                if chosen is not None:
                    if chosen not in self.regions_doors_info.keys():
                        self.regions_doors_info[chosen] = []
                    self.regions_doors_info[chosen].append(pid)


        self.rview_ids = {}
        self.rviews = {}
        self.rview_ls = {}
        counter = 0
        self.views_doors_info = {}

        self.to_door_vids = {}
        self.from_door_vids = {}


        for rid1, signature in enumerate(self.signatures):
            neighbours = self.adjacency_matrix[rid1]
            center = self.regions_info[rid1]
            if rid1 in self.regions_doors_info.keys():
                contained = self.regions_doors_info[rid1]
            else:
                contained = []
            all_points = [Point(isovist_object.door_points[p].x(), isovist_object.door_points[p].y())
                          for p in contained]
            all_points.append(center)

            # in region
            for apid, p in enumerate(all_points):
                is_door = True
                if apid == len(all_points) - 1:
                    is_door = False
                for apid2, p2 in enumerate(all_points):
                    is_door2 = True
                    if apid2 == len(all_points) -1:
                        is_door2 = False
                    if apid != apid2:
                        self.rview_ids[counter] = [rid1, rid1]
                        self.rviews[counter] = [p, p2]
                        self.rview_ls[counter] = LineString([p, p2])
                        if is_door:
                            if contained[apid] not in self.from_door_vids.keys():
                                self.from_door_vids[contained[apid]] = []
                            self.from_door_vids[contained[apid]].append(counter)
                            if not is_door2:
                                self.views_doors_info[contained[apid]] = counter
                        if is_door2:
                            if contained[apid2] not in self.to_door_vids.keys():
                                self.to_door_vids[contained[apid2]] = []
                            self.to_door_vids[contained[apid2]].append(counter)
                        counter += 1

                # to neighbours
                for nrid in neighbours:
                    ncenter = self.regions_info[nrid]
                    if nrid in self.regions_doors_info.keys():
                        ncontained = self.regions_doors_info[nrid]
                    else:
                        ncontained = []
                    nall_points = [Point(isovist_object.door_points[p].x(), isovist_object.door_points[p].y())
                                  for p in ncontained]
                    nall_points.append(ncenter)
                    for napid, np in enumerate(nall_points):
                        is_door2 = True
                        if napid == len(nall_points) -1:
                            is_door2 = False
                        self.rview_ids[counter] = [rid1, nrid]
                        self.rviews[counter] = [p, np]
                        self.rview_ls[counter] = LineString([p, np])
                        if is_door:
                            if contained[apid] not in self.from_door_vids.keys():
                                self.from_door_vids[contained[apid]] = []
                            self.from_door_vids[contained[apid]].append(counter)
                        if is_door2:
                            if ncontained[napid] not in self.to_door_vids.keys():
                                self.to_door_vids[ncontained[napid]] = []
                            self.to_door_vids[ncontained[napid]].append(counter)
                        counter += 1

                # to visible points not inside
                for vpid in signature:
                    if vpid not in contained:
                        vrid = None
                        vp = Point(isovist_object.door_points[vpid].x(), isovist_object.door_points[vpid].y())
                        for svrid, dids in self.regions_doors_info.items():
                            if vpid in dids:
                                vrid = svrid
                        if vrid is not None:
                            self.rview_ids[counter] = [rid1, vrid]
                            self.rviews[counter] = [p, vp]
                            self.rview_ls[counter] = LineString([p, vp])
                            if is_door:
                                if contained[apid] not in self.from_door_vids.keys():
                                    self.from_door_vids[contained[apid]] = []
                                self.from_door_vids[contained[apid]].append(counter)
                            if vpid not in self.to_door_vids.keys():
                                self.to_door_vids[vpid] = []
                            self.to_door_vids[vpid].append(counter)
                            counter += 1

        print('decompose views')
        decomposed_views_dict = {}
        c_views = 0
        for vid in self.rview_ids.keys():
            decomposed_views_dict[vid] = self.decompose_view_disappear(isovist_object, vid)
            c_views += len(decomposed_views_dict[vid])

        drviews = {}
        drview_ids = {}
        drview_ls = {}
        r_dr_mapping_ids = {}
        idx = 0
        for vid_old in self.rview_ids.keys():
            vals = decomposed_views_dict[vid_old]
            r_dr_mapping_ids[vid_old] = []

            for val in vals:
                drview_ids[idx] = val['ids']
                drviews[idx] = val['view']
                drview_ls[idx] = LineString(val['view'])
                r_dr_mapping_ids[vid_old].append(idx)
                idx += 1

        for did, vid_old in self.views_doors_info.items():
            self.views_doors_info[did] = r_dr_mapping_ids[vid_old][0]

        to_door_vids_temp = {}
        for did, vid_olds in self.to_door_vids.items():
            to_door_vids_temp[did] = []
            for vid_old in vid_olds:
                to_door_vids_temp[did].append(r_dr_mapping_ids[vid_old][-1])
        self.to_door_vids = to_door_vids_temp

        from_door_vids_temp = {}
        for did, vid_olds in self.from_door_vids.items():
            from_door_vids_temp[did] = []
            for vid_old in vid_olds:
                from_door_vids_temp[did].append(r_dr_mapping_ids[vid_old][0])
        self.from_door_vids = from_door_vids_temp

        self.from_region_ids = {}
        self.to_region_ids = {}
        for rvid, vals in self.rview_ids.items():
            if vals[0] not in self.from_region_ids.keys():
                self.from_region_ids[vals[0]] = r_dr_mapping_ids[rvid][0]
            if vals[1] not in self.to_region_ids.keys():
                self.to_region_ids[vals[1]] = r_dr_mapping_ids[rvid][-1]
            if len(self.to_region_ids) == len(self.signatures) and len(self.from_region_ids) == len(self.signatures):
                break

        # constructing region view graph
        print('constructing view graph for regions')
        self.rviewgraph = nx.DiGraph()
        self.rviewgraph.add_nodes_from(['{0}-V{1}'.format(self.name, idx) for idx in list(drview_ids.keys())])
        for vid, pids in self.rview_ids.items():
            dviews = [drviews[i] for i in r_dr_mapping_ids[vid]]
            dvids = [i for i in r_dr_mapping_ids[vid]]
            v1 = dviews[len(dvids) - 1]
            dv1 = Utility.calculate_distance(v1[0], v1[1])
            for vid2, pids2 in self.rview_ids.items():
                if vid != vid2:
                    dviews2 = [drviews[i] for i in r_dr_mapping_ids[vid2]]
                    dvids2 = [i for i in r_dr_mapping_ids[vid2]]
                    if pids[1] == pids2[0]:  # movement
                        if self.rviews[vid][1].distance(self.rviews[vid2][0]) < Parameters.epsilon:
                            self.rviewgraph.add_edge(
                                '{0}-V{1}'.format(self.name, dvids[len(dvids) - 1]),
                                '{0}-V{1}'.format(self.name, dvids2[0]),
                                weight=dv1, label='move')
                    elif pids[0] == pids2[0]:  # turn
                        if self.rviews[vid][0].distance(self.rviews[vid2][0]) < Parameters.epsilon:
                            self.rviewgraph.add_edge(
                                '{0}-V{1}'.format(self.name, dvids[0]),
                                '{0}-V{1}'.format(self.name, dvids2[0]),
                                weight=0, label='turn')
                for i in range(1, len(dvids)):
                    v0 = dviews[i - 1]
                    v1 = dviews[i]
                    dv0 = Utility.calculate_distance(v0[0], v0[1])
                    self.rviewgraph.add_edge(
                        '{0}-V{1}'.format(self.name, dvids[i - 1]),
                        '{0}-V{1}'.format(self.name, dvids[i]), weight=dv0, label='move')

        self.rview_ids = drview_ids
        self.rviews = drviews
        self.rview_ls = drview_ls

        # spatial relationships
        print('calculating all spatial relationships visible in each view')
        self.srelations = {}
        for vid in self.rview_ids.keys():
            self.srelations[vid] = self.calculate_spatial_relationships(vid, isovist_object)

        # label nodes (views)
        print('Adding actions to views (nodes)')  # pass
        self.v_attributes = {}
        for vid in self.rviews.keys():
            srelation = self.srelations[vid]
            l_action = {}
            r_action = {}
            f_action = {}
            for object, info in srelation.items():
                if 'left' in info['dir']:
                    l_action[info['order']] = object
                elif 'right' in info['dir']:
                    r_action[info['order']] = object
                else:
                    f_action[info['order']] = object
            self.v_attributes['{0}-V{1}'.format(self.name, vid)] = {'l_action': [l[1] for l in sorted(l_action.items())],
                                      'f_action': [f[1] for f in sorted(f_action.items())],
                                      'r_action': [r[1] for r in sorted(r_action.items())]}
        nx.set_node_attributes(self.rviewgraph, self.v_attributes)

        # label edges (view->view)
        print('Adding actions to view relations (edges)')

        self.r_attributes = {}
        for vid in self.rviews.keys():
            bearing1 = Utility.calculate_bearing(self.rviews[vid])
            for vv2, attributes in dict(self.rviewgraph['{0}-V{1}'.format(self.name, vid)]).items():
                if attributes['label'] == 'move':
                    vid2 = int(vv2.replace('{0}-V'.format(self.name), ''))
                    bearing2 = Utility.calculate_bearing(self.rviews[vid2])
                    vv1 = '{0}-V{1}'.format(self.name, vid)
                    if abs(bearing1 - bearing2) <= Parameters.alpha:
                        self.r_attributes[(vv1, vv2)] = {'action': 'follow'}
                    elif 180 - Parameters.alpha <= abs(bearing1 - bearing2) <= 180 + Parameters.alpha:
                        self.r_attributes[(vv1, vv2)] = {'action': 'turn back'}
                    elif bearing1 > bearing2:
                        if bearing1 - bearing2 > 180 + Parameters.alpha:
                            self.r_attributes[(vv1, vv2)] = {'action': 'veer left'}
                        else:
                            self.r_attributes[(vv1, vv2)] = {'action': 'turn left'}
                    else:
                        if bearing2 - bearing1 > 180 - Parameters.alpha:
                            self.r_attributes[(vv1, vv2)] = {'action': 'veer right'}
                        else:
                            self.r_attributes[(vv1, vv2)] = {'action': 'turn right'}
        nx.set_edge_attributes(self.rviewgraph, self.r_attributes)

    def vision_triangle(self, view_id):
        v = self.rviews[view_id]
        p1 = v[0]
        p2 = Utility.calculate_coordinates(v=v, angle=Parameters.fov / 2, d=Parameters.max_distance)
        p3 = Utility.calculate_coordinates(v=v, angle=-Parameters.fov / 2, d=Parameters.max_distance)
        return Poly([[p.x, p.y] for p in [p1, p2, p3]])

    def which_region(self, point):
        for idx, r in enumerate(self.regions_list):
            if r.contains(point) or point.touches(r):
                return idx

    def view_vision(self, isovist_object, view_idx, is_start=True, isovist_view=None):
        triangle = self.vision_triangle(view_idx)
        view_line = self.rview_ls[view_idx]
        if isovist_view is None:
            if is_start:
                x = view_line.xy[0][0]
                y = view_line.xy[1][0]
            else:
                x = view_line.xy[0][1]
                y = view_line.xy[1][1]
            isovist_view = isovist_object.isovist_calc(x, y)
        return isovist_view.intersection(triangle)

    def view_vision_signature(self, view_coverage, door_points):
        signature = []
        for idx, p in enumerate(door_points):
            if view_coverage.contains(Point(p.x(), p.y())) or view_coverage.touches(Point(p.x(), p.y())):
                signature.append(idx)
        return signature

    def ego_order(self, view_points, points):
        start = view_points[0]
        end = Utility.calculate_coordinates(view_points, 0, 20)
        line = LineString([start, end])
        distances = {}
        for idx, p in points.items():
            d = line.project(p)
            distances[idx] = d
        return dict(sorted(distances.items(), key=lambda item: item[1]))

    def decompose_view_disappear(self, isovist_object, view_idx, plot=False):
        decomposed = []
        view = self.rviews[view_idx]
        ids = self.rview_ids[view_idx]
        destinations = []
        if ids[1] in self.regions_doors_info.keys():
            destinations = self.regions_doors_info[ids[1]]
        view_line = self.rview_ls[view_idx]
        vv = self.view_vision(isovist_object, view_idx)
        points = {'end': view[1]}
        door_signature = self.view_vision_signature(vv, isovist_object.door_points)
        for dix in door_signature:
            points['door {}'.format(dix)] = Point(isovist_object.door_points[dix].x(),
                                                  isovist_object.door_points[dix].y())
        landmark_signature = self.view_vision_signature(vv, door_points=isovist_object.landmarks_points)
        for lix in landmark_signature:
            points[self.landmark_info[lix]] = Point(isovist_object.landmarks_points[lix].x(),
                                                      isovist_object.landmarks_points[lix].y())
        orders = self.ego_order(view, points)
        disappear_points = []
        for key, d in orders.items():
            if key == 'end':
                break
            for destination in destinations:
                if destination is not None and 'door {}'.format(destination) == key:
                    break
            disappear_points.append(view_line.interpolate(d - self.disappear_shift(view_idx, d) + Parameters.epsilon))
        if len(disappear_points) > 0:
            rid1 = self.which_region(disappear_points[0])
            decomposed = [{'ids': [ids[0], rid1], 'view': [view[0], disappear_points[0]]}]
            for i in range(1, len(disappear_points)):
                rid2 = self.which_region(disappear_points[i])
                decomposed.append({'ids': [rid1, rid2], 'view': [disappear_points[i - 1], disappear_points[i]]})
                rid1 = rid2
            rid2 = ids[1]
            decomposed.append({'ids': [rid1, rid2], 'view': [disappear_points[len(disappear_points) - 1], view[1]]})
        else:
            decomposed = [{'ids': ids, 'view': view}]
        return decomposed

    def disappear_shift(self, vid, d, fov=Parameters.fov):
        view_line = self.rview_ls[vid]
        point = view_line.interpolate(d)
        p1, p2 = nearest_points(view_line, point)
        a = (90 - fov / 2) / 180 * math.pi
        shift = tan(a) * point.distance(p1)
        return shift

    def shortest_path_regions(self, rid1, rid2, isvid=False):
        if isvid:
            vid1 = rid1
            vid2 = rid2
        else:
            vid1 = self.from_region_ids[rid1]
            vid2 = self.from_region_ids[rid2]

        vpath = nx.shortest_path(self.rviewgraph, '{0}-V{1}'.format(self.name, vid1),
                                 '{0}-V{1}'.format(self.name, vid2),
                                 weight='weight')
        vpath_temp = [int(vv.replace('{0}-V'.format(self.name), '')) for vv in vpath]
        vpath = vpath_temp
        path_view = []

        if not isvid:
            vpath = vpath[:-1]
            if len(vpath) >= 2 and self.rview_ids[vpath[0]][0] == self.rview_ids[vpath[1]][0]:
                vpath = vpath[1:]

        for vid in vpath:
            path_view.append(self.rviews[vid])

        return vpath, path_view

    def calculate_spatial_relationships(self, vid, isovist_object):
        vv = self.view_vision(isovist_object, vid)
        door_signature = self.view_vision_signature(vv, isovist_object.door_points)
        # print(door_signature)
        landmark_signature = self.view_vision_signature(vv, door_points=isovist_object.landmarks_points)
        # print(landmark_signature)

        view_points = self.rviews[vid]
        points = {}
        for d in door_signature:
            d_point = isovist_object.door_points[d]
            points[self.door_info[d]] = Point(d_point.x(), d_point.y())
        for l in landmark_signature:
            l_point = isovist_object.landmarks_points[l]
            points[self.landmark_info[l]] = Point(l_point.x(), l_point.y())
        ego_rels = self.egocentric_relationships(view_points, points)
        return ego_rels

    def ego_dir_det(self, a, b, c):
        det = ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))
        if det > 0:
            return 'on the left'
        elif det == 0:
            return 'in the front'
        else:
            return 'on the right'

    def ego_dir(self, a, b, c):
        angle1 = Utility.calculate_bearing([a, b])
        angle2 = Utility.calculate_bearing([a, c])
        if abs(angle2 - angle1) < Parameters.alpha / 2:
            return 'in the front'
        else:
            return self.ego_dir_det(a, b, c)

    def egocentric_relationships(self, view_points, points):
        dirs = {}
        lefts = {}
        rights = {}
        fronts = {}
        for idx, p in points.items():
            dir_rel = self.ego_dir(view_points[0], view_points[1], p)
            if dir_rel == 'on the left':
                lefts[idx] = p
            elif dir_rel == 'on the right':
                rights[idx] = p
            else:
                fronts[idx] = p
            dirs[idx] = {'dir': dir_rel, 'order': None}
        left_orders = self.ego_order(view_points, lefts)
        right_orders = self.ego_order(view_points, rights)
        front_orders = self.ego_order(view_points, fronts)
        counter = 1
        for k, v in left_orders.items():
            dirs[k]['order'] = counter
            counter += 1
        counter = 1
        for k, v in right_orders.items():
            dirs[k]['order'] = counter
            counter += 1
        counter = 1
        for k, v in front_orders.items():
            dirs[k]['order'] = counter
            counter += 1
        return dirs

    def minimal_description_path(self, attrs, ids):
        instructions = []
        for idx, vid in enumerate(ids):
            if idx < len(ids) - 1:
                vid2 = ids[idx + 1]
                if vid2 - vid == 1:  # possibly decomposed...
                    attr = attrs[idx + 1]
                    attr2 = attrs[idx]
                    for key, objects in attr2.items():
                        if len(attr2[key]) - len(attr[key]) == 1:
                            current_objects = attr[key]
                            for object in objects:
                                if object not in current_objects:
                                    instructions = ['Pass {}'.format(object)]
        return instructions

    def minimal_description_follow(self, attrs, ids=[]):
        if len(ids) > 0:
            mdp = self.minimal_description_path(attrs, ids)
            if len(mdp) > 0:
                return mdp
        instructions = []
        objects_dict = {}
        idx_dict = {}
        max_idx = -1
        for a_idx, attr in enumerate(attrs):
            for key, objects in attr.items():
                heading = key.split('_')[0]
                for object in objects:
                    if object not in objects_dict.keys():
                        objects_dict[object] = [a_idx, a_idx, heading]
                        if a_idx not in idx_dict.keys():
                            idx_dict[a_idx] = []
                        idx_dict[a_idx].append(object)
                    else:
                        objects_dict[object][1] = a_idx
                    if a_idx > max_idx:
                        max_idx = a_idx

        min_idx = min(list(idx_dict.keys()))
        current_idx = min_idx
        while current_idx < max_idx:
            objects = idx_dict[current_idx]
            info_list = []
            for object in objects:
                info_list.append(objects_dict[object])
            max_range = -1
            max_info = None
            max_object_idx = -1
            for object_idx, info in enumerate(info_list):
                if info[1] - info[0] > max_range:
                    max_range = info[1] - info[0]
                    max_info = info
                    max_object_idx = object_idx
            if max_info is not None:
                heading = 'right'
                if max_info[2] == 'f':
                    heading = 'front'
                elif max_info[2] == 'l':
                    heading = 'left'
                instructions.append('Follow {0} on the {1}'.format(objects[max_object_idx], heading))
            temp_idx = max_info[1]
            if temp_idx == max_idx:
                break
            elif temp_idx in idx_dict.keys():
                current_idx = temp_idx
            else:
                while temp_idx not in idx_dict.keys() and temp_idx > current_idx:
                    temp_idx -= 1
                if temp_idx == current_idx:
                    break
                else:
                    current_idx = temp_idx
        return instructions

    def generate_route_description(self, vpath):
        instructions = []
        v_attrs = []
        for v in vpath:
            v_attr = self.v_attributes['{0}-V{1}'.format(self.name, v)]
            v_attrs.append(v_attr)

        start = v_attrs[0]
        if len(start['f_action']) > 0:
            instructions.append('Head towards {}'.format(start['f_action'][0]))
        elif len(start['l_action']) > 0:
            instructions.append('Start with {} on your left'.format(start['l_action'][0]))
        elif len(start['r_action']) > 0:
            instructions.append('Start with {} on your right'.format(start['r_action'][0]))

        r_attrs = []
        for idx, v in enumerate(vpath):
            if idx != len(vpath) - 1:
                r_attr = self.r_attributes[('{0}-V{1}'.format(self.name, v),
                                            '{0}-V{1}'.format(self.name, vpath[idx + 1]))]
                r_attrs.append(r_attr)
        temp = []
        temp_vids = []
        for idx, r_attr in enumerate(r_attrs):
            start = v_attrs[idx]
            end = v_attrs[idx + 1]
            if 'follow' in r_attr['action']:
                temp.append(start)
                temp_vids.append(vpath[idx])
            else:
                if len(temp) > 0:
                    instructions.extend(self.minimal_description_follow(temp, temp_vids))
                    temp = []
                    temp_vids = []
                    instructions[len(instructions) - 1] = instructions[len(instructions) - 1] + \
                                                          ' and ' + r_attr['action']
                else:
                    act = r_attr['action']
                    if act not in instructions[len(instructions) - 1]:
                        instructions.append('move further and ' + act + ' in the first decision point')
        if len(temp) > 0:
            instructions.extend(self.minimal_description_follow(temp, temp_vids))
        instructions[len(instructions)-1] = instructions[len(instructions) - 1] \
                                            + ' and move forward until you reach the destination'
        Utility.print_instructions(instructions)
        return instructions

    def generate_door_to_door_graph(self, isovist_object, only_doors=False):
        print('generate door-to-door graph, only_doors {} from view graph'.format(only_doors))
        dtdgraph= nx.Graph()
        dids = []
        edges = []
        connected = []
        alreadythere = []
        skip_gateways = []
        if only_doors:
            for idx, props in isovist_object.door_props.items():
                if props['type'] == 'dt':
                    skip_gateways.append(idx)
        for rid, doors in self.regions_doors_info.items():
            for d in doors:
                if d not in skip_gateways:
                    dids.append(d)
                    rsignature = self.signatures[rid]
                    for didx in rsignature:
                        if didx != d and str(d) + '-' + str(didx) not in alreadythere \
                                and str(didx) + '-' + str(d) not in alreadythere and didx not in skip_gateways:
                            connected.append([isovist_object.door_points[d], isovist_object.door_points[didx]])
                            edges.append((d, didx))
                            alreadythere.append(str(d) + '-' + str(didx))
                            alreadythere.append(str(didx) + '-' + str(d))
        did_attributes = []
        for d in list(set(dids)):
            dtype = isovist_object.door_props[d]['type']
            attrs = {}
            if dtype == 'dt':
                dtype = 'decision point'
                attrs = {'type': dtype, 'group': 1, 'label': self.door_info[d]}
            else:
                dtype = 'door'
                attrs = {'type': dtype, 'group': 2, 'label': self.door_info[d]}
            did_attributes.append((d, attrs))

        dtdgraph.add_nodes_from(did_attributes)
        dtdgraph.add_edges_from(edges)
        return connected, dtdgraph

    def get_door_region(self, did):
        for rid, dids in self.regions_doors_info.items():
            if did in dids:
                return rid

    def generate_navigation_graph(self, isovist_object, indirect_access=False):
        print('derive navigation graph using spanning tree from viewgraph')
        all_vps = []
        all_pvs = []

        path_graph = nx.complete_graph(len(isovist_object.door_points)
                                        -isovist_object.door_idx)

        all_vps_info = {}
        for did1, vid1 in self.views_doors_info.items():
            rid1 = self.get_door_region(did1)
            for did2, vid2 in self.views_doors_info.items():

                if did1 >= isovist_object.door_idx and did2 >= isovist_object.door_idx \
                        and vid1 != vid2:
                    if indirect_access or did2 in self.signatures[rid1]:
                        vp, pv = self.shortest_path_regions(vid1, vid2, isvid=True)
                        vp_tmp = ['{0}-V{1}'.format(self.name, v) for v in vp]
                        all_vps_info[len(all_vps)] = {'from': did1, 'to': did2, 'index': len(all_vps),
                                                      'length': nx.path_weight(self.rviewgraph, vp_tmp,
                                                                               weight='weight')}
                        path_graph[did1 - isovist_object.door_idx][did2 - isovist_object.door_idx]['weight'] = \
                            all_vps_info[len(all_vps)]['length']
                        all_vps.append(vp)
                        all_pvs.append(pv)
                    else:
                        path_graph[did1 - isovist_object.door_idx][did2 - isovist_object.door_idx]['weight'] = 100000
        T = nx.minimum_spanning_tree(path_graph, weight='weight')
        st = sorted(T.edges(data=True))
        spt_vps = []
        spt_pvs = []
        connections = {}
        connections_details = {}

        attrs = {
            node: {'label': self.door_info[node + isovist_object.door_idx], 'idx': node + isovist_object.door_idx,
                   'group': 'decision points'} for node in range(len(T.nodes))}
        nx.set_node_attributes(T, attrs)

        for record in st:
            idx1 = record[0]
            idx2 = record[1]
            idx1 += isovist_object.door_idx
            idx2 += isovist_object.door_idx
            spt_vp, spt_pv = self.shortest_path_regions(self.views_doors_info[idx1],
                                                        self.views_doors_info[idx2],
                                                        True)
            if record[0]+isovist_object.door_idx not in connections.keys():
                connections[idx1] = 0
                connections_details[idx1] = []
            connections[idx1] += 1
            connections_details[idx1].append(idx2)


            if idx2 not in connections.keys():
                connections[idx2] = 0
                connections_details[idx2] = []
            connections[idx2] += 1
            connections_details[idx2].append(idx1)

            spt_vps.append(spt_vp)
            spt_pvs.append(spt_pv)

        for door in range(isovist_object.door_idx):
            dvid = self.views_doors_info[door]
            max_weight = 10000
            selected_vp = None
            selected_pv = None
            for vids in spt_vps:
                for vid in vids:
                    try:
                        vp, pv = self.shortest_path_regions(dvid, vid, True)
                        vp_temp = ['{0}-V{1}'.format(self.name, idx) for idx in vp]
                        w = nx.path_weight(self.rviewgraph, vp_temp, weight='weight')
                        if w < max_weight:
                            max_weight = w
                            selected_vp = vp
                            selected_pv = pv
                    except:
                        print('no path from {0} to {1}'.format(door, vid))
            if selected_vp is not None:
                spt_pvs.append(selected_pv)
                spt_vps.append(selected_vp)

        return all_vps, all_pvs, spt_vps, spt_pvs, T


    @staticmethod
    def generate_titles(dict_dict):
        for kr, record in dict_dict.items():
            title = ''
            for key, val in record.items():
                if key not in ['in', 'group']:
                    title+='[{0}: {1}] '.format(key, val)
            record['title'] = title


    @staticmethod
    def generate_between_near(lefts, relationships_investigated, nplets, ncounter, references, bearing):
        if len(lefts) > 2:
            for l1 in lefts:
                for l2 in lefts:
                    for l3 in lefts:
                        if l1 != l2 != l3:
                            l1r = list(l1.keys())[0]
                            l2r = list(l2.keys())[0]
                            l3r = list(l3.keys())[0]
                            o1 = l1[l1r]['order']
                            o2 = l2[l2r]['order']
                            o3 = l3[l3r]['order']
                            if o1 < o2 < o3:
                                expression = '{0} between {1} and {2}'.format(l2r, l1r, l3r)
                                if expression not in relationships_investigated:
                                    nplets['n{}'.format(ncounter)] = {
                                        'exp': '{0} between {1} and {2}'.format(l2r, l1r, l3r),
                                        'reference_frame': 'relative',
                                        'bearing': bearing,
                                        'sp_relation': 'between',
                                        'group':1}
                                    references[l2r]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'locatum'})
                                    references[l1r]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'relatum'})
                                    references[l3r]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 2, 'as': 'relatum'})
                                    ncounter += 1
                                    relationships_investigated.append(expression)
                                if o2 - o1 == 1:
                                    expression = '{0} near {1}'.format(l2r, l1r)
                                    if expression not in relationships_investigated:
                                        nplets['n{}'.format(ncounter)] = {
                                            'exp': '{0} near {1}'.
                                            format(l2r, l1r), 'reference_frame': 'relative',
                                            'sp_relation': 'near',
                                            'group':1}
                                        references[l2r]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'locatum'})
                                        references[l1r]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'relatum'})
                                        ncounter += 1
                                        relationships_investigated.append(expression)
                                        relationships_investigated.append('{0} near {1}'.format(l1r, l2r))
                                if o3 - o2 == 1:
                                    expression = '{0} near {1}'.format(l3r, l2r)
                                    if expression not in relationships_investigated:
                                        nplets['n{}'.format(ncounter)] = {
                                            'exp': '{0} near {1}'.
                                                format(l3r, l2r), 'reference_frame': 'relative',
                                            'sp_relation': 'near',
                                            'group':1}
                                        references[l3r]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'locatum'})
                                        references[l2r]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'relatum'})
                                        ncounter += 1
                                        relationships_investigated.append(expression)
                                        relationships_investigated.append('{0} near {1}'.format(l2r, l3r))
        return ncounter

    def generate_place_graph(self, isovist_object):
        print('derive place graph from view graph')
        relationships_investigated = []
        nplets = {}  # nid: {exp: '', rframe: '', sp_relation: sid, place: {id: pid, as: 'f/b'}
        references = {}  # reference: {place: pid, in: [{nplet: nid, pos: int, as: 'r/l'}]}
        pids = {}  # pid: {type: d/l/dt}
        pcounter = 0
        for d, door in enumerate(isovist_object.door_points):
            if d < isovist_object.door_idx:
                pids['place{}'.format(pcounter)] = {'type': 'door', 'group':2}
            else:
                pids['place{}'.format(pcounter)] = {'type': 'decision point', 'group':2}
            references[self.door_info[d]] = {'place': 'place{}'.format(pcounter), 'in': [], 'group':3}
            pcounter+=1
        for l, landmark in enumerate(isovist_object.landmarks_points):
            pids['place{}'.format(pcounter)] = {'type': 'landmark', 'group':2}
            references[self.landmark_info[l]] = {'place': 'place{}'.format(pcounter), 'in': [], 'group':3,
                                                 'label': self.landmark_info[l]}
            pcounter+=1
        sids = {
            # 'front': {'relation': 'front', 'family': 'relative direction'},
            'left': {'relation': 'left', 'family': 'relative direction', 'group':4},
            'right': {'relation': 'right', 'family': 'relative direction', 'group':4},
            'between': {'relation': 'between', 'family': 'ternary', 'group':4},
            # 'across': {'relation': 'across', 'family': 'ternary', 'group':4},
            # 'inside': {'relation': 'in', 'family': 'topological', 'group':4},
            # 'disjoint': {'relation': 'disjoint', 'family': 'topological', 'group':4},
            'near': {'relation': 'near', 'family': 'distance', 'group':4}
            }
        # sid: {relation: '', family: ''}  # incomplete

        ncounter = 0
        for vid, relationships in self.srelations.items():
            bearing = Utility.calculate_bearing(self.rviews[vid])
            if len(relationships) > 0:
                fronts = []
                lefts = []
                rights = []
                for place, relation in relationships.items():
                    if 'dir' in relation.keys():
                        if 'front' in relation['dir']:
                            fronts.append({place: relation})
                        elif 'right' in relation['dir']:
                            rights.append({place: relation})
                        else:
                            lefts.append({place: relation})

                # directional
                for f in fronts:
                    for l in lefts:
                        for r in rights:
                            fp = references[list(f.keys())[0]]['place']
                            lpr = list(l.keys())[0]
                            rpr = list(r.keys())[0]
                            expression = '{0} left of {1}'.format(lpr, rpr)
                            if expression not in relationships_investigated:
                                nplets['n{}'.format(ncounter)] = {
                                    'exp': '{0} left of {1}'.format(lpr, rpr),
                                    'reference_frame': 'relative',
                                    'sp_relation': 'left',
                                    'place': {'id': fp, 'as': 'front', 'bearing': bearing},
                                    'group':1}
                                references[lpr]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'locatum'})
                                references[rpr]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'relatum'})
                                ncounter+=1
                                nplets['n{}'.format(ncounter)] = {
                                    'exp': '{0} right of {1}'.format(rpr, lpr),
                                    'reference_frame': 'relative',
                                    'sp_relation': 'right',
                                    'place': {'id': fp, 'as': 'front', 'bearing': bearing},
                                    'group':1}
                                references[lpr]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'relatum'})
                                references[rpr]['in'].append({'nid': 'n{}'.format(ncounter), 'pos': 1, 'as': 'locatum'})
                                ncounter += 1
                                relationships_investigated.append(expression)
                                relationships_investigated.append('{0} right of {1}'.format(rpr, lpr))

                # between and near
                ncounter = ViewGraph.generate_between_near(lefts, relationships_investigated, nplets, ncounter,
                                                           references, bearing)
                ncounter = ViewGraph.generate_between_near(rights, relationships_investigated, nplets, ncounter,
                                                           references, bearing)
                ncounter = ViewGraph.generate_between_near(fronts, relationships_investigated, nplets, ncounter,
                                                           references, bearing)

        place_graph = nx.DiGraph()
        # nodes: nplet (nid), place (pid), reference (rid), sp_rel (sid)
        ViewGraph.generate_titles(nplets)
        place_graph.add_nodes_from([(nid, attrs) for nid, attrs in nplets.items()])
        ViewGraph.generate_titles(pids)
        place_graph.add_nodes_from([(pid, attrs) for pid, attrs in pids.items()])
        ViewGraph.generate_titles(sids)
        place_graph.add_nodes_from([(sid, attrs) for sid, attrs in sids.items()])
        ViewGraph.generate_titles(references)
        place_graph.add_nodes_from([(rid, {'group': 3, 'title': references[rid]['title']})
                                    for rid in list(references.keys())])


        for r, vals in references.items():  # relations: reference -> pid, reference -> nid(s)
            pid = vals['place']
            place_graph.add_edge(pid, r)
            place_graph[pid][r]['label'] = 'referred by'
            in_list = vals['in']
            for in_rec in in_list:
                nid = in_rec['nid']
                place_graph.add_edge(r, nid)
                place_graph[r][nid]['pos'] = in_rec['pos']
                place_graph[r][nid]['as'] = in_rec['as']
                place_graph[r][nid]['label'] = '{0}-{1}'.format(in_rec['as'], in_rec['pos'])
        for nid, vals in nplets.items():  # relations: nid -> sid, {nid -> pid}
            sid = vals['sp_relation']
            place_graph.add_edge(nid, sid)
            place_graph[nid][sid]['label'] = 'map'
            if 'place' in vals.keys():
                pid = vals['place']['id']
                place_graph.add_edge(nid, pid)
                place_graph[nid][pid]['as'] = vals['place']['as']
                place_graph[nid][pid]['label'] = 'has_reference_direction {}'.format(vals['place']['as'])
        return place_graph
