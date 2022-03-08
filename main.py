########################################################################
# views [spoint -> spoint]
########################################################################
import statistics
import math
import geojson
import visilibity as vis
import matplotlib.pyplot as plt
from geojson import Polygon, FeatureCollection, Feature
from shapely.geometry import shape, Point, LineString, MultiPolygon, LinearRing
from shapely.geometry import Polygon as Poly
from shapely.ops import unary_union, polygonize, nearest_points
import networkx as nx
import geopy.distance as distance
from numpy import arctan2, sin, cos, degrees
import geopandas as gpd

# address = '/Users/ehsanhamzei/Desktop/PostDoc/Floorplans/Melbourne Connect/'
# polygon_file = 'study_area_all.geojson'
# holes_file = 'study_area_holes.geojson'
# doors_file = 'study_doors.geojson'
# dpoint_file = 'study_area_dpoints.geojson'
address = 'envs/hypo/'
polygon_file = 'hypo_env.geojson'
holes_file = 'hypo_holes.geojson'
doors_file = 'hypo_doors.geojson'
landmarks = 'hypo_landmarks.geojson'

test = False
test_regions = True


def read_geojson(address, file):
    with open(address+file, encoding='utf-8') as fp:
        file = geojson.load(fp)
    return file


def reformat_point(record):
    return vis.Point(record[0],record[1])


def signed_area(pr2):
    xs, ys = map(list, zip(*pr2))
    xs.append(xs[1])
    ys.append(ys[1])
    return sum(xs[i] * (ys[i + 1] - ys[i - 1]) for i in range(1, len(pr2))) / 2.0


def reformat_polygon(records_original, is_hole=True):
    points = []
    records_rev = records_original.copy()
    if records_rev[-1] == records_rev[0]:
        records_rev.pop()

    if is_hole:
        records = records_rev
    else:
        records = list(reversed(records_rev))
    for record in records:
        points.append(reformat_point(record))
    return vis.Polygon(points)


def create_env(space_polygon, holes_polygons):
    env_list = [space_polygon]
    for hole in holes_polygons:
        env_list.append(hole)
    return vis.Environment(env_list)


def save_print(polygon):
    end_pos_x = []
    end_pos_y = []
    for i in range(polygon.n()):
        x = polygon[i].x()
        y = polygon[i].y()

        end_pos_x.append(x)
        end_pos_y.append(y)
    return end_pos_x, end_pos_y


def save_print_geojson(polygon):
    x = []
    y = []
    for p_x_y in polygon:
        x.append(p_x_y[0])
        y.append(p_x_y[1])
    return x, y


def to_polygon_geojson(x_list, y_list):
    formatted_list = [(x_list[i], y_list[i]) for i in range(len(x_list))]
    return Polygon([formatted_list])


def to_polygon_shape(polygon_geojson, clip = False):
    shp = shape(polygon_geojson)
    if not shp.is_valid:
        print('not valid: {}'.format(polygon_geojson))
    elif clip:
        shp = shp.intersection(space_shp)
        for hole in holes:
            shp = shp.difference(shape(hole['geometry']))
    return shp


def calculate_distance(d1, d2):
    return distance.distance((d1.y, d1.x), (d2.y, d2.x)).km * 1000

def calculate_bearing(v):
    lat1 = v[0].y
    lat2 = v[1].y
    lon1 = v[0].x
    lon2 = v[1].x
    dL = lon2 - lon1
    X = cos(lat2) * sin(dL)
    Y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dL)
    return (degrees(arctan2(X, Y))+360) % 360

def calculate_coordinates(v, angle, d):
    bearing = calculate_bearing(v)
    nbearing = bearing+angle
    td = distance.GeodesicDistance(kilometers=d)
    res = td.destination((v[0].y, v[0].x), nbearing)
    return Point(res.longitude, res.latitude)

def slope(x1, y1, x2, y2):  # Line slope given two points:
    return (y2-y1)/(x2-x1)

def angle(s1, s2):
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

def calculate_angle(p1, p2, p3):
    slope1 = slope(p1[0], p1[1], p2[0], p2[1])
    slope2 = slope(p1[0], p1[1], p3[0], p3[1])
    return angle(slope1, slope2)


epsilon = 0.000001
print('reading GeoJSON files (boundary, holes, doors and decision points)')
boundary = read_geojson(address, polygon_file)['features'][0]['geometry']['coordinates'][0][0]
holes = read_geojson(address, holes_file)['features']
doors = read_geojson(address, doors_file)['features']
# dpoints = read_geojson(address, dpoint_file)['features']
landmarks = read_geojson(address, landmarks)['features']

space_poly = reformat_polygon(boundary, is_hole=False)
space_x, space_y = save_print_geojson(boundary)
space_shp = to_polygon_shape(to_polygon_geojson(space_x, space_y))

holes_list = []
holes_shape = []
holes_x = []
holes_y = []
for h in holes:
    holes_list.append(reformat_polygon(h['geometry']['coordinates'][0][0]))
    h_x, h_y = save_print_geojson(h['geometry']['coordinates'][0][0])
    holes_x.append(h_x)
    holes_y.append(h_y)
    holes_shape.append(shape(h['geometry']).geoms[0])

env = create_env(space_poly, holes_list)

door_points = []
for d in doors:
    door_points.append(reformat_point(d['geometry']['coordinates']))
door_idx = len(door_points)
# for d in dpoints:
#     door_points.append(reformat_point(d['geometry']['coordinates']))
landmarks_points = []
for l in landmarks:
    landmarks_points.append(reformat_point(l['geometry']['coordinates']))

print('calculating the isovist polygons')
isovists = []
door_props = {}
idx = 0
for door in door_points:
    door.snap_to_boundary_of(env, epsilon)
    door.snap_to_vertices_of(env, epsilon)
    isovist = vis.Visibility_Polygon(door, env, epsilon)
    isovists.append(isovist)
    door_props[idx] = {'id': idx, 'y': door.y(), 'x': door.x()}
    if idx < door_idx:
        door_props[idx]['type'] = 'gateway'
    else:
        door_props[idx]['type'] = 'dt'
    idx += 1


features = []
isovists_x_y = []
shapes = {}
idx = 0
test_case = None
test_idx = None
for isovist in isovists: #
    iso_x, iso_y = save_print(isovist)
    geojson_polygon = to_polygon_geojson(iso_x, iso_y)
    shp = to_polygon_shape(geojson_polygon, clip=True)
    if not shp.is_valid:
        print('invalid')
    if isinstance(shp, MultiPolygon):
        test_case = shp
        shps = list(shp)
        areas = [s.area for s in shps]
        max_area = max(areas)
        idxs = areas.index(max_area)
        shp = shps[idxs]
        test_idx = idx
    shapes[idx] = {'shape': shp, 'props': door_props[idx]}

    # iso_x, iso_y = save_print_geojson(list(shp.exterior.coords))
    # iso_x.append(isovist[0].x())
    # iso_y.append(isovist[0].y())

    geojson_polygon = to_polygon_geojson(iso_x, iso_y)
    features.append(Feature(geometry=geojson_polygon, properties=door_props[idx]))

    isovists_x_y.append([iso_x, iso_y])
    idx += 1

feature_collection = FeatureCollection(features)
# counter = 0
# for x_y in isovists_x_y:
#     vantage_point = door_points[counter]
#     plt.plot(space_x, space_y, 'black')
#     plt.plot(x_y[0], x_y[1])
#     for i in range(0, len(holes_x)):
#         plt.plot(holes_x[i], holes_y[i], 'r')
#
#     plt.plot([vantage_point.x()], [vantage_point.y()], 'go')
#     plt.axis('equal')
#     plt.savefig('img/isovist-{0}-{1}.pdf'.format(counter, door_props[counter]['type']))
#     plt.close()
#     plt.cla()
#     plt.clf()
#     counter += 1

views = {}
view_ids = {}
view_counter = 0
for idx, info in shapes.items():
    start = door_props[idx]  # x, y
    shp = info['shape']
    for d in door_points:
        if idx != door_points.index(d):
            tmp_point = Point(d.x(), d.y())
            if shp.contains(tmp_point) or tmp_point.touches(shp):
                views[view_counter] = [Point(start['x'], start['y']), tmp_point]
                view_ids[view_counter] = [idx, door_points.index(d)]
                view_counter += 1
fig, ax = plt.subplots()
plt.plot(space_x, space_y, 'black')
for i in range(0, len(holes_x)):
    plt.plot(holes_x[i], holes_y[i], 'r')
for d in door_points:
    plt.plot([d.x()], [d.y()], 'go')
plt.show()
plt.close()
plt.cla()
plt.clf()

print('constructing object-based visibility graph')
viewgraph = nx.DiGraph()
viewgraph.add_nodes_from(list(view_ids.keys()))
for vid, pids in view_ids.items():
    v1 = views[vid]
    dv1 = calculate_distance(v1[0], v1[1])
    for vid2, pids2 in view_ids.items():
        if vid != vid2:
            if pids[1] == pids2[0]:
                viewgraph.add_edge(vid, vid2, weight=dv1)
            elif pids[0] == pids2[0]:
                viewgraph.add_edge(vid, vid2, weight=0)


def find_smallest_view(point_id, to_point=True):
    minimum_view = 1000
    selected_view = None
    for idx, view_id in view_ids.items():
        if (to_point and view_id[1] == point_id) or (not to_point and view_id[0] == point_id):
            v1 = views[idx]
            dv1 = calculate_distance(v1[0], v1[1])
            if dv1 < minimum_view or selected_view is None:
                selected_view = idx
                minimum_view = dv1
    return selected_view


def find_all_views(pid, to_point=True):
    selected_views = []
    for idx, view_id in view_ids.items():
        if (to_point and view_id[1] == pid) or (not to_point and view_id[0] == pid):
            selected_views.append(views[idx])
    return selected_views


def shortest_path(pid1, pid2):
    vid1 = find_smallest_view(pid1)
    vid2 = find_smallest_view(pid2, to_point=False)
    vpath = nx.shortest_path(viewgraph, vid1, vid2, weight='weight')
    path_view = []
    # print(vpath)
    vpath = vpath[:-1]
    vpath = vpath[1:]
    # print(vpath)
    for vid in vpath:
        path_view.append(views[vid])
    X = []
    Y = []
    U = []
    V = []
    for view in path_view:
        X.append(view[0].x)
        Y.append(view[0].y)
        U.append(view[1].x - view[0].x)
        V.append(view[1].y - view[0].y)
    fig, ax = plt.subplots()
    plt.plot(space_x, space_y, 'black')
    for i in range(0, len(holes_x)):
        plt.plot(holes_x[i], holes_y[i], 'r')
    for d in door_points:
        plt.plot([d.x()], [d.y()], 'go')
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


def plot_views_isovist(pid, to_point=False):
    selected_views = find_all_views(pid, to_point)
    X = []
    Y = []
    U = []
    V = []
    for view in selected_views:
        X.append(view[0].x)
        Y.append(view[0].y)
        U.append(view[1].x - view[0].x)
        V.append(view[1].y - view[0].y)
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    plt.plot(space_x, space_y, 'black')
    for i in range(0, len(holes_x)):
        plt.plot(holes_x[i], holes_y[i], 'r')
    for d in door_points:
        plt.plot([d.x()], [d.y()], 'go')
    x_y = list(shapes_list[pid].exterior.xy)
    plt.plot(x_y[0], x_y[1])
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


def view_intersects_holes(view_ls):
    for hole in holes_shape:
        if hole.intersects(view_ls):
            return True
    return False


def plot_view_sequence(vid, to_view=False, two_sidded=False, turns=False):
    chosen_view = views[vid]
    all_views = []
    selected_vids = []
    remove_turn_views = [vid]
    subgraph = nx.DiGraph()
    subgraph.add_node(vid, color='b')

    turns_views = []
    for vid2, info in view_ids.items():
        if vid2 != vid and info[0] == view_ids[vid][0]:
            turns_views.append(vid2)
            if turns:
                subgraph.add_node(vid2)
                subgraph.add_edge(vid, vid2, color='black')
                subgraph.add_edge(vid2, vid, color='black')
    if not turns:
        remove_turn_views.extend(turns_views)

    if two_sidded or to_view:
        to_views = [k[0] for k in list(viewgraph.in_edges(vid))]
        for to_id in to_views:
            if to_id not in turns_views:
                subgraph.add_node(to_id)
                subgraph.add_edge(to_id, vid, color='r')
        selected_vids.extend(to_views)
    if two_sidded or not to_view:
        from_views = [k[1] for k in list(viewgraph.out_edges(vid))]
        for from_id in from_views:
            if from_id not in turns_views:
                subgraph.add_node(from_id)
                subgraph.add_edge(vid, from_id, color='g')
        selected_vids.extend(from_views)

    selected_vids = set(selected_vids)
    selected_vids.difference_update(remove_turn_views)
    selected_vids = list(selected_vids)

    for vid2 in selected_vids:
        all_views.append(views[vid2])
    X = []
    Y = []
    U = []
    V = []
    for view in all_views:
        X.append(view[0].x)
        Y.append(view[0].y)
        U.append(view[1].x - view[0].x)
        V.append(view[1].y - view[0].y)
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    ax.quiver([chosen_view[0].x], [chosen_view[0].y], [chosen_view[1].x - chosen_view[0].x],
              [chosen_view[1].y - chosen_view[0].y], angles='xy', scale_units='xy', scale=1,
              color='b')

    plt.plot(space_x, space_y, 'black')
    for i in range(0, len(holes_x)):
        plt.plot(holes_x[i], holes_y[i], 'r')
    for d in door_points:
        plt.plot([d.x()], [d.y()], 'go')
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()

    pos = nx.circular_layout(subgraph)
    ecolors = list(nx.get_edge_attributes(subgraph,'color').values())
    nx.draw(subgraph, pos, edge_color=ecolors, with_labels=True)
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


# overlay regions ...
shapes_list = [shapes[i]['shape'] for i in range(0, len(shapes))]
overlay_regions = list(polygonize(unary_union(list(x.exterior for x in shapes_list))))
gdf = gpd.GeoDataFrame(geometry=overlay_regions)
regions_list = list(gdf['geometry'])
regions_area = [r.area for r in regions_list]
regions_list = [r for (idx, r) in enumerate(regions_list) if regions_area[idx] > 0.0000001]
holes_centroids = []
for idx, holex in enumerate(holes_x):
    holey = holes_y[idx]
    holes_centroids.append(Point(statistics.mean(holex), statistics.mean(holey)))

regions_list_no_holes = []
for r in regions_list:
    is_hole = False
    for cent in holes_centroids:
        if r.contains(cent):
            is_hole = True
            break
    if not is_hole:
        regions_list_no_holes.append(r)
regions_list = regions_list_no_holes
# gdf.to_file("regions.shp")

# shapes_list_gateways = [shapes[i]['shape'] for i in range(0, 5)]
# overlay_regions_gateways = list(polygonize(unary_union(list(x.exterior for x in shapes_list_gateways))))
# gdf_gateways = gpd.GeoDataFrame(geometry=overlay_regions_gateways)
# gdf_gateways.to_file("regions_gateways.shp")

# field of view
# a set of views with same origins to a single view

# changes in visible objects during view
# view -> a set of views

views_ls = {}
for key, value in views.items():
    views_ls[key] = LineString([value[0], value[1]])


# finding all intersecting regions of a view
print('calculating the intersections for #views {0} and #regions: {1}'.format(len(views_ls), len(regions_list)))
intersections = {}
for key, val in views_ls.items():
    intersections[key] = [i for i in range(len(regions_list)) if val.intersects(regions_list[i])]

# calculate regions signatures
print('calculating the visibility signatures...')
signatures = []
for oregion in regions_list:
    center = oregion.centroid
    signatures.append([shapes_list.index(shp) for shp in shapes_list if shp.contains(center)])

# calculating regions information for start and end of views
print('calculating region information (centroid) and views')
views_regions_info = {}
for key, val in views.items():
    start = None
    end = None
    for oregion in regions_list:
        if start is None and oregion.contains(val[0]) or val[0].touches(oregion):
            start = regions_list.index(oregion)
        if end is None and oregion.contains(val[1]) or val[1].touches(oregion):
            end = regions_list.index(oregion)
        if start is not None and end is not None:
            break
    views_regions_info[key] = {'start': start, 'end': end}


# decompose views based on visibility signatures
# decomposed_views = {}
# for view_id, view_region_info in views_regions_info.items():
#     start = view_region_info['start']
#     end = view_region_info['end']
#
#     intersect_regions = set(intersections[view_id])
#     intersect_regions.remove(start)
#     intersect_regions.remove(end)
#     intersect_regions = list(intersect_regions)
#     regions_info = {}
#     for rid in intersect_regions:
#         regions_info[rid] = regions_list[rid].centroid
#
#     decomposed_views[view_id] = {}

# adjacent regions
print('calculating adjacency matrix for regions')
adjacency_matrix = {}
for i in range(0,len(regions_list)-1):
    ri = regions_list[i]
    if i not in adjacency_matrix.keys():
        adjacency_matrix[i] = []
    for j in range(i+1, len(regions_list)):
        rj = regions_list[j]
        if ri.touches(rj):
            if isinstance(ri.intersection(rj), Point):
                continue
            adjacency_matrix[i].append(j)
            if j not in adjacency_matrix.keys():
                adjacency_matrix[j] = [i]
            else:
                adjacency_matrix[j].append(i)

# constructing view graph for decomposed regions
print('finding regions that contains doors/gateways and decision points')
regions_info = {i: regions_list[i].centroid for i in range(len(regions_list))}
regions_doors_info = {}
for pid, door in enumerate(door_points):
    dpoint = Point(door.x(), door.y())
    for rid, r in enumerate(regions_list):
        if r.contains(dpoint) or dpoint.touches(r):
            regions_doors_info[rid] = pid

rview_ids = {}
rviews = {}
rview_ls = {}
counter = 0
for idx, signature in enumerate(signatures):
    # based on signature --> direct access (reach)
    block = None
    neighbours = adjacency_matrix[idx]

    for rid, pid in regions_doors_info.items():
        if rid != idx and pid in signature and rid not in neighbours:
            rview_ids[counter] = [idx, rid]
            view_line = LineString([regions_info[idx], regions_info[rid]])
            if view_intersects_holes(view_line):
                rviews[counter] = [regions_info[idx], Point(door_points[pid].x(), door_points[pid].y())]
                rview_ls[counter] = LineString([regions_info[idx], Point(door_points[pid].x(), door_points[pid].y())])
            else:
                rviews[counter] = [regions_info[idx], regions_info[rid]]
                rview_ls[counter] = view_line
            counter += 1

    # based on adjacent regions --> access to new information toward a visible object (orient)

    for neighbour in neighbours:
        rview_ids[counter] = [idx, neighbour]
        view_line = LineString([regions_info[idx], regions_info[neighbour]])
        if view_intersects_holes(view_line):
            pol_ext = LinearRing(regions_list[neighbour].exterior.coords)
            d = pol_ext.project(regions_info[idx])
            neighbour_point = pol_ext.interpolate(d)
            rviews[counter] = [regions_info[idx], neighbour_point]
            rview_ls[counter] = LineString([regions_info[idx], neighbour_point])
        else:
            rviews[counter] = [regions_info[idx], regions_info[neighbour]]
            rview_ls[counter] = view_line
        counter+=1

from_region_ids = {}
to_region_ids = {}
for rvid, vals in rview_ids.items():
    if vals[0] not in from_region_ids.keys():
        from_region_ids[vals[0]] = rvid
    if vals[1] not in to_region_ids.keys():
        to_region_ids[vals[1]] = rvid
    if len(to_region_ids) == len(signatures) and len(from_region_ids) == len(signatures):
        break

# constructing region view graph
print('constructing view graph for regions')
rviewgraph = nx.DiGraph()
rviewgraph.add_nodes_from(list(rview_ids.keys()))
for vid, pids in rview_ids.items():
    v1 = rviews[vid]
    dv1 = calculate_distance(v1[0], v1[1])
    for vid2, pids2 in rview_ids.items():
        if vid != vid2:
            if pids[1] == pids2[0]: # movement
                rviewgraph.add_edge(vid, vid2, weight=dv1)
            elif pids[0] == pids2[0]: # turn
                rviewgraph.add_edge(vid, vid2, weight=0)

def shortest_path_regions(rid1, rid2):
    regions_set = {rid1, rid2}

    vid1 = from_region_ids[rid1]
    vid2 = from_region_ids[rid2]

    vpath = nx.shortest_path(rviewgraph, vid1, vid2, weight='weight')
    path_view = []

    vpath = vpath[:-1]
    if len(vpath) >= 2 and rview_ids[vpath[0]][0] == rview_ids[vpath[1]][0]:
        vpath = vpath[1:]
    print(vpath)
    for vid in vpath:
        path_view.append(rviews[vid])
        regions_set = regions_set.union(rview_ids[vid])

    X = []
    Y = []
    U = []
    V = []
    for view in path_view:
        X.append(view[0].x)
        Y.append(view[0].y)
        U.append(view[1].x - view[0].x)
        V.append(view[1].y - view[0].y)
    fig, ax = plt.subplots()
    plt.plot(space_x, space_y, 'black')
    for i in range(0, len(holes_x)):
        plt.plot(holes_x[i], holes_y[i], 'r')

    rcentroid1 = regions_info[rid1]
    plt.plot([rcentroid1.x], [rcentroid1.y], 'go')
    rcentroid2 = regions_info[rid2]
    plt.plot([rcentroid2.x], [rcentroid2.y], 'go')

    for region_id in regions_set:
        region = regions_list[region_id]
        r_x, r_y = save_print_geojson(list(region.exterior.coords))
        plt.plot(r_x, r_y, 'b')
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


def plot_region(rid):
    region = regions_list[rid]
    r_x, r_y = save_print_geojson(list(region.exterior.coords))
    fig, ax = plt.subplots()
    plt.plot(space_x, space_y, 'black')
    for i in range(0, len(holes_x)):
        plt.plot(holes_x[i], holes_y[i], 'r')
    plt.plot(r_x, r_y, 'b')
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


def plot_shp(shp, point=False):
    d_x = None
    d_y = None
    if isinstance(shp, int):
        d_x = door_points[shp].x()
        d_y = door_points[shp].y()
        shp = shapes_list[shp]

    r_x, r_y = save_print_geojson(list(shp.exterior.coords))
    fig, ax = plt.subplots()
    plt.plot(space_x, space_y, 'black')
    plt.plot([d_x], [d_y], 'go')
    if point:
        for p in door_points:
            plt.plot([p.x()], [p.y()], 'go')
    for i in range(0, len(holes_x)):
        plt.plot(holes_x[i], holes_y[i], 'r')
    plt.plot(r_x, r_y, 'b')
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


def demo(object_based=test, region_based=test_regions):
    if object_based:
        input("Start testing the object-object views:\nPress Enter to continue...")
        print('plot isovist for node: 11')
        plot_shp(11)
        input("Press Enter to continue...")
        print('plot isovist for node: 58')
        plot_shp(58)
        input("Press Enter to continue...")
        print('plot shortest path from 15 to 58')
        shortest_path(11, 58)
        input("Press Enter to continue...")

        # print('plot isovist for node 48')
        # plot_isovist(48)
        # input("Press Enter to continue...")
        # print('plot shortest path for 11 to 48')
        # shortest_path(11, 48)
        # input("Press Enter to continue...")

        print('\nplot view graph info...')
        print('plot view graph: view 325: without turns, only from view 325')
        plot_view_sequence(325)
        input("Press Enter to continue...")
        print('plot view graph: view 325: without turns, only to view 325')
        plot_view_sequence(325, to_view=True)
        input("Press Enter to continue...")
        print('plot view graph: view 325: without turn, both from and to view 325')
        plot_view_sequence(325, two_sidded=True)
        input("Press Enter to continue...")
        print('plot view graph: view 325: with turn, both from and to view 325')
        plot_view_sequence(325, two_sidded=True, turns=True)
        input("Press Enter to continue...")

    if region_based:
        input("Start testing the region graph\nPress Enter to continue...")
        print('test shortest path between region 2163 to region 946')
        plot_region(2163)
        plot_region(946)
        shortest_path_regions(2163, 946)
        input("Press Enter to continue...")

        print('similar case with view graph (door to door)')
        plot_shp(42)
        plot_shp(69)
        shortest_path(42, 69)

        # # 2625 3944
        input("\n\nPress Enter to continue...")
        print('test shortest path between region 2625 to region 3944')
        plot_region(2625)
        plot_region(3944)
        shortest_path_regions(2625, 3944)
        #
        print('similar case with view graph (door to door)')
        plot_shp(14)
        plot_shp(48)
        shortest_path(14, 48)

        input("\n\nPress Enter to continue...")
        print('test shortest path for a regions away from decision points')
        plot_region(870)
        plot_region(3944)
        shortest_path_regions(870, 3944)


def vision_triangle(view_id, fov=120):
    v = rviews[view_id]
    p1 = v[0]
    p2 = calculate_coordinates(v=v, angle=fov/2, d=50)
    p3 = calculate_coordinates(v=v, angle=-fov/2, d=50)
    return Poly([[p.x, p.y] for p in [p1, p2, p3]])

def isovist_calc(x, y):
    door = vis.Point(x,y)
    door.snap_to_boundary_of(env, epsilon)
    door.snap_to_vertices_of(env, epsilon)
    isovist = vis.Visibility_Polygon(door, env, epsilon)
    iso_x, iso_y = save_print(isovist)
    geojson_polygon = to_polygon_geojson(iso_x, iso_y)
    shp = to_polygon_shape(geojson_polygon, clip=True)
    if not shp.is_valid:
        print('invalid')
    if isinstance(shp, MultiPolygon):
        test_case = shp
        shps = list(shp)
        areas = [s.area for s in shps]
        max_area = max(areas)
        idxs = areas.index(max_area)
        shp = shps[idxs]
        test_idx = idx
    return shp

def view_vision(view_idx, fov=120, is_start=True, isovist_view=None):
    triangle = vision_triangle(view_idx, fov)
    view_line = rview_ls[view_idx]
    if isovist_view is None:
        if is_start:
            x = view_line.xy[0][0]
            y = view_line.xy[1][0]
        else:
            x = view_line.xy[0][1]
            y = view_line.xy[1][1]
        isovist_view = isovist_calc(x, y)
    return isovist_view.intersection(triangle)

def view_vision_signature(view_coverage, door_points=door_points):
    signature = []
    for idx, p in enumerate(door_points):
        if view_coverage.contains(Point(p.x(), p.y())) or view_coverage.touches(Point(p.x(), p.y())):
            signature.append(idx)
    return signature

def decompose_view(view_idx):
    view_line = rview_ls[view_idx]
    decomposed_views = []
    start_coverage = view_vision(view_idx)
    start_signature = view_vision_signature(start_coverage)
    end_coverage = view_vision(view_line, is_start=False)
    end_signature = view_vision_signature(end_coverage)
    if start_coverage != end_coverage:
        pass   # todo

    return decomposed_views

def demo_vision(vid):
    vv = view_vision(vid)
    ids = rview_ids[vid]
    shortest_path_regions(ids[0], ids[1])
    centroid = regions_info[ids[0]]
    iso = isovist_calc(centroid.x, centroid.y)
    plot_shp(iso)
    plot_shp(vv, point=True)
    print('signature: {}'.format(view_vision_signature(vv)))


# problems:
    # zig-zag views when we consider region-to-region views
        # impact the shortest paths -> going back
    # decompose long views (that intersect multiple visibility regions)
    # reduce the number of nodes (and edges)
        # views -- unique visible information
        # field of view: (should be applied to significantly reduce the number of nodes)
        # algorithmic design for graph pruning
    # pruning based on the triangles ...

# :
    # meaningless regions -- agent's space
    # meaningless region connections --> moving towards a less important region?
    # adding landmarks for capturing spatial information about the environment:
        # egocentric
        # alocentric
        # cardinal
        # order
    # compare views based on their information

def calculate_spatial_relationships(vid):
    vv = view_vision(vid)
    door_signature = view_vision_signature(vv)
    print(door_signature)
    landmark_signature = view_vision_signature(vv, door_points=landmarks_points)
    print(landmark_signature)