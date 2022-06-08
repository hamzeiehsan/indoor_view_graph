class Parameters:
    basic = False
    hypo = False
    if basic or hypo:
        # test example
        epsilon = 0.00001
        door_weight = 0.5
        turn_weight = 0.0001
        precision = 5
        alpha = 40
        fov = 120
        max_distance = 1000
        min_area = 0.0000005
        max_collect_geom = 0.002
    else:
        # real world example
        epsilon = 0.01
        door_weight = 50
        turn_weight = 0.05
        precision = 2
        alpha = 40
        fov = 160
        max_distance = 1000000
        min_area = 0.005
        max_collect_geom = 25
