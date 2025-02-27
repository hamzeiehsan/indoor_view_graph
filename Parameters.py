class Parameters:
    basic = False
    hypo = False
    mc = True
    if basic or hypo:
        # test example
        epsilon = 0.00001
        door_weight = 0.5
        turn_weight = 0.0001
        precision = 5
        alpha = 40
        fov = 160
        max_distance = 1000
        min_area = 0.0000005
        max_collect_geom = 0.002
    elif mc:
        # real world example
        epsilon = 0.01
        door_weight = 50
        turn_weight = 0.05
        precision = 2
        alpha = 40
        fov = 160
        max_distance = 1000000
        min_area = 0.005
        max_collect_geom = 25  # default was 25 in the first round
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
        max_collect_geom = 100

    @staticmethod
    def hypo_basic_parameters():
        Parameters.epsilon = 0.00001
        Parameters.door_weight = 0.5
        Parameters.turn_weight = 0.0001
        Parameters.precision = 5
        Parameters.alpha = 40
        Parameters.fov = 160
        Parameters.max_distance = 1000
        Parameters.min_area = 0.0000005
        Parameters.max_collect_geom = 0.002

    @staticmethod
    def real_world_parameters(mc=True):
        Parameters.epsilon = 0.01
        Parameters.door_weight = 50
        Parameters.turn_weight = 0.05
        Parameters.precision = 2
        Parameters.alpha = 40
        Parameters.fov = 160
        Parameters.max_distance = 1000000
        Parameters.min_area = 0.005
        if mc:
            Parameters.max_collect_geom = 25
        else:
            Parameters.max_collect_geom = 100

    @staticmethod
    def set_env(env="real", mc=True):
        if env == "real":
            Parameters.hypo = False
            Parameters.basic = False
            Parameters.real_world_parameters(mc=mc)
        elif env == "hypo":
            Parameters.hypo = True
            Parameters.basic = False
            Parameters.hypo_basic_parameters()
        else:
            Parameters.basic = True
            Parameters.hypo = False
            Parameters.hypo_basic_parameters()

    @staticmethod
    def print_info():
        print('--------------------------------------')
        if Parameters.basic:
            print("Basic environment is active\n")
        elif Parameters.hypo:
            print("Hypothetical environment is active\n")
        else:
            print("Real-world environment is active\n")
        print("Static Variables: ")
        variables = {'epsilon': Parameters.epsilon, 'precision': Parameters.precision,
                     'alpha': Parameters.alpha, 'fov': Parameters.fov,
                     'min_area': Parameters.min_area, 'max_distance': Parameters.max_distance,
                     'door_weight': Parameters.door_weight, 'turn_weight': Parameters.turn_weight}
        for key, value in variables.items():
            print("\t{0}: {1}".format(key, value))
        print('--------------------------------------')
