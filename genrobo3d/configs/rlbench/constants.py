
def get_robot_workspace(real_robot=False, use_vlm=False):
    if real_robot:  
        # ur5 robotics room
        if use_vlm:
            TABLE_HEIGHT = 0.0 # meters
            X_BBOX = (-0.60, 0.2)        # 0 is the robot base
            Y_BBOX = (-0.54, 0.54)  # 0 is the robot base
            Z_BBOX = (-0.02, 0.75)      # 0 is the table
        else:
            TABLE_HEIGHT = 0.01 # meters
            X_BBOX = (-0.60, 0.2)        # 0 is the robot base
            Y_BBOX = (-0.54, 0.54)  # 0 is the robot base
            Z_BBOX = (0, 0.75)      # 0 is the table
        
    else:
        # rlbench workspace
        TABLE_HEIGHT = 0.7505 # meters

        X_BBOX = (-0.5, 1.5)    # 0 is the robot base
        Y_BBOX = (-1, 1)        # 0 is the robot base 
        Z_BBOX = (0.2, 2)       # 0 is the floor

    return {
        'TABLE_HEIGHT': TABLE_HEIGHT, 
        'X_BBOX': X_BBOX, 
        'Y_BBOX': Y_BBOX, 
        'Z_BBOX': Z_BBOX
    }



def get_rlbench_labels(task, table=True, robot=True, wall=True, floor=True):
    # OLD with polarnet bug
    # # 204,208: table; 199-203: wall; 246: floor; 257?
    # # 212-216, 221, 222, 225: robotic arm
    # LABELS = []
    # if table:
    #     LABELS += [204, 208]
    # if robot:
    #     LABELS += [210, 211, 212, 213, 214, 215, 216, 221, 222, 225]
    # if wall:
    #     LABELS += [199, 200, 201, 202, 203]
    # if floor:
    #     LABELS += [246, 257]
    LABELS = []
    if table:
        LABELS += [48, 51, 52]
        if task == "close_jar_peract":
            LABELS += [86]
        elif task == "close_jar":
            LABELS += [86]
        elif task == "light_bulb_in_peract":
            LABELS += [98]
        elif task == "change_channel":
            LABELS += [102]
        elif task == "empty_container":
            LABELS += [86]
        elif task == "light_bulb_in":
            LABELS += [97]
        elif task == "light_bulb_out":
            LABELS += [95]
        elif task == "open_jar":
            LABELS += [89]
        elif task == "tv_on":
            LABELS += [102]
        elif task == 'close_fridge':
            LABELS += [81]
    if floor:
        LABELS += [8, 9, 10, 70, 71]
    if robot:
        LABELS += list(range(12, 48)) + [67, 68, 69]
    if wall:
        LABELS += [53, 54, 55, 56, 57]

    undefined = 65535
    LABELS += [undefined]

    return LABELS
