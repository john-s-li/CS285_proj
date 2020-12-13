
TOTAL_CYCLE_TIME = 0.3
RATIO = 0.7 # denotes 30% stance and 70% half swing
VON_MISES_KAPPA = 5000 # concentrate heavily around mean


PHASE_VALS = {

    'trot' : {
        'front_left' : 0.0,
        'front_right' : 0.5,
        'rear_left' : 0.5,
        'rear_right' : 0.0
        },

    # can also try bounding 
    # 2 front legs = 1 leg and same for rear
     'gallop' : {
        'front_left' : 0.0,
        'front_right' : 0.0,
        'rear_left' : 0.5,
        'rear_right' : 0.5
        }

}