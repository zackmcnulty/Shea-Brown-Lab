'''
dynamical_systems.py

A collection of common, simple dynamical systems that can be used to generate 
trajectories for objects throughout the space

'''

# NOTE: First two parameters should be x and y coordinates

def f_horz_spring(initial_condition = [0,0,1], mass = 1, k = 1): 
    ''' spring-mass system oscillating horizontally centered at x = 0'''

    # x[0] = x position, x[1] = y position, x[2] = velocity
    if len(initial_condition) != 3:
        raise ValueError('The spring mass system expects three intial conditions: x,y, and velocity (in x)')

    dxdt = lambda x: x[2]
    dydt = lambda x: 0
    dvdt = lambda x: -k/mass * x[0]

    return [lambda t, x: [dxdt(x), dydt(x), dvdt(x)], initial_condition]


def f_vert_spring(initial_condition = [0,0,1], mass = 1, k = 1): 
    ''' spring-mass system oscillating vertically centered at y = 0'''

    if len(initial_condition) != 3:
        raise ValueError('The spring mass system expects three intial conditions: x,y, and velocity (in y)')

    # x[0] = x position, x[1] = y position, x[2] = velocity
    dxdt = lambda x: 0
    dydt = lambda x: x[2] 
    dvdt = lambda x: -k/mass * x[1]

    return [lambda t, x: [dxdt(x), dydt(x), dvdt(x)], initial_condition]
