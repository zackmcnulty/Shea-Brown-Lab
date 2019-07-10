'''
dynamical_systems.py

A collection of common, simple dynamical systems that can be used to generate 
trajectories for objects throughout the space

'''
import numpy as np

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


def f_both_spring(initial_condition = [0,0,1], mass = 1, k1 = 1, k2=1, center=(0,0)): 
    ''' spring-mass system oscillating vertically AND horizontally centered at x,y = 0'''

    if len(initial_condition) != 4:
        raise ValueError('The 2D spring mass system expects four intial conditions: x,y, and velocity in x, velocity in y')

    # x[0] = x position, x[1] = y position, x[2] = velocity in x, x[3] = velocity in y
    dxdt = lambda x: x[2]
    dydt = lambda x: x[3] 
    dvx_dt = lambda x: -k1/mass * x[0]
    dvy_dt = lambda x: -k2/mass * x[1]

    return [lambda t, x: [dxdt(x), dydt(x), dvx_dt(x), dvy_dt(x)], initial_condition]

# dynamical system for a spring oscillating along an axis at angle theta centered at the given initial condition
def f_angled_spring(initial_condition = [0,0,1], theta=0, mass = 1, k = 1, center=(0,0)): 
    ''' spring-mass system oscillating in a line theta radians above horizontal centered at x,y = center'''

    if len(initial_condition) != 3:
        raise ValueError('The angled spring mass system expects three intial conditions: x,y, and velocity')

    # x[0] = x position, x[1] = y position, x[2] = velocity
    dxdt = lambda x:  x[2] * np.cos(theta)
    dydt = lambda x:  x[2] * np.sin(theta)
    dv_dt = lambda x: -k/mass * np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2) * (2*(np.sign(np.sin(theta)) == np.sign(x[1])) - 1)

    return [lambda t, x: [dxdt(x), dydt(x), dv_dt(x)], initial_condition]

# stationary object
def f_stationary(initial_condition = [0,0]):

    return [lambda t, x: [0, 0], initial_condition]


# Spring oscillating between two points
def f_two_point_spring(p1=(0,0), p2=(1,1), v_init=1, k=1):
	center_x = (p1[0] + p2[0]) / 2
	center_y = (p1[1] + p2[1]) / 2

	initial_condition = [center_x, center_y, v_init]
	max_displacement_squared = (p1[0] - center_x)**2 + (p1[0] - center_y)**2
	m = k * max_displacement_squared / (v_init ** 2)

	# starts heading towards p2
	theta = np.arctan2(p2[1] - center_y, p2[0] - center_x)
	
	return f_angled_spring(initial_condition=initial_condition, theta=theta, mass=m, k=k, center=(center_x, center_y))
	


