
import numpy as np
import os
from math import pi, sin, cos
npoints = 30

R=5

def gen_circle_xy(coor_origin, radius, start, end):
    phi_values = np.arange(start,end, (end-start)/npoints)
    phi_values=phi_values[::-1]
    circle_coor = np.empty((0,3), float)
    # print(phi_values)
    for phi in phi_values:
        circle_coor = np.append(circle_coor, np.array([[coor_origin[0] + radius*cos(phi),coor_origin[1]+ radius*sin(phi), coor_origin[2]+ 0]]), axis=0)
    return circle_coor


def gen_circle_yz(coor_origin, radius, start, end):
    phi_values = np.arange(start,end, (end-start)/npoints)
    phi_values=phi_values[::-1]
    circle_coor = np.empty((0,3), float)
    # print(phi_values)
    for phi in phi_values:
        circle_coor = np.append(circle_coor, np.array([[coor_origin[0], coor_origin[1] + radius*cos(phi),coor_origin[2]+ radius*sin(phi)]]), axis=0)
    return circle_coor

def gen_line_z(start, end):
    line_values = np.arange(start, end, (end-start)/npoints)
    line_coor = np.empty((0,3), float)
    for line in line_values:
        line_coor= np.append(line_coor, np.array([[0,0, line]]), axis=0)

    return line_coor
    

# generate closed circle 1
circle1 = gen_circle_xy([0,0,0], 13, -pi/2, 3/2*pi)
np.savetxt('circle1', circle1, fmt='%.3f', delimiter='\t')

# generate circle 2, can be open
circle2 = gen_circle_yz([2,25,0], 13, 0.9*pi, 1.1*pi)
np.savetxt('circle2', circle2, fmt='%.3f', delimiter='\t')

# merge two circle
os.system('cat circle1 circle2 > system')


line1 = gen_line_z(-5, 1)
np.savetxt('line1', line1, fmt='%3f', delimiter='\t')
# merge circle1 and line
# os.system('cat circle1 line1 > system')