import numpy as np
import time as t
import os
import matplotlib.pyplot as plt

def ray(b, r_s, theta0, phi0, sign):
    theta = np.linspace(-np.pi, np.pi, ray_points)
    r = b**2/(r_s*(1+sign*np.sqrt(1+(b/r_s)**2)*np.cos(theta-theta0)))
    x = r*np.cos(phi0)*np.sin(theta)
    y = r*np.sin(phi0)*np.sin(theta)
    z = r*np.cos(theta)
    return np.array([x, y, z])

def rays_from_point(r_s, l_x, l_y, l_z, l_o):
    l_s = np.sqrt(l_x**2+l_y**2+l_z**2)
    l_h = np.sqrt(l_x**2+l_y**2)
    theta_s = np.arctan(l_h/l_z)
    theta_o = np.pi-np.arcsin(l_h/l_o)
    theta_so = np.pi+theta_s-theta_o
    phi0 = np.arctan(l_y/l_x)
    a = l_s**2 + 2*l_s*l_o*np.cos(theta_so) + l_o**2
    b = l_s*l_o*(l_s*l_o*np.cos(theta_so)**2 - l_s*l_o - 2*r_s*(np.cos(theta_so) + 1)*(l_s + l_o))
    c = (l_s*l_o*r_s)**2*(np.cos(theta_so)**2+2*np.cos(theta_so)+1)
    b1 = np.sqrt((-b+np.sqrt(b**2-4*a*c))/(2*a))
    b2 = np.sqrt((-b-np.sqrt(b**2-4*a*c))/(2*a))
    theta1 = theta_o-np.pi+np.arccos((1-(b1**2)/(l_o*r_s))/np.sqrt(1+(b1**2)/(r_s**2)))
    theta2 = theta_o-np.pi+np.arccos(((b2**2)/(l_o*r_s)-1)/np.sqrt(1+(b2**2)/(r_s**2)))
    ray1 = ray(b1, r_s, theta1, phi0, 1)
    ray2 = ray(b2, r_s, theta2, phi0, -1)
    return (ray1, ray2)

def cluster_source(x0, z0, radius, r_space, theta_space, phi_space):
    points = []
    rs = np.arange(r_space, radius, r_space)
    thetas = np.arange(theta_space, np.pi, theta_space)
    phis = np.arange(phi_space/2, 2*np.pi, phi_space)
    for r in rs:
        for theta in thetas:
            for phi in phis:
                x = x0 + r*np.cos(phi)*np.sin(theta)
                y = r*np.sin(phi)*np.sin(theta)
                z = z0 + r*np.cos(theta)
                point = (x, y, z)
                points.append(point)
    return points

fig_size = 12
ray_width = 0.1
ray_points = 100

r_s = 2
l_o = 10
l_x = 5
l_y = 0
l_z = 5
source_size = 0.2

theta_E = np.sqrt(2*r_s*l_z/(l_o*(l_o+l_z)))
r_E = theta_E*l_o

source = cluster_source(l_x, l_z, source_size, source_size/3, np.pi/4, np.pi/12)
# source = [[l_x, l_y, l_z]]

rays = np.empty(shape=(2*len(source), 3, ray_points))
for i in range(0, len(source), 2):
    x, y, z = source[i]
    ray1, ray2 = rays_from_point(r_s, x, y, z, l_o)
    rays[i] = np.array(ray1)
    rays[i+1] = np.array(ray2)

im_x = []
im_y = []
for ray in rays:
    ray_dir = ray.T[1] - ray.T[0]
    x = ray_dir[0]/ray_dir[2]*(l_z+l_o)
    y = ray_dir[1]/ray_dir[2]*(l_z+l_o)
    im_x.append(x)
    im_y.append(y)

fig1 = plt.figure(dpi=100)
ax1 = fig1.add_subplot(projection="3d")
for ray in rays:
    x, y, z = ray
    ax1.scatter(x, y, z, s=ray_width)
ax1.scatter([l_x, 0, l_x], [l_y, 0, l_y], [-l_o, 0, l_z], s=20)
# for point in source:
#     ax.scatter(point[0], point[1], point[2])

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_xlim(-fig_size, fig_size)
ax1.set_ylim(-fig_size, fig_size)
ax1.set_zlim(-fig_size, fig_size)
plt.show()

# fig2, ax2 = plt.subplots()
# ax2.scatter(im_x, im_y, alpha=0.5)
# ax2.scatter(0, 0)
# # circle = plt.Circle((0,0), r_E, color="r", fill=False)
# # ax2.add_patch(circle)
# ax2.set_aspect("equal")
# ax2.set_ylabel("Y")
# ax2.set_xlabel("X")
# ax2.set_xlim(-2*r_E, 2*r_E)
# ax2.set_ylim(-2*r_E, 2*r_E)
# plt.show()