import numpy as np
import time as t
import matplotlib.pyplot as plt
from tqdm import tqdm

def asymptote_poi(params, l_x, l_y, l_z):
    p, ecc, theta0, phi0, sign = params
    a = np.tan(np.pi/2-theta0-np.arccos(-1*sign/ecc))
    r_mid = sign*p*ecc/(1-ecc**2) 
    z_mid = r_mid*np.cos(theta0-np.pi)
    y_mid = r_mid*np.sin(theta0-np.pi)*np.sin(phi0)
    x_mid = r_mid*np.sin(theta0-np.pi)*np.cos(phi0)
    b = z_mid - sign*a*np.sqrt(x_mid**2+y_mid**2)
    theta = np.pi/2-np.arctan(a/(1-b/l_z))
    r = b/(np.cos(theta)-a*np.sin(theta))
    x = r*np.sin(theta)*np.cos(phi0)
    y = r*np.sin(theta)*np.sin(phi0)
    z = r*np.cos(theta)
    u = np.sqrt(l_x**2+l_y**2)*theta_E
    # if u < 0.0001:
    #     u = 0.0001
    mag = 0.5*((u**2+2)/(u*np.sqrt(u**2+4))+sign)
    return x, y, z, mag

def rays_from_point_params(r_s, l_x, l_y, l_z, l_o):
    l_s = np.sqrt(l_x**2+l_y**2+l_z**2)
    l_h = np.sqrt(l_x**2+l_y**2)
    theta_s = np.arctan(l_h/l_z)
    theta_o = np.pi
    # theta_o = np.pi-np.arcsin(l_h/l_o)
    theta_so = np.pi+theta_s-theta_o
    if l_x > 0:
        phi0 = np.arctan(l_y/l_x)
    else:
        phi0 = np.pi + np.arctan(l_y/l_x)
    a = l_s**2 + 2*l_s*l_o*np.cos(theta_so) + l_o**2
    b = l_s*l_o*(l_s*l_o*np.cos(theta_so)**2 - l_s*l_o - 2*r_s*(np.cos(theta_so) + 1)*(l_s + l_o))
    c = (l_s*l_o*r_s)**2*(np.cos(theta_so)**2+2*np.cos(theta_so)+1)
    b1 = np.sqrt((-b+np.sqrt(b**2-4*a*c))/(2*a))
    b2 = np.sqrt((-b-np.sqrt(b**2-4*a*c))/(2*a))
    theta1 = theta_o-np.pi+np.arccos((1-(b1**2)/(l_o*r_s))/np.sqrt(1+(b1**2)/(r_s**2)))
    theta2 = theta_o-np.pi+np.arccos(((b2**2)/(l_o*r_s)-1)/np.sqrt(1+(b2**2)/(r_s**2)))
    p1 = b1**2/r_s
    p2 = b2**2/r_s
    ecc1 = np.sqrt(1+(b1/r_s)**2)
    ecc2 = np.sqrt(1+(b2/r_s)**2)
    return [(p1, ecc1, theta1, phi0, 1), (p2, ecc2, theta2, phi0, -1)]

def cluster_source(x0, y0, z0, radius, r_space, theta_space, phi_space):
    points = []
    rs = np.arange(r_space, radius, r_space)
    thetas = np.arange(theta_space, np.pi, theta_space)
    phis = np.arange(phi_space/2, 2*np.pi, phi_space)
    for r in rs:
        for theta in thetas:
            for phi in phis:
                x = x0 + r*np.cos(phi)*np.sin(theta)
                y = y0 + r*np.sin(phi)*np.sin(theta)
                z = z0 + r*np.cos(theta)
                point = (x, y, z)
                points.append(point)
    return points

start_time = t.time()

to_centre = False
res = 100
l_x_max = 0.025

r_s = 2
l_z = 500
l_y = 0
l_o = 1000
source_size = 1

theta_E = np.sqrt(2*r_s*l_z/(l_o*(l_o+l_z)))
r_E = theta_E*(l_o+l_z)

if to_centre:
    l_xs = np.linspace(0, l_x_max, res)
else:
    l_xs = np.linspace(-l_x_max, l_x_max, res)

mags = []
for i in tqdm(range(res)):
    l_x = l_xs[i]
    source = cluster_source(l_x, l_y, l_z, source_size, source_size/3, np.pi/12, np.pi/12)
    mag1 = 0
    mag2 = 0
    for point in source:
        x, y, z = point
        params1, params2 = rays_from_point_params(r_s, x, y, z, l_o)
        mag1 += asymptote_poi(params1, l_x, l_y, l_z)[3]
        mag2 += asymptote_poi(params2, l_x, l_y, l_z)[3]
        # mag1 += poi1[3]
        # mag2 += poi2[3]
    mags.append(mag1+mag2)

print("Runtime:", t.time()-start_time)

_, ax = plt.subplots()

ax.scatter(l_xs, mags)

plt.show()