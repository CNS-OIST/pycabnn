#
# This is an implementation of Fast Poisson Disk Sampling in Arbitrary Dimensions:
# https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/sketches/0250.pdf
# Pygame is used to visualize the generated samples.
#
import pygame
import random
import math
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

samples = []
active_list = []


def distance(p1, p2):
    p_diff = (p2[0] - p1[0], p2[1] - p1[1])
    return math.sqrt(math.pow(p_diff[0], 2) + math.pow(p_diff[1], 2))


def is_in_circle(p):
    d = distance(p, (0, 0))
    return d < 1.0


def generate_random_point(o, r):
    # Generate random point around o between r and 2r away.
    r0 = random.random()
    r1 = random.random()

    dist = r + (r0 * (2 * r - r))
    angle = 2 * math.pi * r1

    return (o[0] + dist * math.cos(angle), o[1] + dist * math.sin(angle))


def generate_points(minimum_dist):
    del samples[:]

    # Choose a point randomly in the domain.
    initial_point = (0, 0)
    while True:
        ix = random.random()
        iy = random.random()

        initial_point = (ix, iy)

        if is_in_circle(initial_point):
            samples.append(initial_point)
            active_list.append(initial_point)
            break

    while len(active_list) > 0:
        # Choose a random point from the active list.
        p_index = random.randint(0, len(active_list) - 1)
        random_p = active_list[p_index]

        found = False

        # Generate up to k points chosen uniformly at random from between r and 2r away from p.
        k = 2
        for i in range(k):
            pn = generate_random_point(random_p, minimum_dist)
            fits = is_in_circle(pn)
            if fits:
                # TODO: Optimize.  Maintain a grid of existing samples, and only check viable nearest neighbors.
                for point in samples:
                    if distance(point, pn) < minimum_dist:
                        fits = False
                        break

            samples.append(pn)
            active_list.append(pn)
            found = True
            break

        else:
            active_list.remove(random_p)

    # Print the samples in a form that can be copy-pasted into other code.
    print("There are %d samples:" % len(samples))
    for point in samples:
        print("\t{\t%08f,\t%08f\t}," % (point[0], point[1]))

if __name__ == "__main__":
    generate_points(1)
