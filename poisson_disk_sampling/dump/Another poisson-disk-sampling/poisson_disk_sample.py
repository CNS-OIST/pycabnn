#
# This is an implementation of Fast Poisson Disk Sampling in Arbitrary Dimensions:
# https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/sketches/0250.pdf
# Pygame is used to visualize the generated samples.
#
import pygame
import random
import math
import sys

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
		k = 30
		for it in range(k):
			pn = generate_random_point(random_p, minimum_dist)
			
			fits = is_in_circle(pn)
			if fits:
				# TODO: Optimize.  Maintain a grid of existing samples, and only check viable nearest neighbors.
				for point in samples:
					if distance(point, pn) < minimum_dist:
						fits = False
						break
					
			if fits:
				samples.append(pn)
				active_list.append(pn)
				found = True
				break
		
		if not found:
			active_list.remove(random_p)
	
	# Print the samples in a form that can be copy-pasted into other code.
	print("There are %d samples:" % len(samples))
	for point in samples:
		print("\t{\t%08f,\t%08f\t}," % (point[0], point[1]))

pygame.init()

screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()

random.seed()
min_dist = float(sys.argv[1])
if len(sys.argv)<2:
    print ("Fatal: You forgot to include the directory name on the command line.")
    print ("Usage:  python %s <directoryname>" % sys.argv[0])
    sys.exit(1)
generate_points(min_dist)

while True:
	break_loop = False
	clock.tick(60)
	
	for event in pygame.event.get():
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				break_loop = True
			if event.key == pygame.K_SPACE:
				generate_points(min_dist)
	
	screen.fill((0,0,0))
	
	pygame.draw.circle(screen, (50, 50, 200), (250, 250), 250, 1)
	
	for point in samples:
		lx = 250 + int(point[0] * 250)
		ly = 250 + int(point[1] * 250)
		pygame.draw.circle(screen, (255,255,255), (lx, ly), 2)
		pygame.draw.circle(screen, (25, 40, 25), (lx, ly), int(250 * min_dist), 1)
	
	pygame.display.flip()
	
	if break_loop:
		break