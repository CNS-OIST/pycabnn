import numpy as np
from scipy import spatial
import time

def distance_test(x, p):
    return np.sqrt((x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2 + (x[2] - p[2]) ** 2)

points = np.loadtxt('noise_glo.txt')  # Glomeruli coordinates
grc_points = np.loadtxt('noise_grc.txt') # Granular cells coordinates

Transverse_range = 1500  # Y range
Horizontal_range = 700  # X range
Vertical_range = 200  # Z range
a_x = 24.55
a_y = 6.15
a_z = 15

MF = np.loadtxt('readMF_final.txt') # Mossy Fiber coordinates
points = np.loadtxt('noise_glo.txt')
grc_points = np.loadtxt('noise_grc.txt')
total = np.vstack((points, grc_points))
total_MF = np.vstack((MF, points))
semi_y = 64/2 #1500 / 2  # 64 / 2
semi_x = 84/2 #700 / 2  # 84 / 2
semi_z = 50/2 #50/2 #50 / 2  # 50 왜 50으로 잡았는지 알아야 함
# for i in range(0, np.size(points[:, 0])):
#     add_point_grc(points[i, :])


# MF_GL kdtree_query_ball(x, total)
prob = 1
r_MF = semi_x
MF_GL = []
temp_MF = []
ball = spatial.cKDTree(total_MF).query_ball_point(MF, r_MF)
for results in ball:  # Setting temporal coordinates which points
    temp_MF.append(total_MF[results])
for i in range(0, len(MF)):  # Go to each central point
    new = []
    distance= []
    num = len(temp_MF[i])  # Setting how many points around one center point(i)
    count = 0
    while num >= 0:
        if MF[i, 1] - semi_y <= temp_MF[i][num - 1, 1] <= MF[i, 1] + semi_y and MF[i, 2] - semi_z <= temp_MF[i][num - 1, 2] <= MF[i, 2] + semi_z:
            new.append(temp_MF[i][num - 1])
            distance.append(np.sqrt((MF[i, 0] - new[count][0])**2 + (MF[i, 1] - new[count][1])**2 + (MF[i, 2] - new[count][2])**2)) # Measuring the distance
            if distance_test(new[count], MF[i]) == 0:
                del (new[-1])
                del (distance[-1])
                break
            count += 1
        num -= 1
    distance = distance / np.sum(distance)
    new = np.reshape(new, (-1, 3))
    new_ind = np.random.choice(new[:, 0], size=int(len(new)*prob), replace=False, p=distance)
    # print('i : {}, count: {}, MF_GL(x): {}, MF_GL(y): {}, MF_GL(z): {}'.format(i, count, new[count][0], new[count][1], new[count][2]))
    MF_GL.append(new)

#GRC_GL mapping
r_grc = 18
GRC_GL = []
temp_grc = []
ball_grc = spatial.cKDTree(total).query_ball_point(grc_points, r_grc)
for results in ball_grc:  # Setting temporal coordinates which points
    temp_grc.append(total[results])
for i in range(0, len(grc_points)):  # Go to each central point
    new_grc = []
    num_grc = len(temp_grc[i])  # Setting how many points around one center point(i)
    count_grc = 0
    distance_grc = []
    while num_grc >= 0:
        if grc_points[i, 1] - r_grc/4 <= temp_grc[i][num_grc - 1, 1] <= grc_points[i, 1] + r_grc/4 and grc_points[i, 2] - r_grc/4 <= temp_grc[i][num_grc - 1, 2] <= grc_points[i, 2] + r_grc/4:
            new_grc.append(temp_grc[i][num_grc - 1])
            distance_grc.append(np.sqrt((grc_points[i, 0] - new_grc[count_grc][0])**2 + (grc_points[i, 1] - new_grc[count_grc][1])**2 + (grc_points[i, 2] - new_grc[count_grc][2])**2))
            if distance_test(new_grc[count_grc], grc_points[i]) == 0:
                del (new_grc[-1])
                del (distance_grc[-1])
                break
            # print('i : {}, count: {}, GRC_GL(x): {}, GRC_GL(y): {}, GRC_GL(z): {}'.format(i, count_grc, new_grc[count_grc][0], new_grc[count_grc][1], new_grc[count_grc][2]))
            count_grc += 1
        num_grc -= 1
    distance_grc = distance_grc / np.sum(distance_grc)
    new_grc = np.reshape(new_grc, (-1, 3))
    if len(new_grc) !=0:
        new_grc = np.random.choice(new_grc[:, 0], size=int(len(new_grc)*prob), replace = False, p= distance_grc)
        GRC_GL.append(new_grc)

print(MF_GL)
print(GRC_GL)
print(goooooood)

#MF_GOC mapping
prob_MGOC = 1
r_MGOC = 13.7
MF_GOC = []
temp_MGOC = []
total_MGOC = np.vstack((MF, goc_points))
ball_MGOC = spatial.cKDTree(total_MGOC).query_ball_point(goc_points, r_MGOC)
for results in ball_MGOC:  # Setting temporal coordinates which points
    temp_MGOC.append(total_MGOC[results])
for i in range(0, len(goc_points)):  # Go to each central point
    new_MGOC = []
    num_MGOC = len(temp_MGOC[i])  # Setting how many points around one center point(i)
    count_MGOC = 0
    distance_MGOC = []
    while num_MGOC >= 0:
        # TODO: FIND THE MORE DETAIL GOC DIAMETER. IN OUR PAPER, WE USED 13.7 UM SPHERE
        if goc_points[i, 1] - r_MGOC <= temp_MGOC[i][num_MGOC - 1, 1] <= goc_points[i, 1] + r_grc/4 and goc_points[i, 2] - r_MGOC <= temp_MGOC[i][num_MGOC - 1, 2] <= goc_points[i, 2] + r_MGOC:
            new_MGOC.append(temp_MGOC[i][num_grc - 1])
            distance_MGOC.append(np.sqrt((goc_points[i, 0] - new_MGOC[count_MGOC][0])**2 + (goc_points[i, 1] - new_MGOC[count_MGOC][1])**2 + (goc_points[i, 2] - new_MGOC[count_MGOC][2])**2))
            if distance_test(new_MGOC[count_MGOC], goc_points[i]) == 0:
                del (new_MGOC[-1])
                del (distance_MGOC[-1])
                break
            # print('i : {}, count: {}, GRC_GL(x): {}, GRC_GL(y): {}, GRC_GL(z): {}'.format(i, count_grc, new_grc[count_grc][0], new_grc[count_grc][1], new_grc[count_grc][2]))
            count_MGOC += 1
        num_MGOC -= 1
    distance_MGOC = distance_MGOC / np.sum(distance_MGOC)
    new_MGOC = np.reshape(new_MGOC, (-1, 3))
    new_MGOC = np.random.choice(new_MGOC[:, 0], size=int(len(new_MGOC)*prob_MGOC), replace = False, p= distance_MGOC)
    MF_GOC.append(new_MGOC)


