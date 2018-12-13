import numpy as np
from scipy import spatial

points = np.loadtxt('noise_glo.txt')  # Glomeruli coordinates
grc_points = np.loadtxt('noise_grc.txt') # Granular cells coordinates

Transverse_range = 1500  # Y range
Horizontal_range = 700  # X range
Vertical_range = 200  # Z range
a_x = 24.55
a_y = 6.15
a_z = 15
z_mid = np.random.uniform(50, 150, 1) # Random z values

MF = np.loadtxt('readMF_final.txt') # Mossy Fiber coordinates
MF_GL = np.zeros((len(MF), 300, 3)) # Connection matrix with Mossy fiber and Glomerulus

points = np.loadtxt('noise_glo.txt')
grc_points = np.loadtxt('noise_grc.txt')
total = np.vstack((points, grc_points))
total_MF = np.vstack((MF, points))
semi_y = 64/2 #1500 / 2  # 64 / 2
semi_x = 84/2 #700 / 2  # 84 / 2
semi_z = 50/2 #50/2 #50 / 2  # 50 왜 50으로 잡았는지 알아야 함
# for i in range(0, np.size(points[:, 0])):
#     add_point_grc(points[i, :])
GRC_GL = np.zeros((np.size(grc_points[:, 0]), 100, 3))

# MF_GL kdtree_query_ball(x, total)
r_MF = semi_x
temp_MF = []
ball = spatial.cKDTree(total_MF).query_ball_point(MF, r_MF)
for results in ball:  # Setting temporal coordinates which points
    temp_MF.append(total_MF[results])
for i in range(0, len(MF)):  # Go to each central point
    new = []
    num = len(temp_MF[i])  # Setting how many points around one center point(i)
    count = 0
    while num >= 0:
        if MF[i, 1] - semi_y <= temp_MF[i][num - 1, 1] <= MF[i, 1] + semi_y and MF[i, 2] - semi_z <= temp_MF[i][num - 1, 2] <= MF[i, 2] + semi_z:
            MF_GL[i, count, :] = temp_MF[i][num - 1]
            print('i : {}, count: {}, MF_GL(x): {}, MF_GL(y): {}, MF_GL(z): {}'.format(i, count, MF_GL[i, count, 0], MF_GL[i, count, 1], MF_GL[i, count, 2]))
            count += 1
        num -= 1

#GRC_GL mapping
r_grc = 18
temp_grc = []
ball_grc = spatial.cKDTree(total).query_ball_point(grc_points, r_grc)
for results in ball_grc:  # Setting temporal coordinates which points
    temp_grc.append(total[results])
for i in range(0, len(grc_points)):  # Go to each central point
    new_grc = []
    num_grc = len(temp_grc[i])  # Setting how many points around one center point(i)
    count_grc = 0
    while num_grc >= 0:
        if grc_points[i, 1] - r_grc/4 <= temp_grc[i][num_grc - 1, 1] <= grc_points[i, 1] + r_grc/4 and grc_points[i, 2] - r_grc/4 <= temp_grc[i][num_grc - 1, 2] <= grc_points[i, 2] + r_grc/4:
            GRC_GL[i, count_grc, :]  = temp_grc[i][num_grc - 1]
            print('i : {}, count: {}, GRC_GL(x): {}, GRC_GL(y): {}, GRC_GL(z): {}'.format(i, count_grc, GRC_GL[i, count_grc, 0], GRC_GL[i, count_grc, 1], GRC_GL[i, count_grc, 2]))
            count_grc += 1
        num_grc -= 1
