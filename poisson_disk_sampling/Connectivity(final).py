import numpy as np
from scipy import spatial
import time

def distance_test(x, p):
    return np.sqrt((x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2 + (x[2] - p[2]) ** 2)


MF = np.loadtxt('mf_coordinates.txt') # Mossy Fiber coordinates
points = np.loadtxt('glo_coordinates.txt')  # Glomeruli coordinates
grc_points = np.loadtxt('grc_coordinates.txt') # Granular cells coordinates
goc_points = np.loadtxt('goc_coordinates.txt') # Golgi cell coordinates

Transverse_range = 1500  # Y range
Horizontal_range = 700  # X range
Vertical_range = 200  # Z range
a_x = 24.55
a_y = 6.15
a_z = 15

total_MGoc = np.vstack((goc_points, MF))
total = np.vstack((grc_points, points)) #This order is very important especially for deleting from KDTree process.
total_MF = np.vstack((MF, points)) # Same with 'total'
semi_y = 64/2 #1500 / 2  # 64 / 2
semi_x = 84/2 #700 / 2  # 84 / 2
semi_z = 50/2 #50/2 #50 / 2
# for i in range(0, np.size(points[:, 0])):
#     add_point_grc(points[i, :])

# Mossy fiber to Glomeruli connectivity
prob = 1
r_MF = semi_x
MF_GL = []
temp_MF = []
MF_GL_temp = []
ball = spatial.cKDTree(total_MF).query_ball_point(MF, r_MF)
for i in range(0, len(ball)): # Deleting MF coordinates in KDTree
    num = len(ball[i])
    while num >= 0:
        if ball[i][num-1] < len(MF):
            del ball[i][num-1]
        num -= 1
for results in ball:  # Setting temporal coordinates which points
    temp_MF.append(total_MF[results])
for i in range(0, len(MF)):  # Go to each central point
    new = []
    distance= []
    num = len(temp_MF[i])  # Setting how many points around one center point(i)
    count = 0
    while num >= 0:
        if MF[i, 1] - semi_y <= temp_MF[i][num - 1, 1] <= MF[i, 1] + semi_y and MF[i, 2] - semi_z <= temp_MF[i][num - 1, 2] <= MF[i, 2] + semi_z: # Restrict the searching area
            new.append(temp_MF[i][num - 1])
            distance.append(np.sqrt((MF[i, 0] - new[count][0])**2 + (MF[i, 1] - new[count][1])**2 + (MF[i, 2] - new[count][2])**2)) # Measuring the distance
            count += 1
        num -= 1
    distance = distance / np.sum(distance) # Distance norm to make a weight
    new = np.reshape(new, (-1, 3))
    new_ind = np.random.choice(np.arange(0, len(new), 1), size=int(len(new) * prob), replace=False, p=distance)
    for j in new_ind:
        MF_GL_temp.append((i, new[j])) # Now, we can recognize which number of MF connects with glomerulus
MF_GL = np.zeros((len(MF_GL_temp), 3))
for k in range(0, len(MF_GL_temp)):
    MF_GL[k, 0] = MF_GL_temp[k][0] # Source index
    MF_GL[k, 1] = np.argwhere(MF_GL_temp[k][1][0] == points[:, 0]) # Target index # The hardest part. This part takes lots of time (Now, it just compare x coordinates if it makes errors, we should compare y co too)
    MF_GL[k, 2] = distance_test(MF[MF_GL_temp[k][0]], MF_GL_temp[k][1]) # Distance between the source and target
del MF_GL_temp
np.savetxt('MF_GL.txt', MF_GL)
print('Complete MF_GL')

#Granular cell to glomeruli connectivity
prob = 1
r_grc = 18
GRC_GL = []
temp_grc = []
GRC_GL_temp = []
ball_grc = spatial.cKDTree(total).query_ball_point(grc_points, r_grc)
for i in range(0, len(ball_grc)): # Deleting grc_points coordinates in KDTree
    num = len(ball_grc[i])
    while num >= 0:
        if ball_grc[i][num-1] < len(grc_points):
            del ball_grc[i][num-1]
        num -= 1
for results in ball_grc:  # Setting temporal coordinates which points
    temp_grc.append(total[results])
for i in range(0, len(grc_points)):  # Go to each central point
    new_grc = []
    num_grc = len(temp_grc[i])  # Setting how many points around one center point(i)
    count_grc = 0
    distance_grc = []
    while num_grc >= 0:
         #Since searching is processed at the grc center, we don't have to restrict searching area #if grc_points[i, 1] - 7.5 <= temp_grc[i][num_grc - 1, 1] <= grc_points[i, 1] + 7.5 and grc_points[i, 2] - 7.5 <= temp_grc[i][num_grc - 1, 2] <= grc_points[i, 2] + 7.5:
        new_grc.append(temp_grc[i][num_grc - 1])
        distance_grc.append(np.sqrt((grc_points[i, 0] - new_grc[count_grc][0])**2 + (grc_points[i, 1] - new_grc[count_grc][1])**2 + (grc_points[i, 2] - new_grc[count_grc][2])**2))
        count_grc += 1
        num_grc -= 1
    distance_grc = distance_grc / np.sum(distance_grc)
    new_grc = np.reshape(new_grc, (-1, 3))
    new_ind = np.random.choice(np.arange(0, len(new_grc), 1), size=int(len(new_grc) * prob), replace=False, p=distance_grc)
    for j in new_ind:
        GRC_GL_temp.append((i, new_grc[j])) # Now, we can recognize which number of grc_points connects with glomerulus
GRC_GL = np.zeros((len(GRC_GL_temp), 3))
for k in range(0, len(GRC_GL_temp)):
    GRC_GL[k, 0] = GRC_GL_temp[k][0] # Source index
    GRC_GL[k, 1] = np.argwhere(GRC_GL_temp[k][1][0] == points[:, 0]) # Target index # The hardest part. This part takes lots of time (Now, it just compare x coordinates if it makes errors, we should compare y co too)
    GRC_GL[k, 2] = distance_test(grc_points[GRC_GL_temp[k][0]], GRC_GL_temp[k][1]) # Distance between the source and target
del GRC_GL_temp
np.savetxt('GRC_GL.txt', GRC_GL)
print('Complete GRC_GL')

#Golgi cell to Mossy fiber connectivity
prob = 1
r_MGoc = 100
MF_Goc = []
temp_MGoc = []
MF_Goc_temp = []
ball_MGoc = spatial.cKDTree(total_MGoc).query_ball_point(goc_points, r_MGoc)
for i in range(0, len(ball_MGoc)): # Deleting goc coordinates in KDTree
    num = len(ball_MGoc[i])
    while num >= 0:
        if ball_MGoc[i][num-1] < len(goc_points):
            del ball_MGoc[i][num-1]
        num -= 1
for results in ball_MGoc:  # Setting temporal coordinates which points
    temp_MGoc.append(total_MGoc[results])
for i in range(0, len(goc_points)):  # Go to each central point
    new_MGoc = []
    distance_MGoc= []
    num = len(temp_MGoc[i])  # Setting how many points around one center point(i)
    count = 0
    while num >= 0:
        new_MGoc.append(temp_MGoc[i][num - 1])
        distance_MGoc.append(np.sqrt((goc_points[i, 0] - new_MGoc[count][0])**2 + (goc_points[i, 1] - new_MGoc[count][1])**2 + (goc_points[i, 2] - new_MGoc[count][2])**2)) # Measuring the distance
        count += 1
        num -= 1
    distance_MGoc = distance_MGoc / np.sum(distance_MGoc) # Distance norm to make a weight
    new_MGoc = np.reshape(new_MGoc, (-1, 3))
    new_ind = np.random.choice(np.arange(0, len(new_MGoc), 1), size=int(len(new_MGoc) * prob), replace=False, p=distance_MGoc)
    for j in new_ind:
        MF_Goc_temp.append((i, new_MGoc[j])) # Now, we can recognize which number of goc_points connects with glomerulus
MF_Goc = np.zeros((len(MF_Goc_temp), 3))
for k in range(0, len(MF_Goc_temp)):
    MF_Goc[k, 0] = MF_Goc_temp[k][0] # Source index
    MF_Goc[k, 1] = np.argwhere(MF_Goc_temp[k][1][0] == MF[:, 0]) # Target index # The hardest part. This part takes lots of time (Now, it just compare x coordinates if it makes errors, we should compare y co too)
    MF_Goc[k, 2] = distance_test(goc_points[MF_Goc_temp[k][0]], MF_Goc_temp[k][1]) # Distance between the source and target
del MF_Goc_temp
np.savetxt('MF_Goc.txt', MF_Goc)
print('Complete MF_Goc')


#Mossy fiber to golgi cell mapping (Optional)
# prob_MGOC = 1
# r_MGOC = 13.7
# MF_GOC = []
# temp_MGOC = []
# total_MGOC = np.vstack((MF, goc_points))
# ball_MGOC = spatial.cKDTree(total_MGOC).query_ball_point(goc_points, r_MGOC)
# for results in ball_MGOC:  # Setting temporal coordinates which points
#     temp_MGOC.append(total_MGOC[results])
# for i in range(0, len(goc_points)):  # Go to each central point
#     new_MGOC = []
#     num_MGOC = len(temp_MGOC[i])  # Setting how many points around one center point(i)
#     count_MGOC = 0
#     distance_MGOC = []
#     while num_MGOC >= 0:
#         # TODO: FIND THE MORE DETAIL GOC DIAMETER. IN OUR PAPER, WE USED 13.7 UM SPHERE
#         if goc_points[i, 1] - r_MGOC <= temp_MGOC[i][num_MGOC - 1, 1] <= goc_points[i, 1] + r_grc/4 and goc_points[i, 2] - r_MGOC <= temp_MGOC[i][num_MGOC - 1, 2] <= goc_points[i, 2] + r_MGOC:
#             new_MGOC.append(temp_MGOC[i][num_grc - 1])
#             distance_MGOC.append(np.sqrt((goc_points[i, 0] - new_MGOC[count_MGOC][0])**2 + (goc_points[i, 1] - new_MGOC[count_MGOC][1])**2 + (goc_points[i, 2] - new_MGOC[count_MGOC][2])**2))
#             if distance_test(new_MGOC[count_MGOC], goc_points[i]) == 0:
#                 del (new_MGOC[-1])
#                 del (distance_MGOC[-1])
#                 break
#             # print('i : {}, count: {}, GRC_GL(x): {}, GRC_GL(y): {}, GRC_GL(z): {}'.format(i, count_grc, new_grc[count_grc][0], new_grc[count_grc][1], new_grc[count_grc][2]))
#             count_MGOC += 1
#         num_MGOC -= 1
#     distance_MGOC = distance_MGOC / np.sum(distance_MGOC)
#     new_MGOC = np.reshape(new_MGOC, (-1, 3))
#     new_MGOC = np.random.choice(new_MGOC[:, 0], size=int(len(new_MGOC)*prob_MGOC), replace = False, p= distance_MGOC)
#     MF_GOC.append(new_MGOC)


