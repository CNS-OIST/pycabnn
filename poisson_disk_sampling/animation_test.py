import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cell_ind = 0
mid_temp = []
cell_num = 10
cc0 = np.random.uniform(low=0, high=1500, size = cell_num)
cc1 = np.random.uniform(low=0, high=200, size = cell_num)
cc2 = np.random.uniform(low=0, high=700, size = cell_num)

center_candi = np.stack((cc0, cc1, cc2), axis=1)

def y_inters(z):  # To get the intersection between z axis and The y coordinate
    result = np.sqrt(np.abs(max_length ** 2 - (z - center[1]) ** 2))
    return [center[0] + result, center[0] + (-1) * result]


for center in center_candi:
    # Let's make angle and length
    den_num = 4  # parameter
    max_length = 100  # parameter: Maximum dendrite length
    den_len = np.random.uniform(low=70, high=100, size=4)  # There is no reason for min length, parameter
    seg_inval = 2  # parameter : Segment points interval length
    seg_num = 10  # parameter : Segmentation number of each dendrite
    angle_total = np.arange(0, 360, 1) * 180 / (2 * np.pi)

    # Let's draw a circle
    y_total = max_length * np.cos(angle_total) + center[1]  # To draw boarder line(circle)
    z_total = max_length * np.sin(angle_total) + center[2]
    no_point = []  # Let's check the angle we should avoid
    for i in range(0, len(y_total)):
        if not 0 < z_total[i] < 200:
            if z_total[i] < 0:
                if (center[1] - max_length < y_inters(0)[0] < center[1] + max_length):
                    no_point.append((y_inters(0)[0], 0))
                if (center[1] - max_length < y_inters(0)[1] < center[1] + max_length):
                    no_point.append((y_inters(0)[1], 0))
            elif 200 < z_total[i]:
                if (center[1] - max_length < y_inters(200)[0] < center[1] + max_length):
                    no_point.append((y_inters(200)[0], 200))
                if (center[1] - max_length < y_inters(200)[1] < center[1] + max_length):
                    no_point.append((y_inters(200)[1], 200))

    if len(no_point) != 0:
        no_point = np.unique(np.reshape(no_point, (-1, 2)), axis=0)
        if len(no_point) == 2:
            print('no_point : {}'.format(no_point))
            line_a = np.sqrt((center[1] - no_point[0][0]) ** 2 + (center[2] - no_point[0][1]) ** 2)
            line_b = np.sqrt((center[1] - no_point[1][0]) ** 2 + (center[2] - no_point[1][1]) ** 2)
            line_c = np.sqrt((no_point[0][0] - no_point[1][0]) ** 2 + (no_point[0][1] - no_point[1][1]) ** 2)
            #         line_d = np.sqrt((no_point[0]-center[1])**2)
            line_d = np.sqrt((no_point[0][1] - center[2]) ** 2)

            angle_a = np.clip(np.arccos(1 - (line_c ** 2 / (2 * max_length ** 2))), 0, np.pi)  # Law of cosines
            angle_b = np.clip(np.arcsin(line_d / max_length), -np.pi / 2,
                              np.pi / 2)  # Law of sines (angle_b will indicate the angle which angle_a will start)
            if no_point[0, 1] == 0:
                angle_b = (2 * np.pi - (angle_a + angle_b))[0]

    # Let's draw
    angle_pre = np.arange(0, 2 * np.pi - 0.349, 0.349)
    angle_candi = []
    x_candi = np.arange(0, np.pi - 0.349, 0.349)
    x_candi = np.random.choice(x_candi, replace=False, size=4)

    if len(no_point) == 2:
        for i in range(0, len(angle_pre)):
            if not (angle_b <= angle_pre[i] <= (angle_a + angle_b)):
                angle_candi.append(angle_pre[i])
    else:
        angle_candi = angle_pre

    angle = np.random.choice(angle_candi, replace=False, size=4)

    y = den_len * np.sin(x_candi) * np.cos(angle) + center[1]
    z = den_len * np.sin(x_candi) * np.sin(angle) + center[2]
    x = den_len * np.cos(x_candi) + center[0]

    # Let's make output
    ''' We will make two files: coordinates and index. Each column of index file is dendrite index and segment index'''
    x0 = []
    y0 = []
    z0 = []
    seg = []
    for i in range(0, 4):
        x0.append(np.arange(0, den_len[i], seg_inval) * np.cos(x_candi[i]) + center[0])
        y0.append(np.arange(0, den_len[i], seg_inval) * np.sin(x_candi[i]) * np.cos(angle[i]) + center[1])
        z0.append(np.arange(0, den_len[i], seg_inval) * np.sin(x_candi[i]) * np.sin(angle[i]) + center[2])
    for i in range(0, den_num):
        temp = []
        for j in range(0, len(x0[i])):
            temp.append(np.vstack((x0[i][j], y0[i][j], z0[i][j], i)))
        temp = np.reshape(temp, (-1, 4))
        seg.append(np.array_split(temp, seg_num))
    del temp
    for i in range(0, 4):
        for j in range(0, 10):
            for k in range(0, len(seg[i][j])):
                mid_temp.append(np.hstack((seg[i][j][k], j, cell_ind, center[0], center[1], center[2])))
    final_temp = np.reshape(mid_temp, (-1, 9))
    cell_ind += 1
coords = final_temp[:, 0:3]
segs = final_temp[:, 3:5]
cell = final_temp[:, 5:]

    # Drawing
seg_inval = 2  # Segment points interval length
seg_num = 10  # Segmentation number of each dendrite
half = np.linspace(0, np.pi, len(angle_total))
fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords[:, 1], coords[:, 2], coords[:, 0], c='black', s=5)
cell = np.unique(cell, axis=0)
ax.scatter(cell[:, 2], cell[:, 3], cell[:, 1], c='orange', s=20)
for i in range(0, len(cell)):
    ax.scatter(max_length * np.cos(angle_total) * np.sin(half) + cell[i, 2],
               max_length * np.sin(angle_total) * np.sin(half) + cell[i, 3], max_length * np.cos(half) + cell[i, 1],
               c='green', s=1)
ax.set_xlabel('y')
ax.set_ylabel('z')
ax.set_zlabel('x')
# ax.set_aspect('equal')
ax.axis('tight')
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.001)