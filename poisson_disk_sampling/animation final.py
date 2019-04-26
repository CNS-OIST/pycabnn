import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def y_inters(z): # To get the intersection between z axis and The y coordinate
    result = np.sqrt(np.abs(radius1**2-(z-center[1])**2))
    return [center[0]+result, center[0]+(-1)*result]

# def z_inters(y):
#     result = np.sqrt(np.abs(radius1**2-(y-center[0])**2))
#     return [center[1]+result, center[1]+(-1)*result]
cell_ind = 0
mid_temp=[]
cc1 = np.random.uniform(low=0, high=1500, size=30)
cc2 = np.random.uniform(low=0, high=200, size=30)
cc0 = np.random.uniform(low=0, high=700, size=30)

center_candi = np.stack((cc0, cc1, cc2), axis=1)

for center in center_candi:
    #Let's make angle and length
    den_num = 4
    radius1=100 # Maximum dendrite length
    den_len=np.random.uniform(low=70, high=100, size=4) #There is no reason for min length
    angle_total = np.arange(0, 360, 1) * 180/(2*np.pi)

    #Let's draw a circle
#     center[2] = np.random.uniform(low=0, high=700, size=1) # THIS IS A CENTER OF X coordinate
#     center[0] = np.random.uniform(low=0, high=1500, size=1)
#     center[1] = np.random.uniform(low=0, high=200, size=1)
    print('Center: {}, {}'.format(center[0], center[1]))
    y_total = radius1 * np.cos(angle_total) + center[1] # To draw boarder line(circle)
    z_total = radius1 * np.sin(angle_total) + center[2]


    no_point = [] # Let's check the angle we should avoid
    for i in range(0, len(y_total)):
        if not 0 < z_total[i] < 200:
            if z_total[i] < 0:
                if (center[1]-100 < y_inters(0)[0] < center[1]+100):
                    no_point.append((y_inters(0)[0], 0))
                if (center[1]-100 < y_inters(0)[1] < center[1]+100):
                    no_point.append((y_inters(0)[1], 0))
            elif 200 < z_total[i]:
                if (center[1]-100 < y_inters(200)[0] < center[1]+100):
                    no_point.append((y_inters(200)[0], 200))
                if (center[1]-100 < y_inters(200)[1] < center[1]+100):
                    no_point.append((y_inters(200)[1], 200))

    if len(no_point)!= 0:
        no_point = np.unique(np.reshape(no_point, (-1, 2)), axis=0)
        if len(no_point) == 2:
            print('no_point : {}'.format(no_point))
            line_a = np.sqrt((center[1]-no_point[0][0])**2+(center[2]-no_point[0][1])**2)
            line_b = np.sqrt((center[1]-no_point[1][0])**2+(center[2]-no_point[1][1])**2)
            line_c = np.sqrt((no_point[0][0]-no_point[1][0])**2+(no_point[0][1]-no_point[1][1])**2)
    #         line_d = np.sqrt((no_point[0]-center[1])**2)
            line_d = np.sqrt((no_point[0][1]-center[2])**2)

            angle_a = np.clip(np.arccos(1-(line_c**2/(2*radius1**2))), 0, np.pi) # Law of cosines
            angle_b = np.clip(np.arcsin(line_d/radius1), -np.pi/2, np.pi/2) # Law of sines (angle_b will indicate the angle which angle_a will start)
            if no_point[0, 1] == 0:
                angle_b = (2*np.pi-(angle_a + angle_b))[0]

    #Let's draw
    angle_pre = np.arange(0, 2*np.pi-0.349, 0.349)
    angle_candi = []
    x_candi = np.arange(0, np.pi-0.349, 0.349)
    x_candi = np.random.choice(x_candi, replace=False, size=4)
    if len(no_point) == 2:
        for i in range(0, len(angle_pre)):
            if not (angle_b <= angle_pre[i] <= (angle_a + angle_b)):
                angle_candi.append(angle_pre[i])
        print('angle_a:{}, angle_b:{}'.format(angle_a, angle_b))
    else: angle_candi = angle_pre

    angle = np.random.choice(angle_candi, replace=False, size=4)

    y = den_len * np.sin(x_candi) * np.cos(angle) + center[1]
    z = den_len * np.sin(x_candi) * np.sin(angle) + center[2]
    x = den_len * np.cos(x_candi) + center[0]



    #Let's make output
    ''' We will make two files: coordinates and index. Each column of index file is dendrite index and segment index'''
    x0=[]
    y0=[]
    z0=[]
    seg=[]
    seg_inval = 2 # Segment points interval length
    seg_num = 10 # Segmentation number of each dendrite
    for i in range(0, 4):
        x0.append(np.arange(0, den_len[i], seg_inval)*np.cos(x_candi[i])+center[0])
        y0.append(np.arange(0, den_len[i], seg_inval) * np.sin(x_candi[i]) * np.cos(angle[i])+center[1])
        z0.append(np.arange(0, den_len[i], seg_inval) * np.sin(x_candi[i])*np.sin(angle[i])+center[2])
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

#Drawing
seg_inval = 2 # Segment points interval length
seg_num = 10 # Segmentation number of each dendrite
half = np.linspace(0, np.pi, len(angle_total))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords[:, 1], coords[:, 2], coords[:, 0], c='black', s=5)
ax.scatter(cell[:, 2], cell[:, 3], cell[:, 1], c='orange', s=20)
green_cell = np.unique(cell, axis=0)
for i in range(0, len(green_cell)):
    ax.scatter(radius1*np.cos(angle_total)*np.sin(half)+green_cell[i, 2], radius1*np.sin(angle_total)*np.sin(half)+green_cell[i, 3], radius1*np.cos(half)+green_cell[i, 1], c='green', s=1)
# for i in range(0, len(coords)):
#     ax.scatter(radius1 * np.sin(half) * np.cos(angle_total) + cell_ind[i, 2], radius1 * np.sin(half) * np.sin(angle_total) + cell_ind[i, 3], radius1*np.cos(half)+ cell_ind[i, 1], c='g', s=5)
#     ax.scatter(cell_ind[i, 2], cell_ind[i, 3], cell_ind[i, 1], c='r')
#     ax.scatter(coords[i, 1], coords[i, 2], coords[i, 0], c='black')
# print('dendrite: {}'.format(den_len))
# #     total = angle_a + angle_b

# #     for i in range(0, 4):
# #         ax.scatter(np.arange(0, den_len[i], seg_inval) * np.sin(x_candi[i]) * np.cos(angle[i])+center[0], np.arange(0, den_len[i], seg_inval) * np.sin(x_candi[i])*np.sin(angle[i])+center[1], np.arange(0, den_len[i], seg_inval)*np.cos(x_candi[i])+center[2], color='black', s=10)
# #     if len(no_point) !=0 :
# #         for i in range(0, len(no_point)):
# #             ax.scatter(no_point[i][0], no_point[i][1], center[2], c='purple', s=50)
# #     ax.scatter(center[0], center[1], center[2], c='orange')
# ax.set_aspect('equal')
ax.set_xlabel('y')
ax.set_ylabel('z')
ax.set_zlabel('x')
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
