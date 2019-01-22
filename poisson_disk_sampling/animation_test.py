import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

def y_inters(z):
    result = np.sqrt(np.abs(radius1 ** 2 - (z - center[1]) ** 2))
    return [center[0] + result, center[0] + (-1) * result]


def z_inters(y):
    result = np.sqrt(np.abs(radius1 ** 2 - (y - center[0]) ** 2))
    return [center[1] + result, center[1] + (-1) * result]


# Let's make angle and length
den_num = 4
radius1 = 100
den_len = np.random.uniform(low=70, high=100, size=4)
angle_total = np.arange(0, 360, 1) * 180 / (2 * np.pi)

# Let's draw a circle
center = np.empty((3, 1))
center[0] = np.random.uniform(low=0, high=1500, size=1)
center[1] = np.random.uniform(low=0, high=200, size=1)
center[2] = np.random.uniform(low=0, high=700, size=1)  # THIS IS CENTER OF X
print('Center: {}, {}'.format(center[0], center[1]))
y_total = radius1 * np.cos(angle_total) + center[0]
z_total = radius1 * np.sin(angle_total) + center[1]

no_point = []
for i in range(0, len(y_total)):
    if not 0 < z_total[i] < 200:
        if z_total[i] <= 0:
            if (center[0] - 100 < y_inters(0)[0] < center[0] + 100):
                no_point.append((y_inters(0)[0][0], 0))
            if (center[0] - 100 < y_inters(0)[1] < center[0] + 100):
                no_point.append((y_inters(0)[1][0], 0))
        elif 200 <= z_total[i]:
            if (center[0] - 100 < y_inters(200)[0] < center[0] + 100):
                no_point.append((y_inters(200)[0][0], 200))
            if (center[0] - 100 < y_inters(200)[1] < center[0] + 100):
                no_point.append((y_inters(200)[1][0], 200))

if len(no_point) != 0:
    no_point = np.unique(np.reshape(no_point, (-1, 2)), axis=0)
    print('no_point : {}'.format(no_point))
    line_a = np.sqrt((center[0] - no_point[0][0]) ** 2 + (center[1] - no_point[0][1]) ** 2)
    line_b = np.sqrt((center[0] - no_point[1][0]) ** 2 + (center[1] - no_point[1][1]) ** 2)
    line_c = np.sqrt((no_point[0][0] - no_point[1][0]) ** 2 + (no_point[0][1] - no_point[1][1]) ** 2)
    line_d = np.sqrt((no_point[0] - center[0]) ** 2)

    line_d = np.sqrt((no_point[0][1] - center[1]) ** 2)

    angle_a = np.clip(np.arccos(1 - (line_c ** 2 / (2 * radius1 ** 2))), 0, np.pi)
    angle_b = np.clip(np.arcsin(line_d / radius1), -np.pi / 2, np.pi / 2)
    if no_point[0, 1] == 0:
        angle_b = (2 * np.pi - (angle_a + angle_b))[0]

# Let's draw
a_test = np.linspace(0, 2 * np.pi, 100)
angle_re = np.arange(0, 2 * np.pi - 0.349, 0.349)  # I split the degree by 20
angle_candi = []
half = np.linspace(0, np.pi, len(angle_total))
x_candi = np.arange(0, np.pi - 0.349, 0.349)
x_candi = np.random.choice(x_candi, replace=False, size=4)
for i in range(0, len(angle_re)):
    if not (angle_b <= angle_re[i] <= (angle_a + angle_b)):
        angle_candi.append(angle_re[i])

angle = np.random.choice(angle_candi, replace=False, size=4)

# y = den_len * np.cos(angle) + center[0]
# z = den_len * np.sin(angle) + center[1]
y = den_len * np.sin(x_candi) * np.cos(angle) + center[0]
z = den_len * np.sin(x_candi) * np.sin(angle) + center[1]
x = den_len * np.cos(x_candi) + center[2]

print('angle_a:{}, angle_b:{}'.format(angle_a, angle_b))

#Let's make output
''' We will make two files: coordinates and index. Each column of index file is dendrite index and segment index'''
x0=[]
y0=[]
z0=[]
seg=[]
final_temp = []
seg_inval = 2 # Segment points interval length
seg_num = 10 # Segmentation number of each dendrite
for i in range(0, 4):
    x0.append(np.arange(0, den_len[i], seg_inval)*np.cos(x_candi[i])+center[2])
    y0.append(np.arange(0, den_len[i], seg_inval) * np.sin(x_candi[i]) * np.cos(angle[i])+center[0])
    z0.append(np.arange(0, den_len[i], seg_inval) * np.sin(x_candi[i])*np.sin(angle[i])+center[1])
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
            final_temp.append(np.hstack((seg[i][j][k], j)))
final_temp = np.reshape(final_temp, (-1, 5))
coordinates = final_temp[:, 0:3]
index = final_temp[:, 3:5]

# Drawing
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(radius1 * np.sin(half) * np.cos(angle_total) + center[0],
           radius1 * np.sin(half) * np.sin(angle_total) + center[1], radius1 * np.cos(half) + center[2], c='g', s=5)
ax.scatter(y, z, x, c='r')
print('dendrite: {}'.format(den_len))
total = angle_a + angle_b
for i in range(0, 4):
    ax.scatter(np.arange(0, den_len[i], 5) * np.sin(x_candi[i]) * np.cos(angle[i]) + center[0],
               np.arange(0, den_len[i], 5) * np.sin(x_candi[i]) * np.sin(angle[i]) + center[1],
               np.arange(0, den_len[i], 5) * np.cos(x_candi[i]) + center[2], color='black', s=10)
if len(no_point) != 0:
    for i in range(0, len(no_point)):
        ax.scatter(no_point[i][0], no_point[i][1], center[2], c='purple', s=50)
ax.scatter(center[0], center[1], center[2], c='orange')
ax.set_aspect('equal')
ax.set_xlabel('y')
ax.set_ylabel('z')
ax.set_zlabel('x')

print(index)
# rot_animation = animation.FuncAnimation(fig, rotate, frames=1000, interval=1000)
# plt.show(rot_animation)
# rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')
gif_filename = '3d rotating ball'
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
    plt.savefig(gif_filename + str(i).zfill(3) + '.png', bbox_inches='tight')

# plt.close()
# images = [PIL_Image.open(image) for image in glob.glob('images/' + gif_filename + '/*.png')]
# file_path_name = 'images/' + gif_filename + '.gif'
# writeGif(file_path_name, images, duration=0.1)
# IPdisplay.Image(url=file_path_name)