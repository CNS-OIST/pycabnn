import numpy as np
import matplotlib.pyplot as plt
from GL_poisson import Bridson_sampling
from mpl_toolkits.mplot3d import Axes3D

##Functions###
def neighborhood(shape, index, n):
    row, col, third = index
    row0, row1 = max(row - n, 0), min(row + n + 1, shape[0])
    col0, col1 = max(col - n, 0), min(col + n + 1, shape[1])
    third0, third1 = max(third - n, 0), min(third + n + 1, shape[2])
    I = np.stack(np.mgrid[row0:row1, col0:col1, third0:third1], axis=3)
    I = I.reshape(I.size // 3, 3).tolist()
    return I

def add_point(p):
    i, j, u = int(p[0] / cellsize), int(p[1] / cellsize), int(p[2] / cellsize)
    P[i, j, u] = p

def in_limit(p):
    return 0 < p[0] < Horizontal_range and 0 < p[1] < Transverse_range and 0 < p[2] < Vertical_range

def ifinEplipse(p, Ell_xo, Ell_yo, Ell_zo):
    ptin = ((p[0] - Ell_xo) ** 2 / (semi_x ** 2)) + ((p[1] - Ell_yo) ** 2 / (semi_y ** 2)) + (
                (p[2] - Ell_zo) ** 2 / (semi_z ** 2))
    if ptin <= 1:
        return True
    else:
        return False

def in_neighborhood(Ell_xo, Ell_yo, Ell_zo):
    i, j, u = int(Ell_xo / cellsize), int(Ell_yo / cellsize), int(Ell_zo / cellsize)
    M[i, j, u] = True
    for (i, j, u) in N[(i, j, u)]:
        M[i, j, u] = True
    return P[M]
def non_zero(p):
    if p[0] != 0 and p[1] != 0 and p[2] != 0: return True
    else: return False

##Parameters ##
Transverse_range = 1500  # Y range
Horizontal_range = 700  # X range
Vertical_range = 200  # 140 for daria
numRosetteperMF_mu = 750
numRosetteperMF_sd = 37
numpmperMF_mu = 7
numpmperMF_sd = 1
box_fac = 2.5
semi_x = 64/2 #1500 / 2  # 64 / 2
semi_y = 84/2 #700 / 2  # 84 / 2
semi_z = 50/2 #50/2 #50 / 2  # 50 왜 50으로 잡았는지 알아야 함
insvolc = 0
offset = 0
Box_Id_x = 64 + 40  # 무슨 의도인지 잘 모르겠음
Box_Id_y = 84 + 40 * box_fac
Box_Id_z = semi_z * 2
z_mid = np.random.randint(50, 150)
offset = 0
k = 0
numRosetteperMF_mu = 750
numRosetteperMF_sd = 37
numpmperMF_mu = 7
numpmperMF_sd = 1
select_point = []

# Import MF coordinates
MF_coordinates = np.loadtxt('readMF.txt')  # Importing MF coordinates
nMF = len(MF_coordinates)
MF_GL = np.zeros((nMF, 20, 3))

##Searching parameters##
cellsize = 18/np.sqrt(3)
rows = int(np.ceil(Horizontal_range / cellsize))
cols = int(np.ceil(Transverse_range / cellsize))
third = int(np.ceil(Vertical_range / cellsize))
P = np.zeros((rows, cols, third, 3), dtype=np.float32)
M = np.zeros((rows, cols, third), dtype=bool)

#Define Searching Area 
N = {}
for i in range(rows):
    for j in range(cols):
        for u in range(third):
            N[(i, j, u)] = neighborhood(M.shape, (i, j, u), 3)
print('Complete area searching')

#Generate Poisson sampling points
points = Bridson_sampling((Horizontal_range, Transverse_range, Vertical_range), 20, 18000, True)
#Marking the GL point cells
for k in range(0, 18000):
    add_point(points[k, :])
    print('k is {}'.format(k))

fig = plt.figure(1)
ax1 = fig.add_subplot(1, 1, 1, projection = '3d')
for k in range(0, int(nMF)):
    count = 0
    Ell_xo = MF_coordinates[k, 0]
    Ell_yo = MF_coordinates[k, 1] + offset
    Ell_zo = z_mid
    print('Complete making {}th mossy fiber coordination'.format(k))
    Exist_point = in_neighborhood(Ell_xo, Ell_yo, Ell_zo)
    point_num = np.size(Exist_point[:, 0])
    while point_num > 0:
        candidate = Exist_point[point_num-1, :]
        if non_zero(candidate) and ifinEplipse(candidate, Ell_xo, Ell_yo, Ell_zo) and in_limit(candidate): #범위에 안드는녀석은 포함시키면 안될듯? 이건 connection을 만드는 행위니까
            # if inside volume record the coor
            print('{}'.format(candidate))
            print('count number : {}'.format(count))
#             MF_GL[k, count, 0] = candidate[0, 0]
#             MF_GL[k, count, 1] = candidate[0, 1]
#             MF_GL[k, count, 2] = candidate[0, 2]
            MF_GL[k, count, :] = candidate
            print('k : {}, MF_GL(x): {}, MF_GL(y): {}, MF_GL(z): {}'.format(k, MF_GL[k, count, 0], MF_GL[k, count, 1], MF_GL[k, count, 2]))
            # ax1.scatter(GLx, GLy, s=10, c='r')
            ax1.scatter(MF_GL[k, 0, 0], MF_GL[k, 0, 1], MF_GL[k, 0, 2], s=10, c='r')
            ax1.scatter(MF_GL[k, 1, 0], MF_GL[k, 1, 1], MF_GL[k, 1, 2], s=10, c='y')
            ax1.scatter(MF_GL[k, 2, 0], MF_GL[k, 2, 1], MF_GL[k, 2, 2], s=10, c='green')
            ax1.scatter(MF_GL[k, 3:, 0], MF_GL[k, 3:, 1], MF_GL[k, 3:, 2], s=10, c='blue')
#                 MF_GL[k, count, 0] = P[M][0]
#                 MF_GL[k, count, 1] = P[M][1]
#                 MF_GL[k, count, 2] = P[M][2]
#                 print('i : {}, MF_GL(x): {}, MF_GL(y): {}, MF_GL(z): {}'.format(i, MF_GL[k, count, 0], MF_GL[k, count, 1], MF_GL[k, count, 2]))
#                 ax1.scatter(MF_GL[k, count, 0], MF_GL[k, count, 1], MF_GL[k, count, 2], s=10, c='black')
#             ax1.scatter(candidate[:, 0], candidate[:, 1], candidate[:, 2], s=10, c='black')
            count += 1
        point_num -= 1

s = np.random.seed(1) # 1) 왜 seed를 고정하는지 질문하기

Shortaxis = 1500  # 185%eval (readParameters ('GoCxrange', 'Parameters.hoc'))  % um
Longaxis = 700  # 185%eval (readParameters ('GoCxrange', 'Parameters.hoc'))  % um
MFdensity = 1650  # 5000;%cells/mm2%190  추가확인필요

#fid = np.loadtxt('datasp.dat', delimiter = '\t')   #왜 필요한지? (아직은 필요x)
box_fac = 2.5
Xinstantiate = 64 + 40  # 297+40
Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac
# numMF = (Longaxis + (2 * Xinstantiate)) * (Shortaxis + (2 * Yinstantiate)) * MFdensity * 1e-6
numMF = 3009
plotMF = 1
fcoor = np.loadtxt('readMF.txt')
dt = 0.025

## Generate MF Coordinates ##
MF_coordinates= np.loadtxt('readMF.txt')

# plt.figure(2)
# plt.scatter(pts[:, 0], pts[:, 1])
# for i in range(0, int(numMF)):
#     MF_coordinates[i, 0] = np.random.randint(0 - Xinstantiate, Longaxis + Xinstantiate)
#     MF_coordinates[i, 1] = np.random.randint(0 - Yinstantiate, Shortaxis + Yinstantiate)
#     print('{}, {}' .format(MF_coordinates[i, 0], MF_coordinates[i, 1]))


    # fcoor.close()
#plt.scatter(MF_coordinates[:, 0], MF_coordinates[:, 1])
#plt.show()

centrey = Shortaxis / 2
centrex = Longaxis / 2
radius = 100
finalMF = []
finalMF = (MF_coordinates[:, 0] - centrex) ** 2 + (MF_coordinates[:, 1] - centrey) ** 2
finalMF[finalMF <= radius ** 2] = 1 #activated
finalMF[finalMF > radius ** 2] = 0  #inactivated

    # Second spatial kernal 1250, 350
finalMF1 = []
centrex1 = Shortaxis / 2
centrey1 = Longaxis / 2
finalMF1 = (MF_coordinates[:, 0] - centrex1) ** 2 + (MF_coordinates[:, 1] - centrey1) ** 2
finalMF1[finalMF1 <= radius ** 2] = 1 #activated
finalMF1[finalMF1 > radius ** 2] = 0 #inactivated
find_ac1 = np.where(finalMF1)
finalMF1[find_ac1] = 1

finalMF2 = []
centrex2 = Shortaxis / 2
centrey2 = Longaxis / 2
finalMF2 = (MF_coordinates[:, 0] - centrex2) ** 2 + (MF_coordinates[:, 1] - centrey2) ** 2
finalMF2[finalMF2 <= radius * radius] = 1
finalMF2[finalMF2 > radius * radius] = 0
find_ac2 = np.where(finalMF2)
finalMF2[find_ac2] = 1

fig1 = plt.figure(2)
ax2 = fig1.add_subplot(1, 1, 1, projection = '3d')
# Plot activatd and inactivated MF
if plotMF ==1:
    for i in range(0, int(numMF)):
        if finalMF[i] ==1:
            ax2.scatter(MF_coordinates[i, 0], MF_coordinates[i, 1], c = 'r')
            ax2.scatter(MF_GL[i, :, 0], MF_GL[i, :, 1], MF_GL[i, :, 2], c='yellow')
#             ax1.ylabel('long axis um')
#             ax1.xlabel('short axis um')
        else:
            ax2scatter(MF_coordinates[i, 0], MF_coordinates[i, 1], c='blue')
            ax2.scatter(MF_GL[i, :, 0], MF_GL[i, :, 1], MF_GL[i, :, 2], s=1, c='black')

# Plot activatd and inactivated MF
if plotMF ==1:
    for i in range(0, int(numMF)):
        if finalMF[i] ==1:
            plt.figure(3)
            plt.scatter(MF_coordinates[i, 0], MF_coordinates[i, 1], c = 'r')
            plt.scatter(MF_GL[i, :, 0], MF_GL[i, :, 1], c='yellow')
#             ax1.ylabel('long axis um')
#             ax1.xlabel('short axis um')
        else:
            plt.scatter(MF_coordinates[i, 0], MF_coordinates[i, 1], c='blue')
            plt.scatter(MF_GL[i, :, 0], MF_GL[i, :, 1], s=1, c='black')

plt.show()