import numpy as np
import matplotlib.pyplot as plt
from GL_poisson import Bridson_sampling

##Functions###
def neighborhood(shape, index, n):
    row, col, third = index
    row0, row1 = max(row - n, 0), min(row + n + 1, shape[0])
    col0, col1 = max(col - n, 0), min(col + n + 1, shape[1])
    third0, third1 = max(third - n, 0), min(third + n + 1, shape[2])
    I = np.stack(np.mgrid[row0:row1, col0:col1, third0:third1], axis=3)
    I = I.reshape(I.size // 3, 3).tolist()
    I.remove([row, col, third])
    return I

def add_point(p):
    i, j, u = int(p[0] / cellsize), int(p[1] / cellsize), int(p[2] / cellsize)
    P[i, j, u] = p #Dictionary 로 p coordinates까지 저장하게 하는건..?

def ifinsideElipse(Ell_xo, Ell_yo, Ell_zo):
    i, j, u = int(Ell_xo / cellsize), int(Ell_yo / cellsize), int(Ell_zo / cellsize)
    ptins = ((P[i] - Ell_xo) ** 2 / (semi_x ** 2)) + ((P[j] - Ell_yo) ** 2 / (semi_y ** 2)) + (
                (P[u] - Ell_zo) ** 2 / (semi_z ** 2))
    if M[i, j, u]:
        return True
    for (i, j, u) in N[(i, j, u)]:
        if M[i, j, u] and ptins <= 1:
            return True
    return False

##Parameters ##
Transverse_range = 1500  # Y range
Horizontal_range = 700  # X range
Vertical_range = 430  # 140 for daria
numRosetteperMF_mu = 750
numRosetteperMF_sd = 37
numpmperMF_mu = 7
numpmperMF_sd = 1
box_fac = 2.5
semi_x = 1500 / 2  # 64 / 2
semi_y = 700 / 2  # 84 / 2
semi_z = 50 / 2  # 50 왜 50으로 잡았는지 알아야 함
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

# Import MF coordinates
MF_coordinates = np.loadtxt('MFCr.dat')  # Importing MF coordinates
nMF = len(MF_coordinates)
MF_GL = np.empty(shape=(nMF, 3))
MF_GL_Id = np.empty(shape=(nMF, 3))

##Searching parameters##
cellsize = 20/np.sqrt(3)
rows = int(np.ceil(Horizontal_range / cellsize))
cols = int(np.ceil(Transverse_range / cellsize))
third = int(np.ceil(Vertical_range / cellsize))
P = np.zeros((rows, cols, third), dtype=np.float32)
M = np.zeros((rows, cols, third), dtype=bool)

#Define Searching Area
N = {}
for i in range(rows):
    for j in range(cols):
        for u in range(third):
            N[(i, j, u)] = neighborhood(M.shape, (i, j, u), 3)

#Generate Poisson sampling points
points = Bridson_sampling((Horizontal_range, Transverse_range, Vertical_range), 20, nMF, False)
points[:, 1] = points[:, 1] * 3 # To make (20, 60)um of each GL

#Marking the GL point cells
for k in range(0, nMF):
    add_point(points[k, :])

for k in range(0, 50):
    # numRosetteperpm = np.ceil(np.random.normal(numRosetteperMF_mu, 37) / numpmperMF_mu)
    # nGL = np.ceil((Box_Id_y / 1200) * numRosetteperpm)
    # nGL_hd = np.ceil(0.8 * nGL)
    # nGL_Id = nGL - nGL_hd
    # z_mid = np.random.randint(50, 150)
    # Ell_xo = MF_coordinates[k, 0]
    # Ell_yo = MF_coordinates[k, 1] + offset
    # GLcount = 0
    # hdcount = 0
    # Idcount = 0
    # GLcount1 = 1
    # MF_GL_temp = Bridson_sampling((Horizontal_range, Transverse_range, Vertical_range), 20, int(nGL), False, pts1)
    # MF_GL_temp[:, 1] = MF_GL_temp[:, 1] * 3 #To make 60um for the Long axis
    # MF_GL_select = np.empty(shape=(int(nGL), 3))
    # for j in range(0, int(nGL)):
    #     GLx = MF_GL_temp[j, 0]
    #     GLy = MF_GL_temp[j, 1]
    #     GLz = MF_GL_temp[j, 2]
    #     find = ifinsideElipse(Ell_xo, Ell_yo, Ell_zo, GLx, GLy, GLz)
    Ell_xo = MF_coordinates[k, 0]
    Ell_yo = MF_coordinates[k, 1] + offset
    Ell_zo = z_mid
    find = ifinsideElipse(Ell_xo, Ell_yo, Ell_zo)
        if find > 0:
            if P[I]




            if hdcount < nGL_hd:
                GLcount += 1
                hdcount += 1
                ax1.scatter(GLx, GLy, s=10, c = 'b')
                if GLx <= Transverse_range and GLx > 0 and GLy <= Horizontal_range and GLy > 0 and GLz <= Vertical_range:
                    # if inside volume record the coord
                    insvolc += 1
                    MF_GL_select[GLcount, 0] = GLx
                    MF_GL_select[GLcount, 1] = GLy
                    MF_GL_select[GLcount, 2] = GLz
                    GLcount1 += 1
                    print('i : {}, MF_GL(x): {}, MF_GL(y): {}, MF_GL(z): {}'.format(i, MF_GL_select[GLcount1 - 1, 0], MF_GL_select[GLcount1 - 1, 1], MF_GL_select[GLcount1 - 1, 2]))
                    #ax1.scatter(GLx, GLy, s=10, c='r')
                    ax1.scatter(GLx, GLy, s = 10, c = 'r')
        else:
            if Idcount < nGL_Id:
                GLcount += 1
                Idcount += 1
                ax1.scatter(GLx, GLy, s=15, marker="X", c='y')
                if GLx <= Transverse_range and GLx > 0 and GLy <= Horizontal_range and GLy > 0 and GLz <= Vertical_range:
                    insvolc += 1
                    MF_GL_select[GLcount1, 0] = GLx
                    MF_GL_select[GLcount1, 1] = GLy
                    MF_GL_select[GLcount1, 2] = GLz
                    GLcount1 += 1
                    print('i : {}, MF_GL(x): {}, MF_GL(y): {}, MF_GL(z): {}'.format(i, MF_GL_select[GLcount1 - 1, 0], MF_GL_select[GLcount1 - 1, 1], MF_GL_select[GLcount1 - 1, 2]))

                    ax1.scatter(GLx, GLy, s=10, c='black')
            #ax2.scatter(MF_GL[i, GLcount, 0], MF_GL[i, GLcount, 1], s=10, marker = "v" ,c = 'black')
                    #print(''%d %d %d %d\\n' % i, MF_GL(i, GLcount1 - 1, 1), MF_GL(i, GLcount1 - 1, 2), MF_GL(i, GLcount1 - 1, 3))
             # MF_GL_Id[Idcount, 0] = MF_GL
             # MF_GL_Id[Idcount, 1] = GLy
             # MF_GL_Id[Idcount, 2] = GLz
        MF_GL = np.stack(MF_GL_select, axis=1)
        pts1 = np.vstack((pts1, MF_GL_select[:, :]))

#ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# for i in range(0, nMF):
#     for j in range(0, GLcount1):
#         X = MF_GL[i, j, 0]
#         Y = MF_GL[i, j, 1]
#         Z = MF_GL[i, j, 2]
plt.show()
#ax1.scatter(X, Y, Z)
# plt.figure(1)
# plt.scatter(MF_GL)
# plt.scatter(MF_GL_Id[:, 0], MF_GL_Id[:, 1], MF_GL_Id[:, 2])

# del size
# d = np.load()  # 이것 어떻게 해야할까?
# size(d)
# len(np.unique(d[:, 1]))