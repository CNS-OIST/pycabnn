import numpy as np
import matplotlib.pyplot as plt
from GL_poisson import Bridson_sampling
from mpl_toolkits.mplot3d import Axes3D
s = np.random.seed(252)

Transverse_range = 1500  # x range
Horizontal_range = 700  # y range
Vertical_range = 200  # 140 for daria
z_mid = np.random.randint(50, 150)
semi_x = 1500 / 2 #64 / 2
semi_y = 700 / 2 #84 / 2
semi_z = 50 / 2 #50
insvolc = 0
box_fac = 2.5
Box_Id_x = 64 + 40  #무슨 의도인지 잘 모르겠음
Box_Id_y = 84 + 40 * box_fac
Box_Id_z = semi_z * 2
offset = 0
k = 0
numRosetteperMF_mu = 750
numRosetteperMF_sd = 37
numpmperMF_mu = 7
numpmperMF_sd = 1

def ifinsideElipse(Ell_xo, Ell_yo, Ell_zo, GLx, GLy, GLz):
    ptins = ((GLx - Ell_xo) ** 2 / (semi_x ** 2)) + ((GLy - Ell_yo) ** 2 / (semi_y ** 2)) + (
                (GLz - Ell_zo) ** 2 / (semi_z ** 2))
    if ptins <= 1:
        return True
    return False

#fileID = np.loadtxt('GLpoints.dat')
MF_coordinates = np.loadtxt('MFCr.dat')

Ell_zo = z_mid
nMF = len(MF_coordinates)
MF_GL = np.zeros(shape=(nMF, nMF, 3))
MF_GL_Id = np.empty(shape=(nMF, 3))
X=[]
Y=[]
Z=[]
#fig = plt.figure()
aig = plt.figure()
#ax1 = fig.add_subplot(1, 1, 1)
ax2 = aig.add_subplot(1, 1, 1)
for i in range(0, nMF):
    numRosetteperpm = np.ceil(np.random.normal(numRosetteperMF_mu, 37) / numpmperMF_mu)
    nGL = np.ceil((Box_Id_y / 1200) * numRosetteperpm)
    nGL_hd = np.ceil(0.8 * nGL)
    nGL_Id = nGL - nGL_hd
    z_mid = np.random.randint(50, 150)
    Ell_xo = MF_coordinates[i, 0]
    Ell_yo = MF_coordinates[i, 1] + offset
    GLcount = 0
    hdcount = 0
    Idcount = 0
    GLcount1 = 1
    while GLcount < nGL:
        k += 1
        GLx = np.random.randint(Ell_xo - np.ceil((Box_Id_x / 2)), Ell_xo + np.ceil((Box_Id_x / 2)))
        GLy = np.random.randint(Ell_yo - np.ceil((Box_Id_y / 2)), Ell_yo + np.ceil((Box_Id_y / 2)))
        GLz = np.random.randint(Ell_zo - semi_z, Ell_zo + semi_z)
        #find = 1  # 찾아봐야함
        find = ifinsideElipse(Ell_xo, Ell_yo, Ell_zo, GLx, GLy, GLz)
        if find > 0:
            if hdcount < nGL_hd:
                GLcount += 1
                hdcount += 1
                #ax1.scatter(GLx, GLy, s=10, c = 'b')
                if GLx <= Transverse_range and GLx > 0 and GLy <= Horizontal_range and GLy > 0 and GLz <= Vertical_range:
                    # if inside volume record the coord
                    insvolc += 1
                    MF_GL[i, GLcount, 0] = GLx
                    MF_GL[i, GLcount, 1] = GLy
                    MF_GL[i, GLcount, 2] = GLz
                    X.append(MF_GL[i, GLcount1, 0])
                    Y.append(MF_GL[i, GLcount1, 1])
                    Z.append(MF_GL[i, GLcount1, 2])
                    GLcount1 += 1
                    print('i : {}, MF_GL(x): {}, MF_GL(y): {}, MF_GL(z): {}'.format(i, MF_GL[i, GLcount1 - 1, 0], MF_GL[i, GLcount1 - 1, 1], MF_GL[i, GLcount1 - 1, 2]))
                    #ax1.scatter(GLx, GLy, s=10, c='r')
            ax2.scatter(MF_GL[i, GLcount, 0], MF_GL[i, GLcount, 1], s = 10, c = 'r')
        else:
            if Idcount < nGL_Id:
                GLcount += 1
                Idcount += 1
                #ax1.scatter(GLx, GLy, s=15, marker="X", c='y')
                if GLx <= Transverse_range and GLx > 0 and GLy <= Horizontal_range and GLy > 0 and GLz <= Vertical_range:
                    insvolc += 1
                    MF_GL[i, GLcount1, 0] = GLx
                    MF_GL[i, GLcount1, 1] = GLy
                    MF_GL[i, GLcount1, 2] = GLz
                    X.append(MF_GL[i, GLcount1, 0])
                    Y.append(MF_GL[i, GLcount1, 1])
                    Z.append(MF_GL[i, GLcount1, 2])
                    GLcount1 += 1
                    #ax1.scatter(GLx, GLy, s=10, c='black')
            ax2.scatter(MF_GL[i, GLcount, 0], MF_GL[i, GLcount, 1], s=10, marker = "v" ,c = 'black')
                    #print(''%d %d %d %d\\n' % i, MF_GL(i, GLcount1 - 1, 1), MF_GL(i, GLcount1 - 1, 2), MF_GL(i, GLcount1 - 1, 3))
             # MF_GL_Id[Idcount, 0] = MF_GL
             # MF_GL_Id[Idcount, 1] = GLy
             # MF_GL_Id[Idcount, 2] = GLz


#ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# for i in range(0, nMF):
#     for j in range(0, GLcount1):
#         X = MF_GL[i, j, 0]
#         Y = MF_GL[i, j, 1]
#         Z = MF_GL[i, j, 2]
#ax1.scatter(X, Y, Z)
# plt.figure(1)
# plt.scatter(MF_GL)
# plt.scatter(MF_GL_Id[:, 0], MF_GL_Id[:, 1], MF_GL_Id[:, 2])
plt.show()

# del size
# d = np.load()  # 이것 어떻게 해야할까?
# size(d)
# len(np.unique(d[:, 1]))