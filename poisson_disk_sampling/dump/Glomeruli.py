import numpy as np
from GL_poisson import Bridson_sampling
from testpoisson import Bridson_sampling_2
from joblib import Parallel, delayed

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

def add_point_grc(p):
    i, j, u = int(p[0] / cellsize), int(p[1] / cellsize), int(p[2] / cellsize)
    Grc_P[i, j, u] = p

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

def in_neighborhood_grc(Ell_xo, Ell_yo, Ell_zo):
    i, j, u = int(Ell_xo / cellsize), int(Ell_yo / cellsize), int(Ell_zo / cellsize)
    for (i, j, u) in N[(i, j, u)]:
        return Grc_P[i, j, u]

def non_zero(p):
    if p[0] != 0 and p[1] != 0 and p[2] != 0:
        return True
    else:
        return False

def neighborhood_grc(shape, index, n):  # n이 7은 되어야 함. cuz 30/(6.72/sqrt(3)) = 7.732
    row_grc, col_grc, third_grc = index
    row0, row1 = max(row_grc - n, 0), min(row_grc + n + 1, shape[0])
    col0, col1 = max(col_grc - n, 0), min(col_grc + n + 1, shape[1])
    third0, third1 = max(third_grc - n, 0), min(third_grc + n + 1, shape[2])
    I = np.stack(np.mgrid[row0:row1, col0:col1, third0:third1], axis=3)
    I = I.reshape(I.size // 3, 3).tolist()
    return I

def ifinSphere(p, Sp_x, Sp_y, Sp_z):  # Grc dendrites rarely exceed 30 um
    ptin = (Sp_x - p[0]) ** 2 + (Sp_y - p[1]) ** 2 + (Sp_z - p[2]) ** 2
    if ptin <= 30 ** 2:
        return True
    else:
        return False

def connecting_MFGL(k):
    count = 0
    Ell_xo = MF_coordinates[k, 0]
    Ell_yo = MF_coordinates[k, 1] + offset
    Ell_zo = z_mid
    print('Complete making {}th mossy fiber coordination'.format(k))
    Exist_point = in_neighborhood(Ell_xo, Ell_yo, Ell_zo)
    point_num = np.size(Exist_point[:, 0])
    while point_num > 0:
        candidate = Exist_point[point_num, :]
        if non_zero(candidate) and ifinEplipse(candidate, Ell_xo, Ell_yo, Ell_zo) and in_limit(
                candidate):  # 범위에 안드는녀석은 포함시키면 안될듯? 이건 connection을 만드는 행위니까
            # if inside volume record the coor
            print('{}'.format(candidate))
            print('count number : {}'.format(count))
            MF_GL[k, count, :] = candidate
            print('k : {}, MF_GL(x): {}, MF_GL(y): {}, MF_GL(z): {}'.format(k, MF_GL[k, count, 0], MF_GL[k, count, 1], MF_GL[k, count, 2]))
            count += 1
        point_num -= 1

def connecting_GRCGL(i):
    count = 0
    grc_x = grc_points[i, 0]
    grc_y = grc_points[i, 1]
    grc_z = grc_points[i, 2]
    print('{}th granular cell coordination'.format(k))
    Exist_glo = in_neighborhood_grc(grc_x, grc_y, grc_z)
    glo_num = np.size(Exist_glo[:, 0])
    while glo_num > 0:
        grc_candidate = Exist_glo[glo_num, :]
        if non_zero(grc_candidate) and ifinSphere(grc_candidate, grc_x, grc_y, grc_z) and in_limit(grc_candidate):  # 범위에 안드는녀석은 포함시키면 안될듯? 이건 connection을 만드는 행위니까
            # if inside volume record the coor
            print('{}'.format(grc_candidate))
            print('count number : {}'.format(count))
            GRC_GL[i, count, :] = grc_candidate
            print('i : {}, GRC_GL(x): {}, GRC_GL(y): {}, GRC_GL(z): {}'.format(i, GRC_GL[i, count, 0], GRC_GL[i, count, 1], GRC_GL[i, count, 2]))
            count += 1
        glo_num -= 1
    print('Its done')

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

MF_coordinates = np.loadtxt('readMF.txt')  # Importing MF coordinates
nMF = len(MF_coordinates)
MF_GL = np.zeros((nMF, 20, 3))

cellsize = 6/np.sqrt(3)
rows = int(np.ceil(Horizontal_range / cellsize))
cols = int(np.ceil(Transverse_range / cellsize))
third = int(np.ceil(Vertical_range / cellsize))
P = np.zeros((rows, cols, third, 3), dtype=np.float32) #Glo coordinates
Grc_P = np.zeros((rows, cols, third, 3), dtype=np.float32) #Grc coordinates
M = np.zeros((rows, cols, third), dtype=bool)

N = {}
N = Parallel(n_jobs=-1)(delayed(neighborhood)(M.shape, (i, j, u), 7) for u in range(third) for j in range(cols) for i in range(rows))


points = Bridson_sampling((int(Horizontal_range/3), Transverse_range, Vertical_range), 6.614, 138600, True) #Glomeruli generation
points[:, 0] = points[:, 0]*3 #x-axis extension

for k in range(0, 18000):
    add_point(points[k, :])
    print('k is {}'.format(k))

Parallel(n_jobs=-1)(delayed(connecting_MFGL)(k) for k in range(int(nMF)-1))

del P[np.where(P==0)] # To delete meaningless data

grc_points = Bridson_sampling_2((Horizontal_range, Transverse_range, Vertical_range), 5, 415800, True, points) #Actually 399,000 but we can descard some grcs later.

for i in range(0, np.grc_points[:, 0]):
    add_point_grc(grc_points[i, :])
    print('i is {}'.format(i))

GRC_GL = np.zeros((np.size(grc_points[:, 0]), 10, 3))

Parallel(n_jobs=-1)(delayed(connecting_GRCGL)(i) for i in range(np.size(grc_points[:, 0])))
del Grc_P[np.where(Grc_P==0)] # To delete meaningless data

np.savetxt('Glocoordinates', points)
np.savetxt('Grccoordinates', grc_points)
np.savetxt('MF_GL', MF_GL)
np.savetxt('GRC_GL', GRC_GL)