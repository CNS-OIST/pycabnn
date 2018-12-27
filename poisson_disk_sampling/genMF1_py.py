import numpy as np
import matplotlib.pyplot as plt
from GL_poisson import Bridson_sampling

s = np.random.seed(1) # 1) 왜 seed를 고정하는지 질문하기

Longaxis = 1500  # 185%eval (readParameters ('GoCxrange', 'Parameters.hoc'))  % um
Shortaxis = 700  # 185%eval (readParameters ('GoCxrange', 'Parameters.hoc'))  % um
MFdensity = 1650  # 5000;%cells/mm2%190  추가확인필요

#fid = np.loadtxt('datasp.dat', delimiter = '\t')   #왜 필요한지? (아직은 필요x)
box_fac = 2.5
Xinstantiate = 64 + 40  # 297+40
Yinstantiate = 84 + 40 * box_fac  # 474+40x*box_fac
numMF = (Longaxis + (2 * Xinstantiate)) * (Shortaxis + (2 * Yinstantiate)) * MFdensity * 1e-6
plotMF = 1
fcoor = np.loadtxt('MFCr.dat')
dt = 0.025

## Generate MF Coordinates ##
pts = Bridson_sampling((Longaxis, Shortaxis), 8, int(numMF), True)
MF_coordinates = np.empty(shape=(int(numMF), 2))
MF_coordinates[:, 0] = pts[:, 0]
MF_coordinates[:, 1] = pts[:, 1]

# plt.figure(2)
# plt.scatter(pts[:, 0], pts[:, 1])
# for i in range(0, int(numMF)):
#     MF_coordinates[i, 0] = np.random.randint(0 - Xinstantiate, Longaxis + Xinstantiate)
#     MF_coordinates[i, 1] = np.random.randint(0 - Yinstantiate, Shortaxis + Yinstantiate)
#     print('{}, {}' .format(MF_coordinates[i, 0], MF_coordinates[i, 1]))


    # fcoor.close()
#plt.scatter(MF_coordinates[:, 0], MF_coordinates[:, 1])
#plt.show()

centrex = Longaxis / 2
centrey = Shortaxis / 2
radius = 100
finalMF = []
finalMF = (MF_coordinates[:, 0] - centrex) ** 2 + (MF_coordinates[:, 1] - centrey) ** 2
finalMF[finalMF <= radius ** 2] = 1 #activated
finalMF[finalMF > radius ** 2] = 0  #inactivated

    # Second spatial kernal 1250, 350
finalMF1 = []
centrex1 = Longaxis / 2
centrey1 = Shortaxis / 2
finalMF1 = (MF_coordinates[:, 0] - centrex1) ** 2 + (MF_coordinates[:, 1] - centrey1) ** 2
finalMF1[finalMF1 <= radius ** 2] = 1 #activated
finalMF1[finalMF1 > radius ** 2] = 0 #inactivated
find_ac1 = np.where(finalMF1)
finalMF1[find_ac1] = 1

finalMF2 = []
centrex2 = Longaxis / 2
centrey2 = Shortaxis / 2
finalMF2 = (MF_coordinates[:, 0] - centrex2) ** 2 + (MF_coordinates[:, 1] - centrey2) ** 2
finalMF2[finalMF2 <= radius * radius] = 1
finalMF2[finalMF2 > radius * radius] = 0
find_ac2 = np.where(finalMF2)
finalMF2[find_ac2] = 1

# Plot activatd and inactivated MF
if plotMF ==1:
    for i in range(0, int(numMF)):
        if finalMF[i] ==1:
            plt.figure(1)
            plt.scatter(MF_coordinates[i, 0], MF_coordinates[i, 1], c = 'r')
            plt.xlabel('long axis um')
            plt.ylabel('short axis um')
        else:
            plt.scatter(MF_coordinates[i, 0], MF_coordinates[i, 1], c='blue')

plt.show()
