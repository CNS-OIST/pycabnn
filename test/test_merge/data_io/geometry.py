import csv
import numpy as np

#read coordinates file
def read_coordinates(filename):
    import numpy as np
    data = []
    results = []
    for row in csv.reader(open(filename), delimiter=' '):
        data.append(row)
    for d in data:
        d = [float(i) for i in d]
        results.append(d)
    return np.array(results)


#return indices of the cells within the ON patch
def filter_patch(data, center, radius):
    i = 0
    center_x = center[0]
    center_y = center[1]
    indices = []
    for d in data:
       if (d[0] - center_x)**2 + (d[1] - center_y)**2 < radius**2:
           indices.append(i)
       i += 1
    return indices


#return indices of the cells within the ON beam
def on_beam(data,center,radius):
    i = 0
    center_y = center[1]
    indices = []
    for d in data:
       if d[1] <= center_y + radius and d[1] >= center_y - radius:
           indices.append(i)
       i += 1
    return indices


#return indices of the cells between the two ON patches
def between_patches(data,centers,radius):
    i = 0
    center1_x = centers[0][0]
    center1_y = centers[0][1]
    center2_x = centers[1][0]
    center2_y = centers[1][1]
    indices = []
    for d in data:
        #if the point is not within the patches
       if (d[0] - center1_x)**2 + (d[1] - center1_y)**2 >= radius**2:
           if (d[0] - center2_x)**2 + (d[1] - center2_y)**2 >= radius**2:
               #if the point is between the two centers
               if d[0] >= center1_x and d[0] <= center2_x:
                   indices.append(i)
       i += 1
    return indices
