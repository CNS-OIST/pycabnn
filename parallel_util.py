def pts_in_tr_ids (kdt, q_pts, lax_c, lax_range, c_rad, ids, lin_in_tree):
    
    import numpy

    res = []
    l_res = []

    for i, pt in enumerate(q_pts[ids]): #iterate through the query points
        # find the points within the critical radius
        ind, = kdt.query_radius(numpy.expand_dims(pt, axis = 0), r = c_rad)

        #check if the found points match along the linearized axis and if so, add distance from the beginning of the linearized axis
        if lin_in_tree: 
            ind = ind[numpy.logical_and(lax_range[ind,0]<=lax_c[i], lax_range[ind,1]>= lax_c[i])]
            l_res.append(lax_c[i] - lax_range[ind,0])
        else:
            ind = ind[numpy.logical_and(lax_range[i,0]<=lax_c[ind], lax_range[i,1]>= lax_c[ind]).ravel()]
            l_res.append(lax_c[ind] - lax_range[i,0])

        res.append(ind.astype('int'))

    return [ids[0], res, l_res]
