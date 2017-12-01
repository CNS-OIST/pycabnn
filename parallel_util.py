def pts_in_tr_ids (kdt, q_pts, c_rad, lax_c, lax_range,  ids, lin_in_tree, pts):
    
    import numpy

    res = []
    res_l = []

    for i, pt in enumerate(q_pts[ids[1]]): #iterate through the query points
        # find the points within the critical radius
        ind, = kdt.query_radius(numpy.expand_dims(pt, axis = 0), r = c_rad)

        #check if the found points match along the linearized axis and if so, add distance from the beginning of the linearized axis
        if lin_in_tree: 
            ind = ind[numpy.logical_and(lax_range[ind,0]<=lax_c[i], lax_range[ind,1]>= lax_c[i])]
            res_l.append(lax_c[i] - lax_range[ind,0])
        else:
            ind = ind[numpy.logical_and(lax_range[i,0]<=lax_c[ind], lax_range[i,1]>= lax_c[ind]).ravel()]
            res_l.append(lax_c[ind] - lax_range[i,0])

        res.append(ind.astype('int'))

    print (len(pts.seg))

    return [ids[0], res, res_l]



def pts_in_tr_ids2 (kdt, q_pts, c_rad, lax_c, lax_range,  ids, lin_in_tree, lin_is_src, prefix, pts):
    
    import numpy

    def str_l (ar): 
        '''make a space-seperated string from all elements in a 1D-array'''
        return (' '.join(str(ar[i]) for i in range(len(ar))))

    res = []
    res_l = []

    for i, pt in enumerate(q_pts[ids[1]]): #iterate through the query points
        # find the points within the critical radius
        ind, = kdt.query_radius(numpy.expand_dims(pt, axis = 0), r = c_rad)

        #check if the found points match along the linearized axis and if so, add distance from the beginning of the linearized axis
        if lin_in_tree: 
            ind = ind[numpy.logical_and(lax_range[ind,0]<=lax_c[i], lax_range[ind,1]>= lax_c[i])]
            res_l.append(lax_c[i] - lax_range[ind,0])
        else:
            ind = ind[numpy.logical_and(lax_range[i,0]<=lax_c[ind], lax_range[i,1]>= lax_c[ind]).ravel()]
            res_l.append(lax_c[ind] - lax_range[i,0])

        res.append(ind.astype('int'))

    print (len(pts.seg))


    fn_tar = prefix + 'target' + str(ids[0])+'.dat'
    fn_src = prefix + 'source' + str(ids[0])+'.dat'
    fn_segs = prefix +'segments'+ str(ids[0])+'.dat'
    fn_dis = prefix + 'distance'+ str(ids[0])+'.dat'

    with open (fn_tar, 'w') as f_tar, open (fn_src, 'w') as f_src, open (fn_dis, 'w') as f_dis, open (fn_segs, 'w') as f_segs: 

        for (l, cl, cl_l) in zip(list(ids[1]), res, res_l):
            
            assert len(cl) == len(cl_l), 'Something went wrong, all corresponding lists in your input arguments should have the same length'               

            if len(cl_l)>0:
                f_dis.write("\n".join(map(str, cl_l)))
                #first, get the cell IDS of the query and tree points (for the linear points, that is just the point ID, 
                #for the other points this information has to be extracted from the corresponding Query_points object.
                #Segments also corresponds to the 3D point population, right value is acquired from Query-points object.
                if lin_in_tree: 
                    s_ar = pts.seg[l,:].astype('int')
                    f_segs.write("\n".join(map(str_l, [s_ar for i in range (len(cl))])))#*numpy.ones((len(cl), len (s_ar)))))) 

                    q_id = numpy.ones(len(cl))*pts.idx[l]
                    tr_id = cl
                else:
                    f_segs.write("\n".join(map(str_l, pts.seg[cl].astype('int'))))
                    q_id = pts.idx[cl] 
                    tr_id = numpy.ones(len(cl))*l 

                #depending on which population should be source and which should be target, save cell IDs accordingly.
                if lin_in_tree == lin_is_src:
                    f_tar.write("\n".join(map(str, tr_id)))
                    f_src.write("\n".join(map(str, q_id)))
                else:
                    f_tar.write("\n".join(map(str, q_id)))
                    f_src.write("\n".join(map(str, tr_id )))

                #need to attach one more line here or we get two elements per line 
                f_dis.write("\n")
                f_src.write("\n")
                f_tar.write("\n")
                f_segs.write("\n")

    return [ids[0], res, res_l]





