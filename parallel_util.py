
'''
parallel_util.py

Contains functions that will be imported to each worker in the parallel version of pyBREP.
Queries given points in a given tree, saves results.
'''


def str_l(ar):
    '''make a space-seperated string from all elements in a 1D-array'''
    return (' '.join(str(ar[i]) for i in range(len(ar))))


def find_connections_2dpar(kdt, pts, lpts, c_rad, lin_axis, lin_in_tree, lin_is_src, ids, prefix, debut=False):
    ''' 
    Performs distance-based searches for the linearized version of pyBREP (currently Connect_2D)
    kdt: 2D Tree with points
    pts: Query_point object for the 3D data points
    lpts: Query_point object for the 2D projected points
    c_rad: critical radius, i.e. maximum distance between points to consider them connected
    lin_axis: axis along which the data is projected
    lin_in_tree: whether the linear structure is in the tree or not
    lin_is_src: whether the linear structure should be saved in the source file
    ids: IDs of the query points that this particular worker should deal with
    prefix: Path and prefix of the files under which the worker should save the data
    '''

    import numpy

    if lin_in_tree:
        q_pts = pts.coo[:, numpy.invert(lin_axis)]
    else:
        q_pts = lpts.coo[:, 0, numpy.invert(lin_axis)]

    lax_c = pts.coo[:, lin_axis]
    if lin_in_tree:
        lax_c = lax_c[ids[1]] # gives correct coordinates via correct indexing
    lax_range = lpts.coo[:, :, lin_axis]
    lax_range = lax_range.reshape((lax_range.shape[0], lax_range.shape[1]))

    res = []
    res_l = []

    #iterate through the query points
    for i, pt in enumerate(q_pts[ids[1]]):
        # find the points within the critical radius
        ind, = kdt.query_radius(numpy.expand_dims(pt, axis=0), r=c_rad)

        #check if the found points match along the linearized axis and if so, add distance from the beginning of the linearized axis
        if lin_in_tree:
            ind = ind[numpy.logical_and(lax_range[ind,0]<=lax_c[i], lax_range[ind,1]>= lax_c[i])]
            res_l.append(abs(lax_c[i] - lax_range[ind,0] -lpts.set_0[ind]) + lpts.lin_offset[ind])
        else:
            ind = ind[numpy.logical_and(lax_range[i,0]<=lax_c[ind], lax_range[i,1]>= lax_c[ind]).ravel()]
            res_l.append(abs(lax_c[ind].ravel() - lax_range[i,0] -lpts.set_0[i])+ lpts.lin_offset[i])

        res.append(ind.astype('int'))

    prefix  = str(prefix)
    fn_tar  = prefix + 'targets'   + str(ids[0]) + '.dat'
    fn_src  = prefix + 'sources'   + str(ids[0]) + '.dat'
    fn_segs = prefix + 'segments'  + str(ids[0]) + '.dat'
    fn_dis  = prefix + 'distances' + str(ids[0]) + '.dat'

    if debug:
        fn_coords = prefix + 'coords' + str(ids[0])+'.dat'
        f_coords = open(fn_coords, 'w')

    with open(fn_tar, 'w') as f_tar, open(fn_src, 'w') as f_src, open(fn_dis, 'w') as f_dis, open(fn_segs, 'w') as f_segs:

        for (l, cl, cl_l) in zip(list(ids[1]), res, res_l):

            assert len(cl) == len(cl_l), 'Something went wrong, all corresponding lists in your input arguments should have the same length'

            if len(cl_l)>0:
                f_dis.write("\n".join(map(str, cl_l)))
                #first, get the cell IDS of the query and tree points (for the linear points, that is just the point ID,
                #for the other points this information has to be extracted from the corresponding Query_points object.
                #Segments also corresponds to the 3D point population, right value is acquired from Query-points object.
                if lin_in_tree:
                    s_ar = pts.seg[l,:].astype('int')
                    f_segs.write("\n".join(map(str_l, [s_ar for i in range(len(cl))])))#*numpy.ones((len(cl), len (s_ar))))))

                    q_id = (numpy.ones(len(cl))*pts.idx[l]).astype('int')
                    tr_id = cl
                else:
                    f_segs.write("\n".join(map(str_l, pts.seg[cl].astype('int'))))
                    q_id = pts.idx[cl].ravel()
                    tr_id = (numpy.ones(len(cl))*l).astype('int')

                #depending on which population should be source and which should be target, save cell IDs accordingly.
                #if l == 1: print (lin_in_tree, lin_is_src)
                if lin_is_src:
                    f_src.write("\n".join(map(str, tr_id)))
                    f_tar.write("\n".join(map(str, q_id)))
                else:
                    f_src.write("\n".join(map(str, q_id)))
                    f_tar.write("\n".join(map(str, tr_id)))

                if debug:
                    f_coords.write((' '.join(map(str, pts.coo[l]))+'\n')*len(cl))

                #need to attach one more line here or we get two elements per line
                f_dis.write("\n")
                f_src.write("\n")
                f_tar.write("\n")
                f_segs.write("\n")

    if debug:
        f_coords.close()

    return [ids[0], res, res_l]


def find_connections_3dpar(kdt, spts, tpts, c_rad,  src_in_tree, ids, prefix):
    ''' 
    Performs distance-based searches between two populations of 3 dimensional point cloud structures
    spts: Query_point object for the source structure
    tpts: Query_point object for the target structure
    c_rad: critical radius, i.e. maximum distance between points to consider them connected
    lin_axis: axis along which the data is projected
    src_in_tree: whether the source population is in the tree or not
    ids: IDs of the query points that this particular worker should deal with
    prefix: Path and prefix of the files under which the worker should save the data
    '''
    import numpy

    if src_in_tree:
        q_pts = tpts.coo
    else:
        q_pts = spts.coo

    res = []

    # iterate through the query points
    for i, pt in enumerate(q_pts[ids[1]]):
        # find the points within the critical radius
        ind, = kdt.query_radius(numpy.expand_dims(pt, axis=0), r=c_rad)
        res.append(ind.astype('int'))

    prefix = str(prefix)

    fn_tar = prefix + 'target' + str(ids[0]) + '.dat'
    fn_src = prefix + 'source' + str(ids[0]) + '.dat'
    fn_sseg = prefix + 'source_segments' + str(ids[0]) + '.dat'
    fn_tseg = prefix + 'target_segments' + str(ids[0]) + '.dat'

    with open(fn_tar, 'w') as f_tar, open(fn_src, 'w') as f_src, open(fn_sseg, 'w') as f_sseg, open(fn_tseg, 'w') as f_tseg:

        for (l, cl) in zip(list(ids[1].astype('int')), res):

            if len(cl)>0:
                #depending on which population should be source and which should be target, save cell IDs accordingly.
                if src_in_tree:
                    f_tar.write("\n".join(map(str, numpy.ones(len(cl)) * tpts.idx[l])))
                    f_src.write("\n".join(map(str, spts.idx[cl.astype('int')])))

                    f_tseg.write("\n".join(map(str_l, [tpts.seg[l,:].astype('int') for i in range(len(cl))] )))
                    f_sseg.write("\n".join(map(str_l, spts.seg[cl.astype('int')].astype('int'))))
                else:
                    f_tar.write("\n".join(map(str, tpts.idx[cl])))
                    f_src.write("\n".join(map(str, numpy.ones(len(cl)) * spts.idx[l])))

                    f_tseg.write("\n".join(map(str, tpts.seg[cl.astype('int')])))
                    f_sseg.write("\n".join(map(str, [spts.seg[l].astype('int') for i in range(len(cl))] )))
                #need to attach one more line here or we get two elements per line
                f_dis.write("\n")
                f_src.write("\n")
                f_tar.write("\n")
                f_segs.write("\n")

    return [ids[0], res]
