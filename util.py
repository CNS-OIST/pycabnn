def str_l(ar):
    '''make a space-seperated string from all elements in a 1D-array'''
    return(' '.join(str(ar[i]) for i in range(len(ar))))


def print_time_and_reset(t, comment='Finished block after '):
    import time
    t_new = time.time()
    print(comment, t_new-t)
    return t_new
