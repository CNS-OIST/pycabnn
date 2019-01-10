def str_l(ar):
    '''make a space-seperated string from all elements in a 1D-array'''
    return(' '.join(str(ar[i]) for i in range(len(ar))))


def print_time_and_reset(t, comment='Finished block after '):
    import time
    t_new = time.time()
    print(comment, t_new-t)
    return t_new

class Pseudo_hoc(object):
    '''Up to now, the program depends on a hoc object that contains the parameters for the simulation.
    However, as in the cluster, neuron is not installed for python 3, this class is a workaround:
    First, a dict has to be generated(and probably pickled) from the hoc file in a python distribution that has neuron installed.
    This dict(or the file containing it) is then read in in this class, and an empty object with no other functionalities gets
    assigned all the parameters from the dict as attributes. The resulting object can then be used as a parameter carrier just as the hoc file.'''

    def __init__(self, ad_or_fn=None):
        '''Add parameters from dict or filename to pseudo_hoc object'''
        # Make a pseudo-hoc object
        if ad_or_fn is None:
            return
        elif type(ad_or_fn) != dict:
            try:
                ad_or_fn = Path(ad_or_fn)
                #print (ad_or_fn.exists())
                #print (ad_or_fn.is_file())
                #print (ad_or_fn.absolute())

                import pickle
                with ad_or_fn.open('rb') as f_in:
                    ad_or_fn = pickle.load(f_in)
            except:
                print('Tried to read in ', ad_or_fn, ' as a file, failed')
                return
        else:
            assert type(ad_or_fn) == dict, 'Could not read in {}'.format(ad_or_fn)
        # Add all elements from the read in file as arguments to the pseudo-hoc object
        for k, v in ad_or_fn.items():
            #As for the pickling process, all values had to be declared as strings, try to convert them back to a number
            try:
                v = float(v)
            except:
                pass
            try:
                setattr(self, k, v)
            except:
                pass

    def convert_hoc_to_pickle(self, config_fn, output_fn = 'pseudo_hoc.pkl'):
        '''Take a .hoc config file and pickle it as a neuron-independent python dict.'''
        try:
            import neuron
        except ModuleNotFoundError:
            print('Could not import neuron, go to a python environment with an installed neuron version and try again.')
            return
        neuron.h.xopen(config_fn)
        d = dir(h)
        h_dict = dict()
        #Transfer parameters from the hoc object to a python dictionary
        for n, el in enumerate(d):
            if el[0].isupper() or el[0].islower():
                try:
                    #The value has to be converted to its string representation to get rid of the hoc properties.
                    #Must be kept in mind when reading in though.
                    h_dict[el] = repr(getattr(h,el))
                except:
                    pass
        # Dump the dictionary
        import pickle
        with output_fn.open('wb') as f:
            pickle.dump(h_dict, f)
