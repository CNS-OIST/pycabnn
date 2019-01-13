x_r = np.array([0.0, 1500.0])
y_r = np.array([0.0, 750.0])
z_r = [0.0, 1000.0]
rrs64 = [x_r/8, y_r/8, z_r]
rrs16 = [x_r/4, y_r/4, z_r]
rrs4 = [x_r/2, y_r/2, z_r]

p2 = './input_data/subsampled/'

fi_go = './input_data/GoCcoordinates.sorted.dat'
fi_gr = './input_data/GCcoordinates.sorted.dat'

fo_go64 = p2+'GoCcoordinates_64.dat'
fo_gr64 = p2+'GCcoordinates_64.dat'
fo_go16 = p2+'GoCcoordinates_16.dat'
fo_gr16 = p2+'GCcoordinates_16.dat'
fo_go4 = p2+'GoCcoordinates_4.dat'
fo_gr4 = p2+'GCcoordinates_4.dat' 

def subsample_coords (rrs, fn_in, fn_out = 'input_files/downsampled.dat', save = True):
    res = []
    rnr = [0, 0]
    with open(fn_in, newline = '') as f, open (fn_out, 'w', newline = '') as w_f:
        rr = csv.reader(f, delimiter = ' ')
        if save: wr = csv.writer(w_f, delimiter = ' ')
        for line in rr:
            in_range = all([float(line[i])>rrs[i][0] and float(line[i])<rrs[i][1] for i in range(len(rrs))]) #check if in range
            if in_range: 
                if save: wr.writerow([float(line[j]) for j in range(len(rrs))])
                res.append([float(line[j]) for j in range(len(rrs))])
                rnr[0] = rnr[0]+1
            else:
                rnr[1] = rnr[1]+1
    print ('Subsampled {} of {}'.format(rnr[0], sum(rnr)))
    return res

_ = subsample_coords (rrs64, fi_go, fo_go64)
_ = subsample_coords (rrs64, fi_gr, fo_gr64)
_ = subsample_coords (rrs16, fi_go, fo_go16)
_ = subsample_coords (rrs16, fi_gr, fo_gr16)
_ = subsample_coords (rrs4, fi_go, fo_go4)
_ = subsample_coords (rrs4, fi_gr, fo_gr4)
