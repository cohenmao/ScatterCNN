from scipy import io as sio
import os
import pickle

filters = {'data':[], 'scale':[]}
flt_dir = 'e:/maoz/filters/mat'
out_dir = 'e:/maoz/filters/pickle'
all_files = os.listdir(flt_dir)
for curr_file in all_files:
    curr_scale = int(curr_file[1])
    curr_flt = sio.loadmat(os.path.join(flt_dir, curr_file))
    filters['data'].append(curr_flt['curr_filter'])
    filters['scale'].append(curr_scale)
with open(out_dir + '/filters.pickle', "wb") as output_file:
    pickle.dump(filters, output_file)



