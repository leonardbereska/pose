# sort by visible keypoints
# input: dictionary of pose estimations
# output: list of lists of image ids/paths sorted by num kp visible

import argparse
import numpy as np
import os

saved_ = '/export/home/lbereska/saved'
assert os.path.exists(saved_)
file_dir = os.path.join(saved_, 'kp_visible')
if not os.path.exists(file_dir):
    os.mkdir(file_dir)
N_KP = 25

def get_filename(n_kp):
    return os.path.join(file_dir, str(n_kp).zfill(2) + '.txt')


def write_vislist(list_, v):
    for n_kp in range(N_KP+1):
        print('visible kp {} has {} elements'.format(n_kp, len(list_[n_kp])))
        file_name = get_filename(n_kp)
        list_imgids = list_[n_kp]
        if v:
            print('writing {} image ids to {}'.format(len(list_imgids), file_name))
        with open(file_name, 'w+') as f:
            for imgid in list_imgids:
                f.write(str(imgid)+'\n')


def read_vislist(n_kp):
    file_name = get_filename(n_kp)
    with open(file_name, 'r') as f:
        imgids = f.read()  # TODO
    return imgids


def main(args):
    dict_ =  np.load("{}/{}.npy".format(saved_, args.kp_dict)).item()
    keys = dict_.keys()
    vis_list = [[] for i in range(N_KP+1)]  # list position is n kp visible
    for it, key in enumerate(keys):
        try:
            visible = dict_[key][0][:, 2] != 0.
        except:
            print('failed to read kp for {} at {}'.format(key, it))
            continue
        n_vis = np.sum(visible)
        # print(n_vis)
        img_id = key
        # print(img_id)
        # print(vis_list)
        vis_list[n_vis].append(img_id)
        # if it ==100:
            # print(vis_list)
    write_vislist(vis_list, args.verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="get pose")
    parser.add_argument('--kp_dict', type=str)
    parser.add_argument('--verbose', action='store_true')
    main(parser.parse_args())
