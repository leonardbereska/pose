import argparse
import numpy as np
from os import path as osp
import os
import cv2
import matplotlib as mp
mp.use('Agg')
from matplotlib import pyplot as plt
saved_ = '/export/home/lbereska/saved'

def euklid_norm(v_):
     """
     vector: (x, y):sqrt(x^2 + y^2)
     """
     v_ = np.power(v_, 2)
     v_ = np.sum(v_, axis=1)  # x^2 + y^2
     v_ = np.sqrt(v_)
     v_ = np.expand_dims(v_, axis=1)
     return v_

def calc_pck(v_):
    # print(v_.shape)
    v_ = np.tile(v_, [1, 40])
    pck_dist = np.arange(40, dtype=np.int32)
    # print(pck_dist.shape)
    pck_dist = np.expand_dims(pck_dist, axis=0)
    # print(pck_dist.shape)
    pck_dist = np.tile(pck_dist, [v_.shape[0], 1])
    # print(pck_dist.shape)
    pck_score = np.where(v_<pck_dist, np.ones_like(pck_dist), np.zeros_like(pck_dist))
    pck_mean_score = np.mean(pck_score, axis=0)
    return pck_mean_score



kp_dict25 = {
    0: "Nose",
    1: "Neck" ,
    2: "RShoulder" ,
    3: "RElbow" ,
    4: "RWrist" ,
    5: "LShoulder" ,
    6: "LElbow" ,
    7: "LWrist" ,
    8: "MidHip" ,
    9: "RHip" ,
    10: "RKnee" ,
    11: "RAnkle" ,
    12: "LHip" ,
    13: "LKnee" ,
    14: "LAnkle" ,
    15: "REye" ,
    16: "LEye" ,
    17: "REar" ,
    18: "LEar" ,
    19: "LBigToe" ,
    20: "LSmallToe" ,
    21: "LHeel" ,
    22: "RBigToe" ,
    23: "RSmallToe" ,
    24: "RHeel" ,
    25: "Background"}

kp_dict18 = {
    0: "Nose",
    1: "Neck" ,
    2: "RShoulder" ,
    3: "RElbow" ,
    4: "RWrist" ,
    5: "LShoulder" ,
    6: "LElbow" ,
    7: "LWrist" ,
    8: "RHip" ,
    9: "RKnee" ,
    10: "RAnkle" ,
    11: "LHip" ,
    12: "LKnee" ,
    13: "LAnkle" ,
    14: "REye" ,
    15: "LEye" ,
    16: "REar" ,
    17: "LEar"}


def save_image(img_path, img_dir):
    img = cv2.imread(img_path)
    img_name = img_path.split('/')[-1]
    img_save = osp.join(save_dir, img_name)
    cv2.imwrite(img_save, img)
    print('save fail at {}'.format(img_save))
    return img

def get_pose_idx(key_):
    # _, pose1 = split_id(key)
    # return pose1
    z1, z2 = key_.split('_')
    app_idx = int(z1) - 655000 # 100000
    pose_idx = int(z2) - 1 + app_idx
    return pose_idx


def split_id(app_pose):
    app_pose = app_pose.split('_')
    app = '_'.join(map(str, app_pose[0:2]))
    pose = '_'.join(map(str, app_pose[2:4]))
    return app, pose


def mirror(kp, n_kp):
    if n_kp == 25:
        idx_mirror = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
    elif n_kp == 18:
        idx_mirror = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17,
                      16]
    assert len(idx_mirror) == n_kp
    kp_new = np.empty(shape=kp.shape)
    for i in range(n_kp):
        kp_new[i, :] = kp[idx_mirror[i], :]
    assert kp_new.shape == kp.shape
    return kp_new


def body_to_coco(body25):
    idx_coco = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    coco = np.empty(shape=body25[0:18, :].shape)
    for i in range(18):
        coco[i, :] = body25[idx_coco[i]]
    assert coco.shape == body25[0:18, :].shape
    return coco


def main(args):
    print('Reading Dictionaries')
    dict1 = np.load("{}/{}.npy".format(saved_, args.dict_gen)).item()  # gen
    dict2 = np.load("{}/{}.npy".format(saved_, args.dict_gt)).item()  # gt

    all_dist = []
    keys1 = dict1.keys()
    keys2 = dict2.keys()

    keys1 = sorted(keys1)

    a_p_path = '/export/home/dlorenz/PycharmProjects/patrick'
    # pose_list = list(np.load("{}/{}.npy".format(a_p_path, 'pose_list')))
    # pose_list = list(np.load("{}/{}.npy".format(a_p_path, 'select_pose')))
    # pose_list = list(np.load("{}/{}.npy".format(a_p_path, 'pose_list')))

    pose_list = list(np.load("{}/{}.npy".format(a_p_path,
                                                args.pose_path)))
    pose_list = [str(p)for p in pose_list]
    print('len keys1 {}'.format(len(keys1)))
    print('len keys2 {}'.format(len(keys2)))
    print('len pose_list {}'.format(len(pose_list)))
    assert len(pose_list) == len(keys1)

    print('Calculating distances..')
    n_flip = 0
    n_total = 0
    img_gen = []
    for key in keys1:
        img_gen.append(dict1[key])

    print('len img_gen {}'.format(len(img_gen)))
    for i in range(len(img_gen)):
        kp_gen = img_gen[i]
        kp_gt_key = pose_list[i]
        kp_gt = dict2[kp_gt_key]

        if kp_gen.shape == (0, 0, 0):
            print('failed to find pose {} for {}'.format(kp_gen.shape, i))
            continue
        kp1 = kp_gen[0][:, 0:2]
        kp2 = kp_gt[0][:, 0:2]
        mask = kp_gt[0][:,2] == 0. # confidence == 0.
        mask_gen = kp_gen[0][:,2] == 0. # confidence == 0.
        mask = 1-(1-mask)*(1-mask_gen)
        mask = np.expand_dims(mask, axis=1)
        mask = np.tile(mask, (1, 2))  # for both x and y

        if args.n_kp == 18:
            kp1 = body_to_coco(kp1)
            kp2 = body_to_coco(kp2)
            mask = body_to_coco(mask)

        assert mask.shape == (args.n_kp, 2), '{} {}'.format(mask.shape, args.n_kp)
        kp2 = np.ma.masked_array(kp2, mask=mask)
        kp1 = np.ma.masked_array(kp1, mask=mask)
        n_total += 1
        assert(kp1.shape==(args.n_kp, 2))
        assert(kp2.shape==(args.n_kp, 2))
        dist1 = euklid_norm(kp1 - kp2)
        dist2 = euklid_norm(kp1 - mirror(kp2, args.n_kp))
        if args.pck:
            dist1 = calc_pck(dist1)
            dist2 = calc_pck(dist2)
            err1 = np.mean(dist1)
            err2 = np.mean(dist2)
        else:
            err1 = np.mean(dist1, axis=0)
            err2 = np.mean(dist2, axis=0)
        if args.flip:
            if args.pck:
                idx_min = np.argmax([err1, err2])
            else:
                idx_min = np.argmin([err1, err2]) # flip frontal-back
        else:
            idx_min = 0
        dist = [dist1, dist2][idx_min]
        if idx_min == 1:
            # print('flipping: {} < {}'.format(err2, err1))
            n_flip += 1
        # assert dist.shape == (args.n_kp, 1)

        if args.save_fails != '':
            fail_thresh = 20
            img_error = np.mean(dist)
            if img_error > fail_thresh:
                img_id = keys1[i]
                save_dir = osp.join(saved_, 'fails', args.save_fails)
                if not os.path.exists(save_dir)
                    os.makedirs(save_dir)
                img_path = osp.join(saved_, img_dir, img_id+'.png')
                assert osp.exists(img_path)
                image = save_image(img_path, save_dir)

        all_dist.append(dist)

    print('flipped {} of {}'.format(n_flip, n_total))
    print('Calculating moments..')
    mu = np.mean(all_dist, axis=0)
    median = np.median(all_dist, axis=0)
    sigma = np.sqrt(np.mean([np.power((x - mu), 2) for x in all_dist], axis=0))
    print('kp mu: \n{}'.format(mu))
    print('kp median: \n{}'.format(median))
    print('kp sigma: \n{}'.format(sigma))
    print('mean (over all kp) mu {}'.format(np.mean(mu, axis=0)))
    print('mean (over all kp) median {}'.format(np.mean(median, axis=0)))
    print('mean (over all kp) sigma {}'.format(np.mean(sigma, axis=0)))

    if args.plot:
        print('Plotting ..')
        dist = []
        save_path = osp.join(saved_, 'hist')
        if not osp.exists(save_path):
            os.mkdir(save_path)
        for i in range(args.n_kp):
            dist.append([x[i][0] for x in all_dist])
        for i in range(args.n_kp):
            plt.figure()
            plt.hist(dist[i])
            if args.n_kp == 18:
                plt.title('KP {}: {}, error mean {}, median {}'.format(i, kp_dict18[i], mu[i], median[i]))
            else:
                plt.title('KP {}: {}, error mean {}, median {}'.format(i, kp_dict25[i], mu[i], median[i]))
            plt.savefig(osp.join(save_path, 'kp{}.png'.format(i)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="get pose")
    parser.add_argument('--n_kp', type=int, default=18, choices=[18, 25])
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--pose_path', type=str)
    parser.add_argument('--save_fails', type=str, default='', help='img dir')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--pck', action='store_true')
    parser.add_argument('--dict_gen', type=str, default='save_dict1')
    parser.add_argument('--dict_gt', type=str, default='dict_gt')

    main(parser.parse_args())
