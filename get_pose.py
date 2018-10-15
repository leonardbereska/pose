import sys
sys.path.append('/export/home/lbereska/openpose_gpu/build/python/openpose')
from openpose import *
from os import path as osp
import glob
import cv2
import argparse
import os


def mkdir_if_missing(name):
    if not osp.exists(name):
        os.mkdir(name)
        return True
    else:
        return False


def main(args):
    """
    Calculating KP, saving images and keypoints
    """
    path_to_data = args.path
    assert osp.exists(path_to_data)
    print('Reading data path {}'.format(path_to_data))
    saved_ = '/export/home/lbereska/saved'

    # make save directory
    mkdir_if_missing(saved_)
    assert osp.exists(saved_)
    for i in range(10):
        save_ = osp.join(saved_,'save_img{}'.format(i))
        if mkdir_if_missing(save_):
            save_dir = save_
            save_file = osp.join(saved_, 'save_dict{}.npy'.format(i))
            print('making dir: {}'.format(save_dir))
            break
        else:
            print('dir {} exists'.format(save_))
            continue


    # initialize openpose
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    params["num_gpu_start"] = args.gpu
    params["disable_blending"] = False
    params["default_model_folder"] = "../../../models/"
    openpose = OpenPose(params)
    print('Initialized OpenPose')
    kp_dict = {}
    imgs = sorted(glob.glob(path_to_data +  '/*'))
    for it, img_path in enumerate(imgs):
        img = cv2.imread(img_path)
        img_id = img_path.split('/')[-1].split('.')[0]
        arr, output_image = openpose.forward(img, True)
        if args.save_img:
            save_path = osp.join(save_dir, '{}.png'.format(img_id))
            cv2.imwrite(save_path, output_image)
        kp_dict[img_id] = arr
        if it % 100 == 0:
            print('{} / {}'.format(it, len(imgs)))
    np.save(save_file, kp_dict)
    print('Saved keypoints in dictionary')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="get pose")
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_img', action='store_true')
    main(parser.parse_args())
