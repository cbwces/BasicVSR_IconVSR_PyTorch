import logging
import torch
import cv2
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str

class VedioSet(torch.utils.data.Dataset):

    def __init__(self, video_file, window_size):
        video_seq = []
        vidcap = cv2.VideoCapture(video_file)
        success = True
        while success:
            success, image = vidcap.read()
            video_seq.append(image)
        self.video_seq_windows = self.roll_window(video_seq, window_size)

    def roll_window(self, video_seq, window_size):
        half_window_size = window_size // 2
        total_frames = len(video_seq)
        video_seq_windows = []
        for i in range(half_window_size, total_frames-half_window_size):
            video_seq_windows.append(video_seq[i-half_window_size:i+half_window_size])
        return video_seq_windows

    def __getitem__(self, index):
        self.video_seq_windows[index]

    def __len__(self, ):
        return len(self.video_seq_windows)


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_set = VedioSet(opt['vedio_list'])
    test_loader = torch.utils.data.DataLoader()
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'])


if __name__ == '__main__':
    main()
