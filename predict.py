import argparse
import time
from typing import Dict
from unidecode import unidecode

import cv2
import numpy as np
import os
import torch
from charnet.config import cfg
from charnet.modeling.model import CharNet
from charnet.modeling.preprocess import get_all_preprocesses
from colorama import Fore, Style, init

init(convert=True)
CUDA = torch.cuda.is_available()
# CUDA = False


def save_word_recognition(word_instances, image_id, save_root, separator=chr(31)):
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, '{}.txt'.format(image_id)), 'wt') as fw:
        for word_ins in word_instances:
            if len(word_ins.text) > 0:
                fw.write(separator.join([str(_) for _ in word_ins.word_bbox.astype(np.int32).flat]))
                fw.write(separator)
                fw.write(word_ins.text)
                fw.write('\n')


def resize(im, size):
    h, w, _ = im.shape
    scale = max(h, w) / float(size)
    image_resize_height = int(round(h / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    image_resize_width = int(round(w / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height), interpolation=cv2.INTER_LINEAR)
    return im, scale_w, scale_h, w, h


def vis(img, word_instances):
    img_word_ins = img.copy()
    for word_ins in word_instances:
        word_bbox = word_ins.word_bbox
        img_word_ins = cv2.polylines(img_word_ins, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)],
                                     True, (0, 255, 0), 2)
        img_word_ins = cv2.putText(
            img_word_ins,
            '{}'.format(word_ins.text),
            (int(word_bbox[0]), int(word_bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    return img_word_ins


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('config_file', help='path to config file', type=str)
    parser.add_argument('image_dir', type=str)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('postfix', nargs='?', type=str, const='', default='')

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    print(cfg)
    pre_dict: Dict = get_all_preprocesses()
    preprocess = pre_dict[cfg.PREPROCESS_TYPE]

    charnet = CharNet()
    charnet.to(device=torch.device('cuda') if CUDA else torch.device('cpu'))
    if not CUDA:
        torch.set_num_threads(18)
    charnet.load_state_dict(torch.load(cfg.WEIGHT))
    charnet.eval()
    print(f'CUDA: {CUDA}\n')

    pics = ['jpg', 'png', 'bmp']
    files = [f for f in sorted(os.listdir(args.image_dir)) if f.split('.')[-1] in pics]
    for im_name in files:
        start_time = time.time()
        print(f"Processing {Fore.GREEN}{im_name}{Style.RESET_ALL}...")
        im_file = os.path.join(args.image_dir, im_name)

        im_original = preprocess.run(im_file)
        im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
        with torch.no_grad():
            char_bboxes, char_scores, word_instances, char_instances = charnet(im, scale_w, scale_h, original_w, original_h)
        print('Saving...')
        t = time.time()
        save_word_recognition(
            word_instances, os.path.splitext(im_name)[0],
            args.results_dir, cfg.RESULTS_SEPARATOR
        )

        vis_img = vis(im_original, word_instances)
        cv2.imwrite(
            os.path.join(
                args.results_dir,
                f'{os.path.splitext(unidecode(im_name))[0]}{args.postfix}.png'
            ), vis_img)
        if char_instances:
            os.makedirs('chars', exist_ok=True)
            save_word_recognition(
               char_instances, f'{os.path.splitext(im_name)[0]}_chars',
               f'{args.results_dir}\\chars', cfg.RESULTS_SEPARATOR
            )
            vis_img = vis(im_original, char_instances)
            cv2.imwrite(
                os.path.join(
                    f'{args.results_dir}\\chars',
                    f'{os.path.splitext(unidecode(im_name))[0]}{args.postfix}_chars.png'
                ), vis_img)

        print(f'\ttime: {time.time() - t}s')
        print(f'{Fore.GREEN}Total time: {round(time.time() - start_time, 2)}s{Style.RESET_ALL}\n')
