# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot

def run(model, output_folder, img_files, brightness, opacity, show, show_wait_time):
    output_folder = Path(output_folder, f"preds_brightened_{brightness:03f}")
    # start looping
    for img_file in img_files:
        # test a single image
        frame = cv2.imread(str(img_file))
        frame = (frame * brightness).astype(np.uint8)

        result = inference_model(model, frame)

        if output_folder is None:
            out_file = None
        else:
            out_file = str(Path(output_folder, img_file.name))
            os.makedirs(output_folder, exist_ok=True)

        # blend raw image and prediction
        draw_img = show_result_pyplot(model, np.flip(frame, axis=2), result, show=False, opacity=opacity)

        if show:
            cv2.imshow('video_demo', draw_img)
            cv2.waitKey(show_wait_time)

        if out_file is not None:
            cv2.imwrite(str(out_file), np.flip(draw_img, axis=2))

def main():
    parser = ArgumentParser()
    parser.add_argument('folder', help='the folder of images to read from')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument("--n-images", type=int, default=50, help="How many random images to run from the folder")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--output-folder', default=None, type=str, help='Where to write the files')
    parser.add_argument(
        '--title', default="", type=str, help='Title to add to plot')
    parser.add_argument("--brightnesses", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    assert args.show or args.output_folder, \
        'At least one output should be enabled.'

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    img_files = list(Path(args.folder).glob("*"))
    img_files = np.random.choice(img_files, args.n_images)
    for brightness in args.brightnesses:
        run(model=model,
             output_folder=args.output_folder,
             img_files=img_files,
             brightness=brightness,
             opacity=args.opacity,
             show=args.show,
             show_wait_time=2)

if __name__ == '__main__':
    main()
