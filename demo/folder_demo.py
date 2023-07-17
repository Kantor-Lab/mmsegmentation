# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mmengine.model.utils import revert_sync_batchnorm
import torch

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot


FULL_PALETTE= np.array([[ 255, 224 ,128],  # Dry Grass, 0
            [ 255, 255 ,0  ],  # Green Grass (canopy), 1
            [ 255, 0   ,80 ],  # Dry Shrubs, 2
            [ 134, 112 ,45 ],  # Green Shrubs, 3
            [ 144, 255 ,0  ],  # Canopy, 4
            [ 199, 255 ,128],  # Wood Pieces, 5
            [ 255, 0   ,224],  # Litterfall (bare earth or fuel), 6
            [ 255, 194 ,0  ],  # Timber Litter, 7
            [ 95 , 134 ,45 ],  # Live Trunks, 8
            [ 111, 0   ,255],  # Bare Earth, 9
            [ 255, 128 ,239],  # People, 10
            [ 255, 128 ,167],  # Sky, 11
            [ 83 , 45  ,134],  # Blurry, 12
            [ 134, 45  ,83 ],  # Obstacle
            [ 134, 68  ,45 ]])  # Drones, 13

REMAP = np.array([
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
])

SMALL_PALETTE = np.array([[0, 0, 255], [0, 255, 0], [0, 0, 0], [255, 0, 255]])

def visualize_with_palette(index_image, palette, ignore_ind=255):
    """
    index_image : np.ndarray
        The predicted semantic map with indices. (H,W)
    palette : np.ndarray
        The colors for each index. (N classes,3)
    """
    h, w = index_image.shape
    index_image = index_image.flatten()

    dont_ignore = index_image != ignore_ind
    output = np.zeros((index_image.shape[0], 3))
    colored_image = palette[index_image[dont_ignore]]
    output[dont_ignore] = colored_image
    colored_image = np.reshape(output, (h, w, 3))
    return colored_image.astype(np.uint8)


def remap_classes_bool_indexing(
    input_classes: np.array, remap: np.array, background_value: int = 255
):
    """Change indices based on input

    https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    """
    input_classes_numpy =input_classes.pred_sem_seg.to_tensor().numpy().values()[0]
    output = np.ones_like(input_classes_numpy) * background_value
    for i, v in enumerate(remap):
        mask = input_classes_numpy == i
        output[mask] = v
    return output

def run(model, output_folder, img_files, brightness, opacity, show,
        show_wait_time, palette, remap=None, prepend_string="", write_labels=False):
    output_folder = Path(output_folder, f"preds_brightened_{brightness:03f}")
    model.dataset_meta["palette"] = palette
    # start looping
    for img_file in img_files:
        # test a single image
        frame = cv2.imread(str(img_file))
        frame = (frame * brightness).astype(np.uint8)
        result = inference_model(model, frame)

        if remap is not None:
            result = remap_classes_bool_indexing(result, remap)

        if output_folder is None:
            out_file = None
        else:
            out_folder = str(Path(output_folder, img_file.stem))
            out_file = Path(out_folder, prepend_string+img_file.name)
            os.makedirs(out_folder, exist_ok=True)

        if write_labels:
            out_file_numpy = Path(out_file.parent, out_file.stem + ".npy")
            if remap is not None:
                labels = result[0]
            else:
                labels = result.pred_sem_seg.data.detach().cpu().numpy()[0]
            np.save(out_file_numpy, labels)
        else:
            # blend raw image and prediction
            if remap is None:
                draw_img = show_result_pyplot(model, np.flip(frame, axis=2), result, show=False, opacity=opacity)
            else:
                draw_img = visualize_with_palette(result[0], palette)
            # TODO could remap here

            if show:
                cv2.imshow('video_demo', draw_img)
                cv2.waitKey(show_wait_time)

            if out_file is not None:
                if remap is None:
                    cv2.imwrite(str(out_file), np.flip(draw_img, axis=2))
                else:
                    cv2.imwrite(str(out_file), draw_img)
                cv2.imwrite(str(out_file).replace(".png", "_img.png"), np.flip(frame, axis=2))

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
    parser.add_argument("--write-labels", action="store_true")
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
    img_files = np.random.choice(img_files, min(args.n_images, len(img_files)), replace=False)
    for brightness in args.brightnesses:
        run(model=model,
             output_folder=args.output_folder,
             img_files=img_files,
             brightness=brightness,
             opacity=args.opacity,
             show=args.show,
             palette=SMALL_PALETTE,
             show_wait_time=2,
             remap=REMAP,
             write_labels=args.write_labels,
             prepend_string="condensed_")
        run(model=model,
             output_folder=args.output_folder,
             img_files=img_files,
             brightness=brightness,
             opacity=args.opacity,
             show=args.show,
             palette=FULL_PALETTE,
             write_labels=args.write_labels,
             show_wait_time=2)

if __name__ == '__main__':
    main()
