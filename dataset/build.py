# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os

from .datasets import CAGClsDataset
from .pretrain_datasets import (  # noqa: F401
    DataAugmentation, CAGPretrainDataset
)


def build_pretraining_dataset(args):
    transform = DataAugmentation(args)
    dataset = CAGPretrainDataset(
        root=args.data_root,
        setting=args.data_path,
        name_pattern=args.fname_tmpl,
        new_length=args.num_frames,
        shift_frames=args.shift_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        lazy_init=False,
        num_sample=args.num_sample)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if is_train:
        mode = 'train'
        anno_path = os.path.join(args.data_path, 'train.csv')
    else:
        mode = 'validation'
        anno_path = os.path.join(args.data_path, 'val.csv')

    if args.data_set == 'CAG':
        dataset = CAGClsDataset(
            anno_path=anno_path,
            data_root=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            sparse_sample=False,
            args=args
        )
        nb_classes = 2
    else:
        raise NotImplementedError('Unsupported Dataset')

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
