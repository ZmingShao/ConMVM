# pylint: disable=line-too-long,too-many-lines,missing-docstring
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import video_transforms, volume_transforms


class CAGClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self,
                 anno_path,
                 data_root='',
                 mode='train',
                 clip_len=8,
                 frame_sample_rate=2,
                 crop_size=224,
                 short_side_size=256,
                 new_height=256,
                 new_width=340,
                 keep_aspect_ratio=True,
                 num_segment=1,
                 num_crop=1,
                 test_num_segment=10,
                 test_num_crop=3,
                 sparse_sample=False,
                 args=None):
        self.anno_path = anno_path
        self.data_root = data_root
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.sparse_sample = sparse_sample
        self.args = args
        self.aug = False
        self.rand_erase = False

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        cleaned = pd.read_csv(self.anno_path, delimiter='\t')
        self.dataset_samples = list(
            cleaned.apply(lambda row: os.path.join(self.data_root, row['folder'], row['video']), axis=1))
        self.label_array = list(cleaned['label'])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(
                    self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(
                    size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            # T H W C
            buffer = self.load_video(sample, sample_rate_scale=scale_t)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during training".format(
                            sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.load_video(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during validation".
                        format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split(
                "/")[-1].split(".")[0]

        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.Compose([
            video_transforms.Resize(
                self.short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(
                size=(self.crop_size, self.crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        buffer = aug_transform(buffer)

        return buffer

    def load_video(self, sample, sample_rate_scale=1):
        fname = sample

        try:
            import pydicom
            video_array = pydicom.dcmread(fname).pixel_array
            if video_array.dtype != np.uint8:
                video_array = (
                    (video_array - video_array.min()) / (video_array.max() - video_array.min()) * 255
                ).astype(np.uint8)
            assert video_array.ndim == 3
            video_array = np.stack([video_array,]*3, axis=-1)
        except Exception as e:
            print(f"Failed to load video from {fname} with error {e}!")
            return []

        length = video_array.shape[0]

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = length // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(
                    0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate(
                    (index,
                     np.ones(self.clip_len - seg_len // self.frame_sample_rate)
                     * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                if self.mode == 'validation':
                    end_idx = (converted_len + seg_len) // 2
                else:
                    end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        buffer = video_array[all_index]
        return buffer

    def __len__(self):
        return len(self.dataset_samples)
