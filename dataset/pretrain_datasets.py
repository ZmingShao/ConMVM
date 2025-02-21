import os
import random
import traceback

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .loader import get_image_loader, get_video_loader
from .masking_generator import (
    RunningCellMaskingGenerator,
    TubeMaskingGenerator,
)
from .transforms import (
    GroupScale, 
    GroupNormalize,
    Stack,
    ToTorchFormatTensor,
)
from .video_transforms import (
    ShiftVideo, 
    RandomApply, 
    ColorJitter, 
    GaussianBlur, 
)


class DataAugmentation(object):

    def __init__(self, args):
        crop_size = args.input_size * 256 // 224 if args.shift_pixel else args.input_size
        self.num_frames = args.num_frames
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.transform_base = transforms.Compose([
            GroupScale(crop_size), 
        ])
        self.transform_final = transforms.Compose([
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(self.input_mean, self.input_std),
        ])
        
        self.shift = ShiftVideo(pixel=0, frame=0, size=args.input_size, clip_len=args.num_frames)
        
        shift_pixel = crop_size - args.input_size
        color_jitter = ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )
        self.transform_aug = transforms.Compose([
            ShiftVideo(
                pixel=shift_pixel, frame=0, size=args.input_size, clip_len=args.num_frames
            ),  # Note: frame shifting aug. is finally not implemented here
            RandomApply([color_jitter], prob=0.8),
            GaussianBlur(magnitude_range=(0.1, 2.0), magnitude_std='inf', prob=0.5),
        ])
        
        if args.mask_type == 'tube':
            self.encoder_mask_map_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio)
        else:
            raise NotImplementedError(
                'Unsupported encoder masking strategy type.')
        if args.decoder_mask_ratio > 0.:
            if args.decoder_mask_type == 'run_cell':
                self.decoder_mask_map_generator = RunningCellMaskingGenerator(
                    args.window_size, args.decoder_mask_ratio)
            else:
                raise NotImplementedError(
                    'Unsupported decoder masking strategy type.')

    def __call__(self, images, debug=False):
        src_data, _ = self.transform_base((images, None))
        if len(src_data) == self.num_frames:
            src_data, src_data_t = src_data, src_data.copy()
        elif len(src_data) == 2 * self.num_frames:  # for frame shifting aug.
            src_data, src_data_t = src_data[:self.num_frames], src_data[self.num_frames:]
        else:
            raise ValueError('Invalid number of frames.')
        
        if debug:
            process_data = self.shift(src_data)
            process_data_t = self.transform_aug(src_data_t)
        else:
            process_data, _ = self.transform_final((self.shift(src_data), None))
            process_data_t, _ = self.transform_final((self.transform_aug(src_data_t), None))
            process_data = process_data.view(
                (-1, 3) + process_data.size()[-2:]).transpose(0, 1)
            process_data_t = process_data_t.view(
                (-1, 3) + process_data_t.size()[-2:]).transpose(0, 1)
        encoder_mask_map = self.encoder_mask_map_generator()
        if hasattr(self, 'decoder_mask_map_generator'):
            decoder_mask_map = self.decoder_mask_map_generator()
        else:
            decoder_mask_map = 1 - encoder_mask_map
        return process_data, process_data_t, encoder_mask_map, decoder_mask_map

    def __repr__(self):
        repr = "(DataAugmentation,\n"
        repr += "  transform = %s,\n" % str(self.transform_base)
        repr += "  Encoder Masking Generator = %s,\n" % str(
            self.encoder_mask_map_generator)
        if hasattr(self, 'decoder_mask_map_generator'):
            repr += "  Decoder Masking Generator = %s,\n" % str(
                self.decoder_mask_map_generator)
        else:
            repr += "  Do not use decoder masking,\n"
        repr += ")"
        return repr


class CAGPretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        setting,
        name_pattern='img_{:05}.jpg',
        new_length=1,
        new_step=1,
        transform=None,
        temporal_jitter=False,
        lazy_init=False,
        num_sample=1, 
        shift_frames=False, 
        debug=False, 
    ):

        super(CAGPretrainDataset, self).__init__()
        self.root = root
        self.setting = setting
        self.num_sample = num_sample
        self.num_segments = 2 if shift_frames else 1  # to implement frame shifting aug.
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.transform = transform
        self.lazy_init = lazy_init
        self.debug = debug
        
        self.image_loader = get_image_loader()

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise (
                    RuntimeError("Found 0 video clips in subfolders of: " +
                                 root + "\n"
                                 "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        try:
            video_name, start_idx, total_frame = self.clips[index]
            
            fname_tmpl = self.name_pattern

            segment_indices, skip_offsets = self._sample_train_indices(
                total_frame)
            frame_id_list = self.get_frame_id_list(total_frame,
                                                    segment_indices,
                                                    skip_offsets)
            
            images = []
            process_data_list = []
            process_data_t_list = []
            encoder_mask_list = []
            decoder_mask_list = []

            for i, seg_frame_id_list in enumerate(frame_id_list):
                if i == 0 or seg_frame_id_list[0] != frame_id_list[i-1][0]:
                    # images = []
                    for idx in seg_frame_id_list:
                        frame_fname = '_'.join(
                            [video_name, fname_tmpl.format(idx + start_idx)])
                        img = self.image_loader(frame_fname)
                        img = Image.fromarray(img)
                        images.append(img)

            for _ in range(self.num_sample):
                process_data, process_data_t, encoder_mask, decoder_mask = self.transform(images, debug=self.debug) 
                process_data_list.append(process_data)
                process_data_t_list.append(process_data_t)
                encoder_mask_list.append(encoder_mask)
                decoder_mask_list.append(decoder_mask)
                
            if self.debug:
                return frame_id_list, process_data_list, process_data_t_list
            return process_data_list, process_data_t_list, encoder_mask_list, decoder_mask_list

        except Exception as e:
            if self.debug:
                traceback.print_exc()
                return 
            print("Failed to load video from {} with error {}".format(
                video_name, e))
            index = random.randint(0, len(self.clips) - 1)
            return self.__getitem__(index)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, root, setting):
        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. "
                % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(',')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise (RuntimeError(
                        'Video input format is not correct, missing one or more element. %s'
                        % line))
                clip_path = os.path.join(root, line_info[0])
                start_idx = int(line_info[1])
                total_frame = int(line_info[2])
                item = (clip_path, start_idx, total_frame)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(
                    num_frames - self.skip_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def get_frame_id_list(self, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            seg_frame_id_list = []
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                seg_frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
            frame_id_list.append(seg_frame_id_list)
        return frame_id_list
