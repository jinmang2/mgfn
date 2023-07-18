import os
import torch.utils.data as data
import numpy as np
from utils.utils import process_feat
import torch


class Dataset(data.Dataset):
    def __init__(
        self,
        rgb_list_file: str,
        is_normal=True,
        transform=None,
        test_mode=False,
        seg_length: int = 32,
        root: str = "",
    ):
        self.root = root
        self.is_normal = is_normal
        self.rgb_list_file = rgb_list_file
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.seg_length = seg_length

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                self.list = self.list[810:]  # ucf 810; sht63; xd 9525
            else:
                self.list = self.list[:810]  # ucf 810; sht 63; 9525
    
    def _get_filename(self, index):
        return os.path.join(self.root, self.list[index])

    def __getitem__(self, index):
        label = self.get_label(index)  # get video level label 0/1
        features = np.load(self._get_filename(index).strip("\n"), allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        name = self._get_filename(index).split("/")[-1].strip("\n")[:-4]
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            mag = np.linalg.norm(features, axis=2)[:, :, np.newaxis]
            features = np.concatenate((features, mag), axis=2)
            return features, name
        else:
            features = features.transpose(1, 0, 2)  # [10, T, F]
            divided_features = []

            divided_mag = []
            for feature in features:
                feature = process_feat(feature, self.seg_length)  # ucf(32,2048)
                divided_features.append(feature)
                divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
            divided_features = np.array(divided_features, dtype=np.float32)
            divided_mag = np.array(divided_mag, dtype=np.float32)
            divided_features = np.concatenate(
                (divided_features, divided_mag), axis=2
            )
            return divided_features, label

    def get_label(self, index):
        label = 0.0 if self.is_normal else 1.0
        return torch.tensor(label)

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
