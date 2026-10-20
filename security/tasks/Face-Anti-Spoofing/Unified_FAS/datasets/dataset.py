"""CelebA-Spoof dataset.

Expected dataset layout (official CelebA-Spoof structure)::

    CelebA_Spoof/
        Data/
            train/
                {subject_id}/
                    live/{image_id}.jpg
                    spoof/{image_id}.jpg
            test/
                {subject_id}/
                    live/{image_id}.jpg
                    spoof/{image_id}.jpg
        metainfo/
            train_label.json
            test_label.json

Each image may have a sibling ``{image_id}_BB.txt`` bounding-box file (used
for face cropping). The metainfo JSON maps a relative image path to a
44-element annotation vector; index 43 is the Live/Spoof label (``0`` = live,
``1`` = spoof), following the official ``intra_dataset_code/client.py``.
"""
import os
import sys
import json

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))


class CelebASpoofDataset(Dataset):
    """Image-level CelebA-Spoof dataset.

    Config keys (passed through **kwargs from the yaml):
        data_root (str): CelebA_Spoof root.
        metainfo_dir (str): directory containing the label JSON files.
        train_label_json / val_label_json / test_label_json (str).
        split_data (dict): optional mapping of split names to data splits,
            e.g. ``{val: train}`` because CelebA-Spoof has no official val set.
        val_from_train_ratio (float): if > 0, build a deterministic
            subject-level validation split from the training split.
        split_seed (int): seed for the validation split.
        use_LMDB (bool): read from LMDB instead of raw files.
        img_mode (str): 'rgb' (3ch), 'rgb_hsv' (6ch) or 'rgb_ycrcb' (6ch).
        aux_map (str|None): 'depth', 'reflection' or None. CelebA-Spoof does
            not provide depth/reflection maps, so this defaults to None.
        aux_map_size (int): spatial size of the auxiliary map.
        crop_face (str): 'bb' to crop with ``_BB.txt`` files, 'none' otherwise.
        bb_scale (float): reference resolution of the ``_BB.txt`` coordinates.
        return_path (bool): return the relative image path.
    """

    def __init__(self,
                 data_root=None,
                 split='train',
                 category=None,
                 transform=None,
                 img_mode='rgb',
                 print_info=False,
                 **kwargs):
        self.data_root = data_root
        self.split = split
        self.category = category
        self.transform = transform
        self.img_mode = img_mode

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.use_lmdb = bool(getattr(self, 'use_LMDB', False))
        self.return_path = getattr(self, 'return_path', True)
        self.aux_map = getattr(self, 'aux_map', None)
        self.aux_map_size = int(getattr(self, 'aux_map_size', 32))
        self.crop_face = getattr(self, 'crop_face', 'bb')
        self.bb_scale = float(getattr(self, 'bb_scale', 224.0))
        self.metainfo_dir = getattr(self, 'metainfo_dir', 'metainfo')
        self.val_from_train_ratio = float(getattr(self, 'val_from_train_ratio', 0.0))
        self.split_seed = int(getattr(self, 'split_seed', 1234))

        if self.use_lmdb:
            self.env = lmdb.open(self.LMDB_root, readonly=True, max_readers=1024)
            self.data = self.env.begin(write=False)

        self._build_items()

        if print_info:
            self._display_infos()

    # ------------------------------------------------------------------ #
    # item list construction
    # ------------------------------------------------------------------ #
    def _data_split(self, split):
        """Map a requested split (train/val/test) to a data split."""
        mapping = getattr(self, 'split_data', None)
        if mapping:
            return mapping.get(split, split)
        return split

    def _json_items(self, split):
        json_name = getattr(self, f'{split}_label_json', f'{split}_label.json')
        json_path = os.path.join(self.data_root, self.metainfo_dir, json_name)
        if not os.path.exists(json_path):
            return []

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        items = []
        for img_rel, attrs in annotations.items():
            if isinstance(attrs, dict):
                label = int(attrs.get('43', attrs.get(43, 1)))
            else:
                label = int(attrs[43])  # 0 = live, 1 = spoof
            items.append((img_rel, label))
        return items

    def _scan_items(self, split):
        split_dir = os.path.join(self.data_root, 'Data', split)
        if not os.path.isdir(split_dir):
            return []

        items = []
        for subject in sorted(os.listdir(split_dir)):
            subject_dir = os.path.join(split_dir, subject)
            if not os.path.isdir(subject_dir):
                continue
            for cls_name, label in (('live', 0), ('spoof', 1)):
                cls_dir = os.path.join(subject_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rel_path = os.path.join('Data', split, subject, cls_name, fname)
                        items.append((rel_path, label))
        return items

    @staticmethod
    def _subject_of(img_rel):
        """Extract subject id from ``Data/{split}/{subject_id}/{live|spoof}/...``."""
        rel = img_rel.replace('\\', '/')
        parts = rel.split('/')
        return parts[2] if len(parts) >= 4 else rel

    def _split_train_val(self, items, data_split):
        """Deterministic subject-level train/val split.

        Returns the subset of ``items`` belonging to the requested split.
        """
        if data_split != 'train' or self.val_from_train_ratio <= 0:
            return items

        rng = np.random.RandomState(self.split_seed)
        subjects = sorted({self._subject_of(rel) for rel, _ in items})
        n_val = max(1, int(round(len(subjects) * self.val_from_train_ratio)))
        val_subjects = set(rng.choice(subjects, size=n_val, replace=False))

        keep_val = (self.split == 'val')
        return [it for it in items
                if (self._subject_of(it[0]) in val_subjects) == keep_val]

    def _build_items(self):
        data_split = self._data_split(self.split)
        items = self._json_items(data_split)
        if not items:
            items = self._scan_items(data_split)

        items = self._split_train_val(items, data_split)

        if self.category == 'pos':
            items = [it for it in items if it[1] == 0]
        elif self.category == 'neg':
            items = [it for it in items if it[1] == 1]

        self.items = items

    # ------------------------------------------------------------------ #
    # low-level loading
    # ------------------------------------------------------------------ #
    def _resolve_path(self, img_rel):
        """Resolve a metainfo key to an on-disk relative path + absolute path."""
        rel = img_rel.replace('\\', '/')
        candidates = [rel]
        if not rel.startswith('Data/'):
            candidates.append(f'Data/{rel}')

        for cand in candidates:
            full = os.path.join(self.data_root, cand)
            if os.path.exists(full):
                return cand, full
        return rel, os.path.join(self.data_root, rel)

    def _load_image(self, img_rel):
        """Load a BGR uint8 image [H, W, 3] from LMDB or disk."""
        rel, img_path = self._resolve_path(img_rel)

        if self.use_lmdb:
            img_bin = self.data.get(rel.encode())
            if img_bin is None and rel != img_rel:
                img_bin = self.data.get(img_rel.encode())
            if img_bin is None:
                raise FileNotFoundError(f'Image not found in LMDB: {img_rel}')
            img_buf = np.frombuffer(img_bin, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(img_path)  # BGR uint8 [H, W, 3]

        if img is None:
            raise FileNotFoundError(f'Image not found: {img_path}')
        return img, img_path

    def _load_aux_map(self, img):
        """Return an auxiliary map [H, W] uint8.

        CelebA-Spoof does not provide depth/reflection ground truth, so we
        return zeros unless a user-provided ``aux_map`` hook is configured.
        """
        if self.aux_map is None:
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Placeholder for depth/reflection-map loading when a dataset that
        # provides such maps is used. Follows the DCN/ANRL convention where the
        # aux cue is zeroed for one class (see their dataset implementations).
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    def _crop_from_bb_file(self, img, img_path):
        """Crop the face using the CelebA-Spoof ``{img}_BB.txt`` bounding box."""
        bb_path = os.path.splitext(img_path)[0] + '_BB.txt'
        if not os.path.exists(bb_path):
            return img

        with open(bb_path, 'r') as f:
            line = f.readline().strip()
        parts = line.split(' ')
        if len(parts) < 4:
            return img

        real_h, real_w = img.shape[:2]
        scale_w = real_w / self.bb_scale
        scale_h = real_h / self.bb_scale

        x = int(float(parts[0]) * scale_w)
        y = int(float(parts[1]) * scale_h)
        w = int(float(parts[2]) * scale_w)
        h = int(float(parts[3]) * scale_h)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(real_w, x + w)
        y2 = min(real_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return img
        return img[y1:y2, x1:x2]

    def _crop_face(self, img, img_path):
        if self.crop_face in (None, 'none', False, 'False'):
            return img
        if self.crop_face == 'bb':
            return self._crop_from_bb_file(img, img_path)
        raise ValueError(f'Unsupported crop_face mode: {self.crop_face}')

    def _prepare_img_channels(self, img):
        """Convert BGR uint8 [H, W, 3] to the configured color mode.

        Returns (img, img2). ``img2`` is only used for 6-channel modes.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB uint8 [H, W, 3]
        if self.img_mode == 'rgb_hsv':
            img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.img_mode == 'rgb_ycrcb':
            img2 = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            img2 = None
        return img, img2

    def _scale_aux(self, aux):
        """Resize the aux map to ``aux_map_size`` and normalize to [0, 1]."""
        if self.aux_map is None:
            return torch.zeros((self.aux_map_size, self.aux_map_size), dtype=torch.float32)

        if not torch.is_tensor(aux):
            aux = torch.from_numpy(aux)
        aux = aux.float()  # [H, W] float in [0, 255]
        aux = aux.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        aux = F.interpolate(aux,
                            size=(self.aux_map_size, self.aux_map_size),
                            mode='bilinear',
                            align_corners=False)
        return aux.squeeze(0).squeeze(0).div_(255.0)  # [S, S] float in [0, 1]

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        img_rel, label = self.items[index]

        img, img_path = self._load_image(img_rel)  # BGR uint8 [H, W, 3]
        img = self._crop_face(img, img_path)
        aux = self._load_aux_map(img)

        img, img2 = self._prepare_img_channels(img)

        # Albumentations expects RGB/HWC uint8 input; ToTensorV2 outputs CHW float.
        if self.transform is not None:
            if img2 is not None:
                results = self.transform(image=img, image1=img2, mask=aux)
                img = results['image']      # float [C, H, W] in [0, 1]
                img2 = results['image1']    # float [C, H, W] in [0, 1]
                aux = results['mask']       # uint8 tensor [H, W]
            else:
                results = self.transform(image=img, mask=aux)
                img = results['image']      # float [C, H, W] in [0, 1]
                aux = results['mask']       # uint8 tensor [H, W]

        aux = self._scale_aux(aux)          # float [S, S] in [0, 1]

        if img2 is not None:
            img = torch.cat([img, img2], dim=0)  # float [2C, H, W]

        if self.return_path:
            return img, label, aux, img_rel
        return img, label, aux

    def __len__(self):
        return len(self.items)

    def _display_infos(self):
        print(f'=> Dataset {self.__class__.__name__} loaded')
        print(f'=> Split {self.split}')
        print(f'=> Category {self.category}')
        print(f'=> Image mode: {self.img_mode}')
        print(f'=> Crop face: {self.crop_face}')
        print(f'=> Total number of items: {len(self.items)}')
