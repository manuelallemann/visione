import argparse
import logging
import os

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import h5py
from visione.extractor import BaseExtractor


loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)


def load_image(image_path, *, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image)
    return image


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return load_image(self.image_paths[idx], transform=self.transform)




class FlatHDF5Saver:
    """Saver that writes two flat datasets: 'ids' and 'data' suitable for frame-cluster."""

    def __init__(self, path, force: bool = False):
        from pathlib import Path
        self.path = Path(path)
        self.force = force
        self._file = None

    def __enter__(self):
        if self.force and self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # open in append or write mode (always write fresh for simplicity)
        self._file = h5py.File(self.path, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None

    # API compatibility with BaseExtractor expectations
    def add_many(self, records, force: bool = False):
        ids = np.array([r['_id'] for r in records], dtype=h5py.string_dtype(encoding='utf-8'))
        data = np.stack([r['feature_vector'] for r in records]).astype('float32')
        self._file.create_dataset('ids', data=ids, compression="gzip")
        self._file.create_dataset('data', data=data, compression="gzip")

    def flush(self):
        if self._file:
            self._file.flush()

    def __contains__(self, key):
        # Flat file saver does not support incremental contains checks; always return False so extractor processes all.
        return False


class DinoV2Extractor(BaseExtractor):

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--model', default='dinov2_vits14', choices=('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'), help='Model to use')
        super(DinoV2Extractor, cls).add_arguments(parser)

    def __init__(self, args):
        super(DinoV2Extractor, self).__init__(args)
        self.device = None
        self.model = None

    # Override saver to write flat datasets
    def get_saver(self, video_id):
        output_path = str(self.args.output).format(video_id=video_id)
        return FlatHDF5Saver(output_path, force=self.args.force)

    def setup(self):
        if self.model is None:
            import os
            import torch
            self.device = 'cuda' if self.args.gpu and torch.cuda.is_available() else 'cpu'
            cache_dir = os.environ.get('VISIONE_CACHE', '/tmp/torch_hub')
            os.makedirs(cache_dir, exist_ok=True)
            torch.hub.set_dir(cache_dir)
            try:
                self.model = torch.hub.load('facebookresearch/dinov2', self.args.model).to(self.device)
            except PermissionError:
                temp_cache = '/tmp/torch_hub_temp'
                os.makedirs(temp_cache, exist_ok=True)
                torch.hub.set_dir(temp_cache)
                self.model = torch.hub.load('facebookresearch/dinov2', self.args.model).to(self.device)
            self.model.eval()

    def extract(self, image_paths):
        self.setup()

        dataset = ImageListDataset(image_paths)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        features = []
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device, non_blocking=True)
                fv = self.model(x).cpu().numpy()
                features.append(fv)
        features = np.concatenate(features, axis=0)
        records = [{'_id': os.path.splitext(os.path.basename(p))[0], 'feature_vector': f.tolist()} for p, f in zip(image_paths, features)]
        return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from a DINOv2 model')
    DinoV2Extractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = DinoV2Extractor(args)
    extractor.run()