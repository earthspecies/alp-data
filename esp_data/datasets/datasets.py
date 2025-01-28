from base import BaseDataset


class HFDataset(BaseDataset):
    def from_dict(self):
        pass

    def __getitem__(self, idx: int):
        pass

    def __len__(self):
        pass

    def subset(self, indices):
        pass

    def sample(self, n, probabilities):
        pass

    def build_dataset_from_path(self):
        pass

    def build_dataset_from_pairs(self):
        pass
