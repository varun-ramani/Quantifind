import tarfile
import random

from torch.utils.data import Dataset
import torch

from workflow import data_workflow


class LanguageNamesDataset(Dataset):
    def __init__(self, tar_path="datasets/languagenames.tar"):
        with tarfile.open(tar_path) as tfile:
            files = [file for file in tfile if "._" not in file.path]
            filedata = {
                file.name.split(".")[0]: tfile.extractfile(file).read().decode().split()
                for file in files
            }

            self.language_to_idx = {
                lang: idx for idx, lang in enumerate(filedata.keys())
            }
            self.all_names = [
                (name, self.language_to_idx[lang])
                for lang in filedata
                for name in filedata[lang]
            ]

            max_len = max(len(name) for name, _ in self.all_names)
            self.all_names = [
                (name.ljust(max_len, "\0"), lang) for name, lang in self.all_names
            ]
            self.all_names = [
                (torch.IntTensor([ord(c) for c in name]), lang)
                for name, lang in self.all_names
            ]

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, idx):
        return self.all_names[idx]


def create_dataloaders(
    batch_size=32, train_split=0.8, val_split=0.1, pin_memory=True, seed=42
):

    language_names = LanguageNamesDataset()
    return data_workflow.create_dataloaders(language_names)
