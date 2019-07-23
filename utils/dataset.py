from torch.utils.data import Dataset
import os


class CTScanDataset(Dataset):
    '''
    __getitem__ returns the 3D numpy arrays pair (source,target)
    for the given index after applying the specified transforms.
    Waiting for data access approval from
    NCTN/NCORP Data Archive to implement the function.
    '''
    def __init__(self, data_dir, transform=None):
        super(CTScanDataset, self).__init__()
        self.dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.dir))//2

    def __getitem__(self, index):
        pass
