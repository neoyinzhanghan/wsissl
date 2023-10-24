import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PFDataset(Dataset):
    def __init__(self, folders, num_images_per_epoch, transform):
        self.transform = transform
        self.num_images_per_epoch = num_images_per_epoch
        self.folders = folders
        self.folder_images = {
            folder: [
                os.path.join(folder, img)
                for img in os.listdir(folder)
                if img.endswith(".jpg")
            ]
            for folder in self.folders
        } # the name of each folder is being mapped to a list of images in that folder

    def __len__(self):
        # 64 batches per epoch
        return self.num_images_per_epoch

    def __getitem__(self, idx):
        # Sample a random folder
        folder = random.choice(self.folders)

        # Sample a random image from the chosen folder
        folder_images = self.folder_images[folder]
        image_path = random.choice(folder_images)

        # load the JPG image in image_path as PIL
        image = Image.open(image_path)

        if self.transform is None:
            raise ValueError('Transform should not be None. ') # TODO -- None transform should in general be supported -- currently not, for debugging purposes
        else:  
            image = self.transform(image)

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)

        # Return the image with a fixed class label (e.g., 0)
        return image, 0


def folder_train_test_split(data_dir, train_prop=0.8):
    """Data dir contains a number of folders, create the list of folders paths and divide them into train and test sets.
    Which are just lists of paths to folders."""

    folders = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    random.shuffle(folders)
    train_size = int(len(folders) * train_prop)
    train_folders = folders[:train_size]
    test_folders = folders[train_size:]
    return train_folders, test_folders
