import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            img = np.flip(img, axis=1)
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Padding is added to all side of the image, and then the image is cropped back to its original size at a random location.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of clipped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )

       # hbsun impl:
        img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        shift_x = np.random.randint(0, self.padding * 2 + 1)
        shift_y = np.random.randint(0, self.padding * 2 + 1)
        img = img[shift_x:shift_x + img.shape[0] - self.padding * 2, shift_y:shift_y + img.shape[1] - self.padding * 2, :]
        return img


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.

    Each Dataset subclass must implement three functions: __init__, __len__, and __getitem__.
    The __init__ function initializes the images, labels, and transforms.
    The __len__ function returns the number of samples in the dataset.
    The __getitem__ function retrieves a sample from the dataset at a given index idx,
    calls the transform functions on the image (if applicable),
    converts the image and label to a numpy array (the data will be converted to Tensors elsewhere).
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(self.dataset)), range(self.batch_size, len(self.dataset), self.batch_size))
        else:
            l = np.arange(len(self.dataset))
            np.random.shuffle(l)
            self.ordering = np.array_split(l, range(self.batch_size, len(self.dataset), self.batch_size))
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if len(self.ordering) == 0:
            raise StopIteration
        ind = self.ordering.pop(0)
        x = Tensor(np.array([self.dataset[i][0] for i in ind]))
        y = None
        if len(self.dataset[0]) > 1:
            y = Tensor(np.array([self.dataset[i][1] for i in ind]))
        return (x, y)
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        self.X, self.y = parse_mnist(image_filename, label_filename)
        self.X = np.reshape(self.X, (self.X.shape[0], 28, 28, 1))

    def __getitem__(self, index) -> object:
        return self.apply_transforms(self.X[index]), self.y[index]

    def __len__(self) -> int:
        return self.X.shape[0]


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        with open(os.path.join(base_folder, "batches.meta"), "rb") as fo:
            self.label_names = pickle.load(fo, encoding="bytes")[b"label_names"]
        self.train = train
        self.transforms = transforms
        self.p = p
        if self.train:
            self.data = []
            self.labels = []
            for i in range(1, 6):
                with open(os.path.join(base_folder, f"data_batch_{i}"), "rb") as fo:
                    entry = pickle.load(fo, encoding="bytes")
                    self.data.append(entry[b"data"])
                    if b"labels" in entry:
                        self.labels.append(entry[b"labels"])
                    else:
                        self.labels.append(entry[b"fine_labels"])
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.labels = np.hstack(self.labels)
        else:
            with open(os.path.join(base_folder, "test_batch"), "rb") as fo:
                entry = pickle.load(fo, encoding="bytes")
                self.data = entry[b"data"].reshape(-1, 3, 32, 32)
                if b"labels" in entry:
                    self.labels = entry[b"labels"]
                else:
                    self.labels = entry[b"fine_labels"]

        self.data = self.data / 255.

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        return self.apply_transforms(self.data[index]).reshape(3, 32, 32), self.labels[index]


    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.data.shape[0]



class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        # hbsun: I think it's should be added.
        super().__init__()
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return len(self.idx2word)



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ids = []
        lineNum = 0
        with open(path, 'r') as f:
            for line in f:
                lineNum += 1
                if max_lines is not None and lineNum >= max_lines:
                    break
                for word in line.split() + ['<eos>']:
                    ids.append(self.dictionary.add_word(word))
        return ids


def batchify(data: list, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    nbatch = len(data) // batch_size
    data = data[:nbatch * batch_size]
    data = np.array(data, dtype=dtype)
    data = data.reshape((batch_size, -1)).T
    return data


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    seq_len = min(bptt, len(batches) - 1 - i)
    print("get_batch:", seq_len, batches.shape, bptt, i)
    data = batches[i:i+seq_len]
    target = batches[i+1:i+1+seq_len].reshape(-1)
    return Tensor(data, device=device, dtype=dtype), Tensor(target, device=device, dtype=dtype)