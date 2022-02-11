#run me!
from typing import Tuple, Union, Optional

import tensorflow as tf
from tensorflow import keras

import h5py as h5
from tensorflow.keras.utils import to_categorical

#from albumentations import Compose
import numpy as np

available_modes = {"train", "test"}
available_labels_encoding = {"hot", "smooth", False}

def my_print(s):
    print(s)
    with open('log.log', 'a') as f:
        f.write(str(s) + '\n')


class HDF5ImageGenerator(keras.utils.Sequence):
    """Just a simple custom Keras HDF5 ImageDataGenerator.
    
    Custom Keras ImageDataGenerator that generates
    batches of tensor images from HDF5 files with (optional) real-time
    data augmentation.
     
    Arguments
    ---------
    src : str
        Path of the hdf5 source file.
    X_key : str
        Key of the h5 file image tensors dataset.
        Default is "images".
    y_key : str
        Key of the h5 file labels dataset.
        Default is "labels".
    classes_key : str
        Key of the h5 file dataset containing
        the raw classes.
        Default is None.
    batch_size : int
        Size of each batch, must be a power of two.
        (16, 32, 64, 128, 256, ...)
        Default is 32.
    shuffle : bool
        Shuffle images at the end of each epoch.
        Default is True.
    scaler : "std", "norm" or False
        "std" mode means standardization to range [-1, 1]
        with 0 mean and unit variance.
        "norm" mode means normalization to range [0, 1].
        Default is "std".
    num_classes : None or int
        Specifies the total number of classes
        for labels encoding.
        Default is None.
    labels_encoding : "hot", "smooth" or False
        "hot" mode means classic one hot encoding.
        "smooth" mode means smooth hot encoding.
        Default is "hot".
    smooth_factor : int or float
        smooth factor used by smooth
        labels encoding.
        Default is 0.1.
    augmenter : albumentations Compose([]) Pipeline or False
        An albumentations transformations pipeline
        to apply to each sample.
        Default is False.
    mode : str "train" or "test"
        Model generator type. "train" is used for
        fit_generator() and evaluate_generator.
        "test" is used for predict_generator().
        Default is "train".
        
    Notes
    -----
    Turn off scaler (scaler=False) if using the
    ToFloat(max_value=255) transformation from
    albumentations.
        
    Examples
    --------
    Example of usage:
    ```python
    my_augmenter = Compose([
        HorizontalFlip(p=0.5),
        RandomContrast(limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        Resize(227, 227, cv2.INTER_AREA)
    ])
    # Create the generator.
    train_gen = HDF5ImageGenerator(
        'path/to/my/file.h5',
         augmenter=my_augmenter)
    ```
    """
    def __init__(
        self,
        src,
        indices_sel = [], #incides are already used by this lib
        X_key="data/gasf",
        y_key="labels/gaf",
        classes_key="classes/appliances",
        batch_size=8,
        shuffle=False,
        scaler=False,
        num_classes=9,
        labels_encoding="hot",
        smooth_factor=0.1,
        augmenter=False,
        mode="train",
    ):

        if mode not in available_modes:
            raise ValueError('`mode` should be `train` '
                             '(fit_generator() and evaluate_generator()) or '
                             '`test` (predict_generator(). '
                             'Received: %s' % mode)
        self.mode = mode

        if labels_encoding not in available_labels_encoding:
            raise ValueError('`labels_encoding` should be `hot` '
                             '(classic binary matrix) or '
                             '`smooth` (smooth encoding) or '
                             'False (no labels encoding). '
                             'Received: %s' % labels_encoding)
        self.labels_encoding = labels_encoding

        if (self.labels_encoding == "smooth") and not (0 < smooth_factor <= 1):
            raise ValueError('`smooth` labels encoding '
                             'must use a `smooth_factor` '
                             '< 0 smooth_factor <= 1')

        if augmenter and not isinstance(augmenter, Compose):
            raise ValueError('`augmenter` argument '
                             'must be an instance of albumentations '
                             '`Compose` class. '
                             'Received type: %s' % type(augmenter))
        self.augmenter = augmenter

        self.src: str = src
        self.X_key: str = X_key
        self.y_key: str = y_key
        self.classes_key: str = classes_key
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.scaler: bool = scaler
        self.num_classes: int = num_classes
        self.smooth_factor: float = smooth_factor
        
    
        if indices_sel:
            self._indices = np.array(x_train_ids)
            print("calculating shape beforehand...")
            self._parse_mode = "indices";
            self._ds_shape = file[self.y_key][np.sort(self._indices)].shape
        else:
            print("Normal way")
            self._parse_mode = "normal";
            self._indices = np.arange(self.__get_dataset_shape(self.X_key, 0))
        
        print(self._indices[:10])
        
       

    def __repr__(self):
        """Representation of the class."""
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def __get_dataset_shape(self, dataset: str, index: int) -> Tuple[int, ...]:
        """Get an h5py dataset shape.
        
        Arguments
        ---------
        dataset : str
            The dataset key.
        index : int
            The dataset index.
         
        Returns
        -------
        tuple of ints
            A tuple of array dimensions.
        """
        with h5.File(self.src, "r", libver="latest", swmr=True) as file:
            #print("getting ds shape, current: ",self._ds_shape,"first time",self._first_time)
            #if self._first_time:
            #    print("calculating")
            #    self._ds_shape = file[self.y_key][np.sort(self._indices)].shape[index]
            #    self._first_time = 0
                
            #print("using dataset shape: ",self._ds_shape)   
            if self._parse_mode == "normal":
                #default way when no inndices defined
                return file[dataset].shape[index]
            else:
                #optimised way for indices
                return self._ds_shape[index]
            


    def __get_dataset_items(
        self,
        indices: np.ndarray,
        dataset: Optional[str] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get an HDF5 dataset items.
        
        Arguments
        ---------
        indices : ndarray, 
            The list of current batch indices.
        dataset : (optional) str
            The dataset key. If None, returns
            a batch of (image tensors, labels).
            Defaults to None.
         
        Returns
        -------
        np.ndarray or a tuple of ndarrays
            A batch of samples.
        """
        
        with h5.File(self.src, "r", libver="latest", swmr=True) as file:
            
            if dataset is not None:
                return file[dataset][indices]
            else:
                return (file[self.X_key][indices], file[self.y_key][indices])
    
    @property
    def num_items(self) -> int:
        """Grab the total number of examples
         from the dataset.
         
        Returns
        -------
        int
            The total number of examples.
        """
        with h5.File(self.src, "r") as file:
            return file[self.X_key].shape[0]
    
    @property 
    def classes(self) -> list:
        """Grab "human" classes from the dataset.
        
        Returns
        -------
        list
            A list of the raw classes.
        """
        if self.classes_key is None:
            raise ValueError('Canceled. parameter `classes_key` '
                             'is set to None.')
        
        with h5.File(self.src, "r") as file:
            return file[self.classes_key][:]

    def __len__(self):
        """Denotes the number of batches per epoch.
         
        Returns
        -------
        int
            The number of batches per epochs.
        """
        return int(
            np.ceil(
                self.__get_dataset_shape(self.X_key, 0) /
                float(self.batch_size)))

    @staticmethod
    def apply_labels_smoothing(batch_y: np.ndarray,
                               factor: float) -> np.ndarray:
        """Applies labels smoothing to the original
         labels binary matrix.
         
        Arguments
        ---------
        batch_y : np.ndarray
            Current batch integer labels.
        factor : float
            Smoothing factor.
        
        Returns
        -------
        np.ndarray
            A binary class matrix.
        """
        batch_y *= 1 - factor
        batch_y += factor / batch_y.shape[1]

        return batch_y

    def apply_labels_encoding(
            self,
            batch_y: np.ndarray,
            smooth_factor: Optional[float] = None) -> np.ndarray:
        """Converts a class vector (integers) to binary class matrix.
         See Keras to_categorical utils function.
         
        Arguments
        ---------
        batch_y : np.ndarray
            Current batch integer labels.
        smooth_factor : (optional) Float
            Smooth factor.
            Defaults to None.
        
        Returns
        -------
        np.ndarray
            A binary class matrix.
        """
        batch_y = to_categorical(batch_y, num_classes=self.num_classes)

        if smooth_factor is not None:
            batch_y = self.apply_labels_smoothing(batch_y,
                                                  factor=smooth_factor)

        return batch_y

    @staticmethod
    def apply_normalization(batch_X: np.ndarray) -> np.ndarray:
        """Normalize the pixel intensities. 
        
        Normalize the pixel intensities to the range [0, 1].
         
        Arguments
        ---------
        batch_X : np.ndarray
            Batch of image tensors to be normalized.
        
        Returns
        -------
        np.ndarray
            A batch of normalized image tensors.
        """
        return batch_X.astype("float32") / 255.0

    def __next_batch_test(self, indices: np.ndarray) -> np.ndarray:
        """Generates a batch of test data for the given indices.
        
        Arguments
        ---------
        index : int
            The index for the batch.
            
        Returns
        -------
        ndarray
            4D tensor (num_samples, height, width, depth).
        """
        # Grab corresponding images from the HDF5 source file.
        batch_X = self.__get_dataset_items(indices, self.X_key)

        # Shall we rescale features?
        if self.scaler:
            batch_X = self.apply_normalization(batch_X)

        return batch_X

    def __next_batch(self,
                     indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a batch of train/val data for the given indices.
        
        Arguments
        ---------
        index : int
            The index for the batch.
            
        Returns
        -------
        tuple of ndarrays
            A tuple containing a batch of image tensors
            and their associated labels.
        """
        # Grab samples (tensors, labels) HDF5 source file.
        (batch_X, batch_y) = self.__get_dataset_items(indices)

        # Shall we apply any data augmentation?
        if self.augmenter:
            batch_X = np.stack(
                [self.augmenter(image=x)["image"] for x in batch_X], axis=0)

        # Shall we rescale features?
        if self.scaler:
            batch_X = self.apply_normalization(batch_X)

        # Shall we apply labels encoding?
        if self.labels_encoding:
            batch_y = self.apply_labels_encoding(
                batch_y,
                smooth_factor=self.smooth_factor
                if self.labels_encoding == "smooth" else None,
            )

        return (batch_X, batch_y)

    def __getitem__(
            self,
            index: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generates a batch of data for the given index.
        
        Arguments
        ---------
        index : int
            The index for the current batch.
            
        Returns
        -------
        tuple of ndarrays or ndarray
            A tuple containing a batch of image tensors
            and their associated labels (train) or
            a tuple of image tensors (predict).
        """
        # Indices for the current batch.
        indices = np.sort(self._indices[index * self.batch_size:(index + 1) *  self.batch_size])

        if self.mode == "train":
            return self.__next_batch(indices)
        else:
            return self.__next_batch_test(indices)

    def __shuffle_indices(self):
        """If the shuffle parameter is set to True,
         dataset will be shuffled (in-place).
         (not available in test 'mode').
        """
        #my_print(self.shuffle)
        if (self.mode == "train") and self.shuffle:
            #my_print(self._indices)
            np.random.shuffle(self._indices)
            #my_print(self.shuffle)
            #my_print(self._indices)

    def on_epoch_end(self):
        """Triggered once at the very beginning as well as 
         at the end of each epoch.
        """
        self.__shuffle_indices()