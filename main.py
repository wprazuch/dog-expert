import dog_expert
from dog_expert import DataLoader, Preprocessor
import tensorflow as tf


def main():
    preprocessor = Preprocessor()
    preprocessor.cast(dtype=tf.float32).normalize()

    class_mapping = {'affenpinscher': 0,
                     'afghan_hound': 1,
                     'african_hunting_dog': 2,
                     'airedale': 3,
                     'american_staffordshire_terrier': 4,
                     'appenzeller': 5,
                     'australian_terrier': 6,
                     'basenji': 7,
                     'basset': 8}

    data_loader = DataLoader(dataset_dir=r'datasets\stanford2',
                             batch_size=4, class_mapping=class_mapping, preprocessor=preprocessor)

    train = data_loader.train_dataset
    val = data_loader.val_dataset

    print(len(train))
    print(len(val))


if __name__ == '__main__':

    main()
