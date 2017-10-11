import io
import bson
from skimage.data import imread
import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, Iterator, ImageDataGenerator

#LOAD THE REAL DATA WITH KERAS
# I DON'T REALLY KNOW HOW THIS WORKS, BUT IT DOES

def make_category_tables():
    
    categories_df = pd.read_csv("categories.csv", index_col=0)
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, target_size=(180, 180), with_labels=True,
                 batch_size=32, shuffle=False, seed=None):

        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)

        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            #batch_y = np.zeros((len(batch_x), 49), dtype=K.floatx())

            #batch_y = np.zeros(len(batch_x), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                #cat1_idx = le.transform(nt.loc[image_row["category_idx"]]['category_level1'])
                #batch_y[i, cat1_idx] = 1
                batch_y[i, image_row["category_idx"]] = 1
                #batch_y[i] = image_row["category_idx"]

        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    
def load_iterators():
    train_bson_path = "train.bson"



    categories_df = pd.read_csv("categories.csv", index_col=0)
    cat2idx, idx2cat = make_category_tables()

    train_offsets_df = pd.read_csv("train_offsets.csv", index_col=0)
    train_images_df = pd.read_csv("train_images.csv", index_col=0)
    val_images_df = pd.read_csv("val_images.csv", index_col=0)

    train_bson_file = open(train_bson_path, "rb")

    num_classes = 5270
    num_train_images = len(train_images_df)
    num_val_images = len(val_images_df)
    batch_size = 128

    # Tip: use ImageDataGenerator for data augmentation and preprocessing.
    train_datagen = ImageDataGenerator()
    train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df, 
                             num_classes, train_datagen, batch_size=batch_size, shuffle=True)

    val_datagen = ImageDataGenerator()
    val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                           num_classes, val_datagen, batch_size=batch_size)

    return train_gen, val_gen

