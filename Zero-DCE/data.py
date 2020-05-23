import tensorflow as tf

class Dataloader:
    def __init__(self, dataPath,
                 max_pixel=255.0,
                 resize_h=512, 
                 resize_w=512, 
                 batch=8, 
                 prefetch=2, 
                 trainsplit=0.8015, 
                 cropsize=128, 
                 standardize=False
                ):

        self.height = resize_h
        self.width = resize_w
        self.max_pixel = max_pixel
        self.batch = batch
        self.prefetch = prefetch
        self.dataPath = dataPath
        self.trainsplit = trainsplit
        self.cropsize = cropsize
        self.standardize = standardize
    
    # Range -1 to 1 with mean 0    
    def _standardize(self, img):
        return ((img * (2.0/self.max_pixel)) - 1.0)
    
    # Range 0 to 1 with mean 0.5
    def _normalize(self, img):
        return img * (1.0/self.max_pixel)
    
    # Load image and perform basic augmentation
    @tf.autograph.experimental.do_not_convert
    def _transform_data(self, img, resize=False, random_crop=False, flips=False):
        input_image = tf.io.read_file(img)
        input_image = tf.cast(tf.io.decode_jpeg(input_image, channels=3), dtype=tf.float32)
        if resize:
            input_image = tf.image.resize(input_image, [self.height, self.width])
        if not resize and random_crop:
            input_image = tf.image.random_crop(input_image, size=[self.cropsize, self.cropsize, input_image.shape[2]])
        if flips:
            tf.cond(tf.random.uniform(()) > 0.5, lambda: tf.image.flip_left_right(input_image), lambda: input_image)
            tf.cond(tf.random.uniform(()) > 0.5, lambda: tf.image.flip_up_down(input_image), lambda: input_image)
        if self.standardize:
            input_image = self._standardize(input_image)
        else:
            input_image = self._normalize(input_image)
        return input_image

    # Generate train and validation data
    def prepareDataset(self, resize=True, random_crop=False, flips=True):
        dataset_images = tf.data.Dataset.list_files(self.dataPath)
        dataset_images = dataset_images.shuffle(tf.data.Dataset.cardinality(dataset_images), seed=tf.random.set_seed(1234),
                                                reshuffle_each_iteration=False)
        training_images = dataset_images.take(int(tf.data.Dataset.cardinality(dataset_images).numpy() * self.trainsplit))

        validation_images = dataset_images.skip(int(tf.data.Dataset.cardinality(dataset_images).numpy() * self.trainsplit))
        training_images = training_images.shuffle(tf.data.Dataset.cardinality(training_images), reshuffle_each_iteration=True)
        training_images = training_images.map(lambda inp: self._transform_data(inp, resize=resize, random_crop=random_crop, flips=flips), 
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        training_images = training_images.batch(self.batch)
        training_images = training_images.prefetch(self.prefetch)

        validation_images = validation_images.map(lambda inp: self._transform_data(inp, resize=resize, random_crop=random_crop, flips=flips),
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_images = validation_images.batch(self.batch)
        validation_images = validation_images.prefetch(self.prefetch)
        return training_images, validation_images
    
    # Generate Test data
    def prepareTestDataset(self):
        dataset_images = tf.data.Dataset.list_files(self.dataPath)
        testing_images = dataset_images.map(lambda inp: self._transform_data(inp), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testing_images = testing_images.batch(1)
        testing_images = testing_images.prefetch(self.prefetch)
        return testing_images