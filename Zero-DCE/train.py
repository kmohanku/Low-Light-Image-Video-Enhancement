import os
import tensorflow as tf
import numpy as np
from absl import flags, app
import time
from data import Dataloader
from loss import computeQuartretLoss
from model import DCENet

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", None, "Path to low light image data")
flags.DEFINE_string("activation_function", "relu", "Activation function to be used")
flags.DEFINE_integer("batch_size", 1, "Batch size for training")
flags.DEFINE_integer("depth", 7, "Number of convolution blocks. Must be odd number")
flags.DEFINE_boolean("preloadweights", False, "Should some pre-determined weights be loaded?")
flags.DEFINE_string("preloadweights_path", "./models/variables/variables", "Path to the weights that have to be loaded")
flags.DEFINE_integer("epochs", 200, "Number of epochs to train")
flags.DEFINE_float("learning_rate", 0.0001, "Learning Rate for the model")
flags.DEFINE_string("model_savepath", "./models", "Path to save the trained model")
flags.DEFINE_float("max_pixel", 255.0, "Maximum pixel value for the dataset")
flags.DEFINE_multi_integer("resize_shape", [512,512], "Shape of the input images [H,W]")
flags.DEFINE_float("trainsplit", 0.8015, "Train - Validation split ratio")
flags.DEFINE_integer("crop_size", 128, "Dimensions for random crop")
flags.DEFINE_boolean("standardize", False, "standardize the data between -1 & 1?")
flags.DEFINE_integer("prefetch", 2, "NUmber of images to prefetch for loading")
flags.DEFINE_integer("features", 32, "Number of features in convolution block")
flags.DEFINE_integer("iterations", 8, "Number of optimizing iterations for curve tuning")
flags.DEFINE_string("final_activation", "tanh", "Activation function for the last convolution block")
flags.DEFINE_integer("kernel_size", 3, "Size of the convolution receptive field")
flags.DEFINE_string("padding", "SAME", "Padding for convolution")
flags.DEFINE_integer("stride", 1, "Kernel stride for convolution")
flags.DEFINE_float("spatial_loss_weight", 1.0, "Weighting for Spacial Constancy Loss")
flags.DEFINE_integer("exposure_loss_patch", 16, "Patch size to compute exposure loss")
flags.DEFINE_float("exposure_loss_mean", 0.6, "Mean value for exposure loss")


ACTIVATION_MAPS = {
    "relu" : tf.nn.relu,
    "lrelu": tf.nn.leaky_relu,
    "tanh" : tf.nn.tanh
}

def forward_pass(model, data):
    interim_image, enhanced_image, curves = model(data)
    total_loss = computeQuartretLoss(data, 
                                     enhanced_image, 
                                     curves, 
                                     spaCon_weight=FLAGS.spatial_loss_weight, 
                                     patchSize=FLAGS.exposure_loss_patch, 
                                     meanVal=FLAGS.exposure_loss_mean
                                     )
    return total_loss

def backprop_pass(model, data):
    with tf.GradientTape() as tape:
        loss_value = forward_pass(model, data)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def main(argvs):
    """Create/Validate model save path"""
    if not os.path.isdir(FLAGS.model_savepath):
        os.mkdir(FLAGS.model_savepath)
    """Load the data"""
    training_images, validation_images = Dataloader(dataPath=FLAGS.data_path,
                                                    batch= FLAGS.batch_size,
                                                    trainsplit=FLAGS.trainsplit,
                                                    resize_w=FLAGS.resize_shape[1],
                                                    resize_h=FLAGS.resize_shape[0],
                                                    cropsize=FLAGS.crop_size,
                                                    max_pixel=FLAGS.max_pixel,
                                                    prefetch=FLAGS.prefetch,
                                                    standardize=FLAGS.standardize
                                                    ).prepareDataset()

    if(FLAGS.activation_function in ACTIVATION_MAPS.keys()):
        FLAGS.activation_function = ACTIVATION_MAPS[FLAGS.activation_function]
    else:
        FLAGS.activation_function = None

    """Load the network/model"""
    dceNet = DCENet(numConvs=FLAGS.depth,
                    blockSize=FLAGS.features,
                    iterations=FLAGS.iterations, 
                    activation_fn=FLAGS.activation_function, 
                    final_activation_fn=ACTIVATION_MAPS[FLAGS.final_activation],
                    kernelSize=FLAGS.kernel_size, 
                    kernelStride=FLAGS.stride, 
                    kernelPadding=FLAGS.padding
                    )
    print("Number of model parameters: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in dceNet.trainable_variables])))
    if FLAGS.preloadweights:
        print("Loading weights from path {}".format("/".join(x for x in FLAGS.preloadweights_path.split('/')[:-2])))
        for var, oldvar in zip(dceNet.trainable_variables, tf.train.list_variables(FLAGS.preloadweights_path)[1:]):
            var.assign(tf.convert_to_tensor(tf.train.load_variable(FLAGS.preloadweights_path, name=oldvar[0]), dtype=tf.float32))
    optimizer = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    best_val_loss = None
    for epoch in range(FLAGS.epochs):
        epoch_trainloss_avg = tf.metrics.Mean()
        epoch_valloss_avg = tf.metrics.Mean()
        print("Training on Epoch {}".format(epoch + 1))
        start_epoch_time = time.time()
        for i, data in enumerate(training_images):
            loss, grads = backprop_pass(dceNet, data)
            optimizer.apply_gradients(zip(grads, dceNet.trainable_variables))
            epoch_trainloss_avg.update_state(loss)
        epoch_time = time.time() - start_epoch_time
        print("Training Summary: Time(s):{:.0f} Loss: {:.6f}".format(epoch_time,
                                                                         epoch_trainloss_avg.result()))

        print("Validating on Epoch {}".format(epoch + 1))
        start_epoch_time = time.time()
        for i, valdata in enumerate(validation_images):
            valloss = forward_pass(dceNet, valdata)
            epoch_valloss_avg.update_state(valloss)
        epoch_time = time.time() - start_epoch_time
        print("Validation Summary: Time(s):{:.0f} Loss: {:.6f}".format(epoch_time,
                                                                         epoch_valloss_avg.result()))

        if best_val_loss is None or best_val_loss > epoch_valloss_avg.result():
            print("New Best Validation Loss: {:.6f}".format(epoch_valloss_avg.result()))
            print("Saving new best model")
            call = dceNet.__call__.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.float32, name="input"))
            tf.saved_model.save(dceNet, FLAGS.model_savepath, signatures=call)
            best_val_loss = epoch_valloss_avg.result()

if __name__ == '__main__':
    flags.mark_flag_as_required('data_path')
    flags.register_validator('depth',
                         lambda value: value % 2 != 0,
                         message='ERROR: Number of convolution blocks must be odd')
    app.run(main)