import os
import tensorflow as tf
import numpy as np
from absl import flags, app
import time
from data import Dataloader
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", None, "Path to low light image data")
flags.DEFINE_string("model_path", None, "Path to the saved model to be loaded")
flags.DEFINE_string("image_savepath", None, "Path to save the output images")
flags.DEFINE_float("max_pixel", 255.0, "Maximum pixel value for the dataset")
flags.DEFINE_multi_integer("resize_shape", [512,512], "Shape of the input images [H,W]")
flags.DEFINE_integer("crop_size", 128, "Dimensions for random crop")
flags.DEFINE_boolean("standardize", False, "standardize the data between -1 & 1?")
flags.DEFINE_integer("prefetch", 2, "NUmber of images to prefetch for loading")
flags.DEFINE_string("save_ext", ".png", "Extension of saved image")
flags.DEFINE_string("tflite_savepath", None, "path to save tflite model")
flags.DEFINE_boolean("float16", False, "Should the model be optimized to float16?")

def main(argvs):
    """Create/Validate image save path"""
    if FLAGS.image_savepath and not os.path.isdir(FLAGS.image_savepath):
        os.mkdir(FLAGS.image_savepath)
    """Load the data"""
    test_images = Dataloader(dataPath=FLAGS.data_path,
                             resize_w=FLAGS.resize_shape[1],
                             resize_h=FLAGS.resize_shape[0],
                             cropsize=FLAGS.crop_size,
                             max_pixel=FLAGS.max_pixel,
                             prefetch=FLAGS.prefetch,
                             standardize=FLAGS.standardize
                            ).prepareTestDataset()

    """Load the network/model"""
    dceNet = tf.saved_model.load(FLAGS.model_path)
    running_psnr = tf.metrics.Mean()
    running_ssim = tf.metrics.Mean()
    print("Running Evaluation")
    start_time = time.time()
    for i, data in enumerate(test_images):
        _, out, _ = dceNet(data)
        psnr = tf.image.psnr(out, data, max_val=1.0)
        running_psnr.update_state(psnr)
        ssim = tf.image.ssim(out, data, max_val=1.0)
        running_ssim.update_state(ssim)
        if FLAGS.image_savepath is not None:
            out = np.uint8(out * FLAGS.max_pixel)
            out = Image.fromarray(out[0])
            out.save(FLAGS.image_savepath + str(i) + FLAGS.save_ext)

    eval_time = time.time() - start_time
    print("Testing Summary: Time(s):{:.0f} Average PSNR: {:.6f} Average SSIM: {:.6f}".format(eval_time,
                                                                         running_psnr.result(), running_ssim.result()))

    if(FLAGS.tflite_savepath is not None):
        print("Generating TFLite model")
        assert FLAGS.tflite_savepath.endswith(".tflite"), "The path should include the intended name of the file ending with \".tflite\""
        converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if(FLAGS.float16):
            converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        open(FLAGS.tflite_savepath, "wb").write(tflite_model)

if __name__ == '__main__':
    flags.mark_flag_as_required('data_path')
    flags.mark_flag_as_required('model_path')
    app.run(main)