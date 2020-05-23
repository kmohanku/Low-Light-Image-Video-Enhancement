import tensorflow as tf

class SpatialConstancyLoss(tf.Module):
    def __init__(self, weighting=1.0):
        super(SpatialConstancyLoss, self).__init__()
        self.weight_left = tf.Variable([[[[0]],[[0]],[[0]]],[[[-1]],[[1]],[[0]]],[[[0]],[[0]],[[0]]]],
                                       trainable=False, dtype=tf.float32)
        self.weight_right = tf.Variable([[[[0]],[[0]],[[0]]],[[[0]],[[1]],[[-1]]],[[[0]],[[0]],[[0]]]],
                                        trainable=False, dtype=tf.float32)
        self.weight_up = tf.Variable([[[[0]],[[-1]],[[0]]],[[[0]],[[1]],[[0]]],[[[0]],[[0]],[[0]]]],
                                        trainable=False, dtype=tf.float32)
        self.weight_down = tf.Variable([[[[0]],[[0]],[[0]]],[[[0]],[[1]],[[0]]],[[[0]],[[-1]],[[0]]]],
                                        trainable=False, dtype=tf.float32)
        self.weighting = weighting
        
    def __call__(self, original, enhanced):
        b,h,w,c = original.shape
        org_mean = tf.math.reduce_mean(original, axis=3, keepdims=True)
        enhanced_mean = tf.math.reduce_mean(enhanced, axis=3, keepdims=True)
        original_pool = tf.nn.avg_pool2d(org_mean, ksize=[1,4,4,1], strides=4, padding='VALID')
        enhanced_pool = tf.nn.avg_pool2d(enhanced_mean, ksize=[1,4,4,1], strides=4, padding='VALID')
        weight_diff = tf.maximum(1.0 + 10000 * tf.minimum(original_pool - 0.3, 0.0), 0.5)
        E_1 = tf.multiply(tf.math.sign(enhanced_pool - 0.5), enhanced_pool - original_pool)
        
        D_orig_left = tf.nn.conv2d(input=original_pool,filters=self.weight_left, strides=[1,1,1,1], padding="SAME")
        D_orig_right = tf.nn.conv2d(input=original_pool,filters=self.weight_right, strides=[1,1,1,1], padding="SAME")
        D_orig_up = tf.nn.conv2d(input=original_pool,filters=self.weight_up, strides=[1,1,1,1], padding="SAME")
        D_orig_down = tf.nn.conv2d(input=original_pool,filters=self.weight_down, strides=[1,1,1,1], padding="SAME")
        
        D_enhance_left = tf.nn.conv2d(input=enhanced_pool,filters=self.weight_left, strides=[1,1,1,1], padding="SAME")
        D_enhance_right = tf.nn.conv2d(input=enhanced_pool,filters=self.weight_right, strides=[1,1,1,1], padding="SAME")
        D_enhance_up = tf.nn.conv2d(input=enhanced_pool,filters=self.weight_up, strides=[1,1,1,1], padding="SAME")
        D_enhance_down = tf.nn.conv2d(input=enhanced_pool,filters=self.weight_down, strides=[1,1,1,1], padding="SAME")
        
        D_left = tf.math.pow(D_orig_left - D_enhance_left, 2)
        D_right = tf.math.pow(D_orig_right - D_enhance_right, 2)
        D_up = tf.math.pow(D_orig_up - D_enhance_up, 2)
        D_down = tf.math.pow(D_orig_down - D_enhance_down, 2)
        error = self.weighting * (D_left + D_right + D_up + D_down)
        return error

def colorLoss(tensor):
    b,h,w,c = tensor.shape
    meanRGB = tf.math.reduce_mean(tensor, axis=[1,2], keepdims=True)
    meanR, meanG, meanB = tf.split(meanRGB, num_or_size_splits=3, axis=3)
    dRG = tf.math.pow(meanR - meanG, 2)
    dRB = tf.math.pow(meanR - meanB, 2)
    dGB = tf.math.pow(meanB - meanG, 2)
    loss = tf.math.pow((tf.math.pow(dRG, 2) + tf.math.pow(dRB, 2) + tf.math.pow(dGB, 2)), 0.5)
    return loss

def exposureLoss(tensor, patchSize, meanVal):
    b,h,w,c = tensor.shape
    tensor = tf.math.reduce_mean(tensor, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(tensor, ksize=patchSize, strides=patchSize, padding='VALID')
    loss = tf.math.reduce_mean(tf.math.pow(mean - meanVal, 2))
    return loss

def totalLoss(tensor, tloss_weight=1):
    batch_size = tensor.shape[0]
    h_x = tensor.shape[1]
    w_x = tensor.shape[2]
    count_h = (h_x - 1) * w_x
    count_w = h_x * (w_x - 1)
    h_tv = tf.math.reduce_sum(tf.math.pow((tensor[:,1:,:,:] - tensor[:,:h_x - 1,:,:]),2))
    w_tv = tf.math.reduce_sum(tf.math.pow((tensor[:,:,1:,:] - tensor[:,:,:w_x - 1,:]),2))
    return tloss_weight * 2 * (h_tv/count_h + w_tv/count_w)/batch_size

def computeQuartretLoss(input_image, enhanced_image, curves, spaCon_weight=1.0, patchSize=16, meanVal=0.6):
    curve_loss = 200 * totalLoss(curves)
    spaConstLoss = SpatialConstancyLoss(spaCon_weight)
    spatial_loss = tf.math.reduce_mean(spaConstLoss(enhanced_image, input_image))
    color_loss = 5 * tf.math.reduce_mean(colorLoss(enhanced_image))
    exposure_loss = 10 * tf.math.reduce_mean(exposureLoss(enhanced_image, patchSize, meanVal))
    total_loss = curve_loss + spatial_loss + color_loss + exposure_loss
    return total_loss