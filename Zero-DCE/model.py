import tensorflow as tf

# Simple convolution block with a non linear activation applied
class ConvBlock(tf.Module):
    def __init__(self, ins, 
                 outs=32, 
                 activation_fn=tf.nn.relu, 
                 kernelSize=3, 
                 kernelStride=1, 
                 kernelPadding="SAME", 
                 scope=None
                ):
        super().__init__(name=scope)
        self.activation_fn = activation_fn
        self.kernelStride = kernelStride
        self.padding = kernelPadding
        self.kernel = tf.Variable(tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)
                                  (shape=[kernelSize, kernelSize, ins, outs], dtype=tf.float32),
                                  name=scope + "_weights")
        self.bias = tf.Variable(tf.zeros_initializer()(shape=[outs], dtype=tf.float32), name=scope+"_bias")
        
    @tf.function
    def __call__(self, inputTensor):
        x = tf.nn.conv2d(input=inputTensor, filters=self.kernel, strides=[1, self.kernelStride, self.kernelStride, 1], padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        if(self.activation_fn):
            x = self.activation_fn(x)
        return x, x

# Chain of convolution blocks with residuals for the second half
class DCENet(tf.Module):
    def __init__(self, numConvs=7,
                 blockSize=32, 
                 iterations=8, 
                 activation_fn=tf.nn.relu, 
                 final_activation_fn=tf.nn.tanh, 
                 kernelSize=3,
                 kernelStride=1,
                 kernelPadding="SAME"
                ):
        super(DCENet, self).__init__()
        self.numConvs = numConvs
        self.forwardPass = []
        self.iterations = iterations
        self.final_activation_fn = final_activation_fn
        outs = blockSize
        ins = 3
        for i in range(1, numConvs):
            if i > ((numConvs - 1)//2 + 1):
                ins = outs * 2
            elif(i != 1):
                ins = outs
            convlayer = ConvBlock(ins, outs, activation_fn=activation_fn, kernelSize=kernelSize, 
                                  kernelStride=kernelStride, kernelPadding=kernelPadding, scope="convLayer"+str(i))
            self.forwardPass.append(convlayer)
        finalconv = ConvBlock(outs * 2, iterations * 3, activation_fn=final_activation_fn, kernelSize=kernelSize, 
                                  kernelStride=kernelStride, kernelPadding=kernelPadding, scope="convLayer"+str(numConvs))
        self.forwardPass.append(finalconv)
        
    @tf.function
    def __call__(self, x):
        inputTensor = tf.identity(x)
        residuals = []
        enhanced_image_intermediate = None
        back_iter = -1
        for i, layer in enumerate(self.forwardPass, start=1):
            residual, x = layer(x)
            if(i < (self.numConvs - 1)//2 + 1):
                residuals.append(residual)
            elif(i >= (self.numConvs - 1)//2 + 1 and i!=self.numConvs):
                x = tf.concat([x, residuals[back_iter]], axis=3)
                back_iter-=1
        x = self.final_activation_fn(x)
        curves = tf.split(x, num_or_size_splits=self.iterations, axis=3)
        for curve in range(len(curves)):
            inputTensor += curves[curve] * (tf.math.pow(inputTensor, 2) - inputTensor)
            if(curve == len(curves)//2):
                enhanced_image_intermediate = tf.identity(inputTensor)
        curves = tf.concat(curves, axis=3)
        return enhanced_image_intermediate, inputTensor, curves  