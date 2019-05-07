import numpy as np
import tensorflow as tf
import ops


def encoder(opts, inputs, reuse=False, is_training=False):

    if opts['e_noise'] == 'add_noise':
        # Particular instance of the implicit random encoder
        def add_noise(x):
            shape = tf.shape(x)
            return x + tf.truncated_normal(shape, 0.0, 0.01)
        def do_nothing(x):
            return x
        inputs = tf.cond(is_training,
                         lambda: add_noise(inputs), lambda: do_nothing(inputs))

    with tf.variable_scope("encoder", reuse=reuse):
        res = dcgan_encoder(opts, inputs, is_training, reuse)

        noise_matrix = None


        return res, noise_matrix

def decoder(opts, noise, reuse=False, is_training=True):

    with tf.variable_scope("generator", reuse=reuse):
        res = dcgan_decoder(opts, noise, is_training, reuse)

        return res

def dcgan_encoder(opts, inputs, is_training=False, reuse=False):
    num_units = opts['e_num_filters']
    num_layers = opts['e_num_layers']
    layer_x = inputs
    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)
        layer_x = ops.conv2d(opts, layer_x, num_units / scale,
                             scope='h%d_conv' % i)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x, is_training,
                                     reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    res = ops.linear(opts, layer_x, opts['zdim'], scope='hfinal_lin')
    return res




def dcgan_decoder(opts, noise, is_training=False, reuse=False):
    output_shape = opts['datashape']
    num_units = opts['g_num_filters']
    batch_size = tf.shape(noise)[0]
    num_layers = opts['g_num_layers']
    
    height = int(output_shape[0] / 2**(num_layers - 1))
    width = int(output_shape[1] / 2**(num_layers - 1))

    h0 = ops.linear(
        opts, noise, num_units * height * width, scope='h0_lin')
    h0 = tf.reshape(h0, [-1, height, width, num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, height * scale,
                      width * scale, int(num_units / scale)]
        layer_x = ops.deconv2d(opts, layer_x, _out_shape,
                               scope='h%d_deconv' % i)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x,
                                     is_training, reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)

    last_h = ops.deconv2d(
            opts, layer_x, _out_shape, d_h=1, d_w=1, scope='hfinal_deconv')
    
    return tf.nn.sigmoid(last_h), last_h



def z_adversary(opts, inputs, reuse=False):
    num_units = opts['d_num_filters']
    num_layers = opts['d_num_layers']
    nowozin_trick = opts['gan_p_trick']
    # No convolutions as GAN happens in the latent space
    with tf.variable_scope('z_adversary', reuse=reuse):
        hi = inputs
        for i in range(num_layers):
            hi = ops.linear(opts, hi, num_units, scope='h%d_lin' % (i + 1))
            hi = tf.nn.relu(hi)
        hi = ops.linear(opts, hi, 1, scope='hfinal_lin')
        if nowozin_trick:
            # We are doing GAN between our model Qz and the true Pz.
            # Imagine we know analytical form of the true Pz.
            # The optimal discriminator for D_JS(Pz, Qz) is given by:
            # Dopt(x) = log dPz(x) - log dQz(x)
            # And we know exactly dPz(x). So add log dPz(x) explicitly 
            # to the discriminator and let it learn only the remaining
            # dQz(x) term. This appeared in the AVB paper.
            assert opts['pz'] == 'normal', \
                'The GAN Pz trick is currently available only for Gaussian Pz'
            sigma2_p = float(opts['pz_scale']) ** 2
            normsq = tf.reduce_sum(tf.square(inputs), 1)
            hi = hi - normsq / 2. / sigma2_p \
                    - 0.5 * tf.log(2. * np.pi) \
                    - 0.5 * opts['zdim'] * np.log(sigma2_p)
    return hi


def transform_noise(opts, code, eps):
    hi = code
    T = 3
    for i in range(T):
        # num_units = max(opts['zdim'] ** 2 / 2 ** (T - i), 2)
        num_units = max(2 * (i + 1) * opts['zdim'], 2)
        hi = ops.linear(opts, hi, num_units, scope='eps_h%d_lin' % (i + 1))
        hi = tf.nn.tanh(hi)
    A = ops.linear(opts, hi, opts['zdim'] ** 2, scope='eps_hfinal_lin')
    A = tf.reshape(A, [-1, opts['zdim'], opts['zdim']])
    eps = tf.reshape(eps, [-1, 1, opts['zdim']])
    res = tf.matmul(eps, A)
    res = tf.reshape(res, [-1, opts['zdim']])
    return res, A
    # return ops.linear(opts, hi, opts['zdim'] ** 2, scope='eps_hfinal_lin')
