
"""
MNIST configuratons.
"""

config_mnist = {}
config_mnist['dataset'] = 'mnist'
config_mnist['datashape'] = [28, 28, 1]
config_mnist['verbose'] = True
config_mnist['save_every_epoch'] = 20
##print every
config_mnist['print_every'] = 100
config_mnist['work_dir'] = 'results_mnist'
config_mnist['plot_num_pics'] = 400
config_mnist['plot_num_cols'] = 20

config_mnist['input_normalize_sym'] = False
config_mnist['data_dir'] = 'mnist'

config_mnist['optimizer'] = 'adam'  # adam, sgd
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 1e-3
config_mnist['lr_adv'] = 1e-4
config_mnist['lr_schedule'] = 'plateau' 
##batch size
config_mnist['batch_size'] = 100
##epoch number
config_mnist['task_num'] = 5
config_mnist['epoch_num_per'] = 5
#Training the last tasks more epochs is to show the stabilization of the method.
config_mnist['epoch_num'] = [config_mnist['epoch_num_per']] * (config_mnist['task_num'] - 1) + [config_mnist['epoch_num_per']*4]
config_mnist['init_std'] = 0.01  
config_mnist['init_bias'] = 0.0
config_mnist['batch_norm'] = False
config_mnist['batch_norm_eps'] = 1e-5
config_mnist['batch_norm_decay'] = 0.9
config_mnist['conv_filters_dim'] = 4

config_mnist['e_pretrain'] = True
config_mnist['e_pretrain_sample_size'] = 1000
config_mnist['e_noise'] = 'add_noise'
config_mnist['e_num_filters'] = 256
config_mnist['e_num_layers'] = 3
config_mnist['e_arch'] = 'dcgan' # mlp, dcgan, ali

config_mnist['g_num_filters'] = 256
config_mnist['g_num_layers'] = 3
config_mnist['g_arch'] = 'dcgan_mod' # mlp, dcgan, dcgan_mod, ali

config_mnist['gan_p_trick'] = False
config_mnist['d_num_filters'] = 512
config_mnist['d_num_layers'] = 4

config_mnist['pz'] = 'normal' # uniform, normal, sphere
config_mnist['cost'] = 'l2sq' #l2, l2sq, l1
config_mnist['pz_scale'] = 1.
config_mnist['z_test'] = 'mmd'
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_mnist['lambda_schedule'] = 'constant'

# main network
if config_mnist['task_num'] > 3:
	#Reducing the number of zdim will improve the performance when addressing more tasks.
	config_mnist['zdim'] = 32
	#More parameters are needed for a lot of tasks.
	config_mnist['main_info'] = [784, 600, 600, 10]
else:
	config_mnist['zdim'] = 64
	config_mnist['main_info'] = [784, 100, 100, 10]

#
config_mnist['auxi_info'] = [0, 0, 80]
config_mnist['t_info'] = [1000, 1000]
# sample size after one task
config_mnist['z_size'] = 40000
# sample size during t
config_mnist['sample_size'] = 400
config_mnist['t_keep_prob'] = 0.7
config_mnist['main_keep_prob'] = 0.8

config_mnist['seed'] = [101, 202, 303, 70]
