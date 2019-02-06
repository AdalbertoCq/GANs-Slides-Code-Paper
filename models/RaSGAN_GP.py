import tensorflow as tf
from models.generative.ops import *
from models.generative.utils import *
from models.generative.loss import *
from models.generative.optimizer import *


class RaSGAN_GP:
	def __init__(self,
				data,                        										# Dataset class, training and test data.
				z_dim,                       										# Latent space dimensions.
				use_bn,                      										# Batch Normalization flag to control usage in discriminator.
				alpha,                       										# Alpha value for LeakyReLU.
				beta_1,                     		 								# Beta 1 value for Adam Optimizer.
				beta_2,                 	 										# Beta 2 value for Adam Optimizer.
				n_critic,                  	 										# Number of batch gradient iterations in Discriminator per Generator.
				gp_coeff,                    										# Gradient Penalty coefficient for the Wasserstein Gradient Penalty loss.
				learning_rate_g,               										# Learning rate of the Generator.
				learning_rate_d,               										# Learning rate of the Discriminator.
				loss_type='relativistic gradient penalty',     						# Loss function type: Standard, Least Square, Wasserstein, Wasserstein Gradient Penalty.
				model_name='RaSGAN_GP'         										# Name to give to the model.
				):

		self.model_name = model_name

		# Input data variables.
		self.image_height = data.training.patch_h
		self.image_width = data.training.patch_w
		self.image_channels = data.training.n_channels
		self.batch_size = data.training.batch_size

		# Latent space dimensions.
		self.z_dim = z_dim

		# Loss function definition.
		self.loss_type = loss_type

		# Network details
		self.use_bn = use_bn
		self.alpha = alpha

		# Training parameters
		self.n_critic = n_critic
		self.gp_coeff = gp_coeff
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.learning_rate_g = learning_rate_g
		self.learning_rate_d = learning_rate_d

		self.build_model()


	def discriminator(self, images, reuse):
		with tf.variable_scope('discriminator', reuse=reuse):
			# Padding = 'Same' -> H_new = H_old // Stride

			# Input Shape = (None, 448, 448, 3)
			
			# Conv.
			# net = convolutional(inputs=images, output_channels=16, filter_size=5, stride=2, padding='SAME', conv_type='convolutional')
			# net = tf.layers.conv2d(inputs=images, filters=16, kernel_size=(5,5), strides=(2, 2), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
			# net = leakyReLU(net, self.alpha)
			# Shape = (None, 224, 224, 16)

			# Conv.
			net = convolutional(inputs=images, output_channels=32, filter_size=5, stride=2, padding='SAME', conv_type='convolutional', scope=1)
			if self.use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 112, 112, 32)

			# Conv.
			net = convolutional(inputs=net, output_channels=64, filter_size=5, stride=2, padding='SAME', conv_type='convolutional', scope=2)
			if self.use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 56, 56, 64)

			# Conv.
			net = convolutional(inputs=net, output_channels=128, filter_size=5, stride=2, padding='SAME', conv_type='convolutional', scope=3)
			if self.use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 28, 28, 128)

			# Conv.
			net = convolutional(inputs=net, output_channels=256, filter_size=5, stride=2, padding='SAME', conv_type='convolutional', scope=4)
			if self.use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 14, 14, 256)

			# Conv.
			net = convolutional(inputs=net, output_channels=512, filter_size=5, stride=2, padding='SAME', conv_type='convolutional', scope=5)
			if self.use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 7, 7, 512)

			# Flatten.
			net = tf.layers.flatten(inputs=net)
			# Shape = (None, 7*7*512)

			# Dense.
			net = dense(inputs=net, out_dim=1024, scope=1)
			if self.use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 1024)

			# Dense
			logits = dense(inputs=net, out_dim=1, scope=2)		
			# Shape = (None, 1)
			output = tf.nn.sigmoid(x=logits)

			# Padding = 'Same' -> H_new = H_old // Stride

		return output, logits

	def generator(self, z_input, reuse, is_train):
		with tf.variable_scope('generator', reuse=reuse):
			# Doesn't work ReLU, tried.
			# Input Shape = (None, 100)
			# Dense.
			net = dense(inputs=z_input, out_dim=1024, scope=1)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 1024)

			# Dense.
			net = dense(inputs=net, out_dim=256*7*7, scope=2)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 256*7*7)

			# Reshape
			net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')
			# Shape = (None, 7, 7, 256)

			# Conv.
			net = convolutional(inputs=net, output_channels=256, filter_size=2, stride=2, padding='SAME', conv_type='transpose', scope=1)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 14, 14, 256)

			# Conv.
			net = convolutional(inputs=net, output_channels=128, filter_size=5, stride=1, padding='SAME', conv_type='convolutional', scope=2)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 14, 14, 128)

			# Conv.
			net = convolutional(inputs=net, output_channels=128, filter_size=2, stride=2, padding='SAME', conv_type='transpose', scope=3)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 28, 28, 128)

			# Conv.
			net = convolutional(inputs=net, output_channels=64, filter_size=5, stride=1, padding='SAME', conv_type='convolutional', scope=4)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 28, 28, 64)

			# Conv.
			net = convolutional(inputs=net, output_channels=64, filter_size=2, stride=2, padding='SAME', conv_type='transpose', scope=5)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 56, 56, 64)

			# Conv.
			net = convolutional(inputs=net, output_channels=32, filter_size=5, stride=1, padding='SAME', conv_type='convolutional', scope=6)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 56, 56, 32)

			# Conv.
			net = convolutional(inputs=net, output_channels=32, filter_size=2, stride=2, padding='SAME', conv_type='transpose', scope=7)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 112, 112, 32)

			# Conv.
			# net = convolutional(inputs=net, output_channels=16, filter_size=5, stride=1, padding='SAME', conv_type='transpose')
			# net = tf.layers.batch_normalization(inputs=net, training=is_train)
			# net = leakyReLU(net, self.alpha)
			# Shape = (None, 112, 112, 16)

			# Conv.
			# net = convolutional(inputs=net, output_channels=16, filter_size=2, stride=2, padding='SAME', conv_type='upscale')
			# net = tf.layers.conv2d_transpose(inputs=net, filters=16, kernel_size=(2,2), strides=(2,2), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
			# net = tf.layers.batch_normalization(inputs=net, training=is_train)
			# net = leakyReLU(net, self.alpha)
			# Shape = (None, 224, 224, 16)

			# Conv.
			logits = convolutional(inputs=net, output_channels=self.image_channels, filter_size=2, stride=2, padding='SAME', conv_type='transpose', scope=8)
			# Shape = (None, 448, 448, 3)
			output = tf.nn.sigmoid(x=logits, name='output')
		return output


	def model_inputs(self):
		real_images = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images')
		z_input = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='z_input')
		learning_rate_g = tf.placeholder(dtype=tf.float32, name='learning_rate_g')
		learning_rate_d = tf.placeholder(dtype=tf.float32, name='learning_rate_d')
		return real_images, z_input, learning_rate_g, learning_rate_d


	def loss(self):
		loss_dis, loss_gen = losses(self.loss_type, self.output_fake, self.output_real, self.logits_fake, self.logits_real, 
									real_images=self.real_images, fake_images=self.fake_images, discriminator=self.discriminator, 
									batch_size=self.batch_size, gp_coeff=self.gp_coeff)
		return loss_dis, loss_gen


	def optimization(self):
		train_discriminator, train_generator = optimizer(self.beta_1, self.loss_gen, self.loss_dis, self.loss_type, self.learning_rate_input_g, self.learning_rate_input_d, beta_2=self.beta_2)
		return train_discriminator, train_generator
    

	def build_model(self):
		# Inputs.
		self.real_images, self.z_input, self.learning_rate_input_g, self.learning_rate_input_d = self.model_inputs()

		# Neural Network Generator and Discriminator.
		self.fake_images = self.generator(self.z_input, reuse=False, is_train=True)

		self.output_fake, self.logits_fake = self.discriminator(images=self.fake_images, reuse=False) 
		self.output_real, self.logits_real = self.discriminator(images=self.real_images, reuse=True)

		# Losses.
		self.loss_dis, self.loss_gen = self.loss()

		# Optimizers.
		self.train_discriminator, self.train_generator = self.optimization()

		self.output_gen = self.generator(self.z_input, reuse=True, is_train=False)


	def train(self, epochs, data_out_path, data, restore, show_epochs=100, print_epochs=10, n_images=10, save_img=False):
		run_epochs = 0    
		losses = list()
		saver = tf.train.Saver()

		img_storage, latent_storage, checkpoints = setup_output(show_epochs, epochs, data, n_images, self.z_dim, data_out_path, self.model_name, restore, save_img)

		with tf.Session() as session:
		    session.run(tf.global_variables_initializer())
		    if restore:
		    	check = get_checkpoint(data_out_path)
		    	saver.restore(session, check)
		    	print('Restored model: %s' % check)
		    for epoch in range(1, epochs+1):
		        for batch_images, batch_labels in data.training:
		            # Inputs.
		            z_batch = np.random.uniform(low=-1., high=1., size=(self.batch_size, self.z_dim))               
		            feed_dict = {self.z_input:z_batch, self.real_images:batch_images, self.learning_rate_input_g: self.learning_rate_g, self.learning_rate_input_d: self.learning_rate_d}

		            # Update critic.
		            session.run(self.train_discriminator, feed_dict=feed_dict)

		            # Update generator after n_critic updates from discriminator.
		            if run_epochs%self.n_critic :
		            	session.run(self.train_generator, feed_dict=feed_dict)

		            # Print losses and Generate samples.
		            if run_epochs % print_epochs == 0:
		                feed_dict = {self.z_input:z_batch, self.real_images:batch_images}
		                epoch_loss_dis, epoch_loss_gen = session.run([self.loss_dis, self.loss_gen], feed_dict=feed_dict)
		                losses.append((epoch_loss_dis, epoch_loss_gen))
		                print('Epochs %s/%s: Generator Loss: %s. Discriminator Loss: %s' % (epoch, epochs, np.round(epoch_loss_gen, 4), np.round(epoch_loss_dis, 4)))
		            if run_epochs % show_epochs == 0:
		                gen_samples, sample_z = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, output_fake=self.output_gen, n_images=n_images, dim=30)
		                if save_img:
			                img_storage[run_epochs//show_epochs] = gen_samples
			                latent_storage[run_epochs//show_epochs] = sample_z
		                saver.save(sess=session, save_path=checkpoints, global_step=run_epochs)

		            run_epochs += 1
		        data.training.reset()

		return losses

	

