#https://arxiv.org/abs/1606.03657
from InfoGAN_Class import GAN

import tensorflow as tf #version 1.4
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt


saver_path = './saver/'
make_image_path = './generate/'

batch_size = 128
start_value = 2 #이미지 생성할때 continuous C 값을 -2~2 범위에서 변경하면서 생성하겠다는 의미임.

def train(model, data):
	total_D_loss = 0
	total_G_loss = 0
	total_Q_loss = 0

	iteration = int(np.ceil(len(data)/batch_size))


	for i in range( iteration ):
		#train set. mini-batch
		input_ = data[batch_size * i: batch_size * (i + 1)]

		#노이즈 생성.
		noise = model.Generate_noise(len(input_))  # len(input_) == batch_size, noise = (batch_size, model.noise_dim)
		latent_code = model.Generate_latent_code(len(input_)) # (batch, self.categorical + self.continuous)
			
		#Discriminator 학습.
		_, D_loss = sess.run([model.D_minimize, model.D_loss], {
						model.X:input_, model.noise_source:noise, model.latent_code:latent_code, model.is_train:True
					}
				)
		
		#Generator 학습. 		#batch_normalization을 하기 때문에 X data도 넣어줘야함.
		_, G_loss = sess.run([model.G_minimize, model.G_loss], {
						model.X:input_, model.noise_source:noise, model.latent_code:latent_code, model.is_train:True
					}
				)
		
		#Q 학습.
		_, Q_loss = sess.run([model.Q_minimize, model.Q_loss], {
						model.X:input_, model.noise_source:noise, model.latent_code:latent_code, model.is_train:True
					}
				)
		

		#parameter sum
		total_D_loss += D_loss
		total_G_loss += G_loss
		total_Q_loss += Q_loss
	

	return total_D_loss/iteration, total_G_loss/iteration, total_Q_loss/iteration



def write_tensorboard(model, D_loss, G_loss, Q_loss, epoch):
	summary = sess.run(model.merged, 
					{
						model.D_loss_tensorboard:D_loss, 
						model.G_loss_tensorboard:G_loss,
						model.Q_loss_tensorboard:Q_loss,
					}
				)

	model.writer.add_summary(summary, epoch)



def gen_image(model, epoch):
	if not os.path.exists(make_image_path):
		os.makedirs(make_image_path)
	
	num_generate = 10

	noise = model.Generate_noise(1) # noise = (num_generate, model.noise_dim)
	noise = np.tile(noise, [num_generate, 1])
	
	image = []
	c = np.linspace(-start_value, start_value, num_generate) # -start_value ~ start_value를 num_generate 등분.

	
	#0~9의 categorical 별로 각도, 너비 변환된 이미지 생성
	for k in range(0, num_generate):
		#category 변환
		latent_code = np.zeros([num_generate, 12])
		latent_code[:, k] = 1	
	

		#continuous code1 변환하여 이미지 생성
		latent_code[:, model.categorical] = c
		generated = sess.run(model.Gen, { #num_generate, 28, 28, 1
						model.noise_source:noise, model.latent_code:latent_code, model.is_train:False
					}
				) 
		generated = np.reshape(generated, (-1, model.height, model.width)) #이미지 형태로. #num_generate, height, width
		image.append(generated)
	


		#continuous code2 변환하여 이미지 생성
		latent_code[:, model.categorical] = [0]*num_generate
		latent_code[:, model.categorical+1] = c
	
		generated = sess.run(model.Gen, { #num_generate, 28, 28, 1
						model.noise_source:noise, model.latent_code:latent_code, model.is_train:False
					}
				) 
		generated = np.reshape(generated, (-1, model.height, model.width)) #이미지 형태로. #num_generate, height, width
		image.append(generated)
		


	#generate image
	fig, axes = plt.subplots(20, num_generate, figsize=(num_generate, 20))
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i in range(20):
		for j in range(num_generate):
			axes[i][j].set_axis_off()
			axes[i][j].imshow(image[i][j])
	
	plt.savefig(make_image_path+str(epoch))
	plt.close(fig)	



def run(model, train_set, restore = 0):
	#restore인지 체크.
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
	
	print('training start')

	#학습 진행
	for epoch in range(restore + 1, 2001):
		D_loss, G_loss, Q_loss = train(model, train_set)

		print("epoch : ", epoch, " D_loss : ", D_loss, " G_loss : ", G_loss, " Q_loss : ", Q_loss)

		
		if epoch % 5 == 0:
			#tensorboard
			write_tensorboard(model, D_loss, G_loss, Q_loss, epoch)

			#weight 저장할 폴더 생성
			if not os.path.exists(saver_path):
				os.makedirs(saver_path)
			save_path = model.saver.save(sess, saver_path+str(epoch)+".ckpt")
		
			#생성된 이미지 저장할 폴더 생성
			if not os.path.exists(make_image_path):
				os.makedirs(make_image_path)
			gen_image(model, epoch)




sess = tf.Session()

#model
model = GAN(sess) #noise_dim, input_dim

#get mnist data #이미지의 값은 0~1 사이임.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#mnist.train.images = 55000, 784 => train_set = 55000, 28, 28, 1 (batch, height, width, channel)
train_set =  np.reshape(mnist.train.images, ([-1, 28, 28, 1]))


#run
run(model, train_set)

