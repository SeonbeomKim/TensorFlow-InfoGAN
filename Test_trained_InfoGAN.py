#https://arxiv.org/abs/1606.03657

from InfoGAN_Class import GAN
import tensorflow as tf #version 1.4
import numpy as np
import os
import matplotlib.pyplot as plt


saver_path = './saver/'
make_image_path = './testcode_generate/'

restore = 315
start_value = 1 ## == 2~-2


def gen_image(model, epoch):
	if not os.path.exists(make_image_path):
		os.makedirs(make_image_path)
	
	num_generate = 10

	noise = model.Generate_noise(1) # noise = (num_generate, model.noise_dim)
	noise = np.tile(noise, [num_generate, 1])
	
	image = []
	
	c = np.linspace(start_value, -start_value, num_generate) # start_value ~ -start_value를 num_generate 등분.
	
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



sess = tf.Session()

#model
model = GAN(sess) 
model.saver.restore(sess, saver_path+str(restore)+".ckpt")

gen_image(model, restore)

