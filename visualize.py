
import cv2
import matplotlib.pyplot as plt
from arrange_images_for_training import arrange_images
import numpy as np


class Vis:
	def __init__(self):
		x = arrange_images()
		self.result = x.prepare()

	def visual(self,pixels,keypoints,name):
		size_of_image =96
		fig, ax = plt.subplots()
		pixels = np.asarray(pixels)
		pixels = pixels.reshape(size_of_image,size_of_image)
		ax.imshow(pixels*255,cmap='gray')

		keypoints = list(keypoints)
		for i in range(0,len(keypoints)-1,2):
		    ax.plot([keypoints[i] * size_of_image],[keypoints[i+1] * size_of_image], marker='o', markersize=3, c = 'red')
		plt.savefig('image/visual_result/'+name)
		plt.close()
		print(name)


#x = Vis()
#x.visual(x.result[0].iloc[0],x.result[1].iloc[0],'train_1.png')
