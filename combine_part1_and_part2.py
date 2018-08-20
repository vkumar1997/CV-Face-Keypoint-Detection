import cv2
import numpy as np
from face_and_eyes_haar import Haar
from test_model import Test
import matplotlib.pyplot as plt

def combine(image_path,scale_face,scale_eye):
	size_of_image =96
	h = Haar()
	faces = h.face_and_eyes(image_path,scale_face,scale_eye,20)
	plt.figure(figsize=(30,15))
	img_c = cv2.imread(image_path)
	for (x,y,w,h) in faces:
		img = cv2.imread(image_path)
		x_scale_factor = float(w/size_of_image)
		y_scale_factor = float(h/size_of_image)

		crop_img = img[y:y+h, x:x+w]
		crop_img = cv2.resize(crop_img,(size_of_image,size_of_image), interpolation=cv2.INTER_CUBIC)
		crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
		crop_img = crop_img/255
		t = Test()
		keypoints = t.test(crop_img)[0]
		#visualize(crop_img,keypoints)

		cv2.rectangle(img_c, (x, y), (x+w, y+h), (0, 255, 0), 2)
		plt.subplot(121)
		plt.title('Keypoints'), plt.xticks([]), plt.yticks([]) 
		for i in range(0,len(keypoints)-1,2):
			plt.plot([keypoints[i] * size_of_image * x_scale_factor + x],[keypoints[i+1] * size_of_image * y_scale_factor + y], marker='o', markersize=2, c = 'red')

	plt.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
	plt.show()

#combine('image/facebook.jpg',1.2,1.07)