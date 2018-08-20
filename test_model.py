from keras.models import load_model
from visualize import Vis
import numpy as np

class Test:
	def test(self,data):
		size_of_image=96
		test_frame = np.empty((1,size_of_image,size_of_image,1))
		model = load_model('models/model4_adam.h5')
		test_frame[0,:,:,0] = np.asarray(data).reshape(size_of_image,size_of_image) 
		return model.predict(test_frame)




#Uncomment the following code to see test results for pretrained model. The test results will be seen in images/visual_results
'''
x = Vis()
t=Test()
for j in range(len(x.result[2])):
	keypoints = t.test(x.result[2].iloc[j])[0]
	x.visual(x.result[2].iloc[j],keypoints,'test_'+str(j)+'.png')
'''