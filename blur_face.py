
import cv2
import matplotlib.pyplot as plt


def face_and_eyes_haar(image_path,scaleFF,scaleFE,size):
    sample_image = cv2.imread(image_path) 
    gray_img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    path = 'image/' # path to  XML file
    plt.figure(figsize=(size,size))
    plt.subplot(121),plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    
    
    haar_face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_alt.xml')
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=scaleFF, minNeighbors=5)
    print('Faces found: ', len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(sample_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
   
    haar_eye_cascade = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')
    eyes = haar_eye_cascade.detectMultiScale(gray_img, scaleFactor=scaleFE, minNeighbors=5)
    print('Eyes found: ', len(eyes))
    for (x, y, w, h) in eyes:
        cv2.rectangle(sample_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
   
    plt.subplot(122),plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
    plt.title('Face and Eye Detection'), plt.xticks([]), plt.yticks([])
    
    plt.show()
    return faces


faces = face_and_eyes_haar('image/accused.jpg',1.2,1.2,20)
for (x,y,w,h) in faces:
    samp_image = cv2.imread('image/accused.jpg')
    blur_image = cv2.imread('image/accused.jpg')
    crp = samp_image[y:y+h, x:x+w]
    crp = cv2.GaussianBlur(crp, (25, 25), 10)
    blur_image[y:y+h,x:x+w] = crp
    
#Plot figure 1: Input
plt.figure(figsize=(20,20))
plt.subplot(121),plt.imshow(cv2.cvtColor(samp_image, cv2.COLOR_BGR2RGB))
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

#Plot figure 2:Blurred
plt.subplot(122),plt.imshow(cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB))
plt.title('Blurred Face'), plt.xticks([]), plt.yticks([])
plt.show()