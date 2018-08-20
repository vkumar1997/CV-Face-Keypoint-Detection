import cv2
import matplotlib.pyplot as plt

def canny_edge_det(image_path,kernel_size=0,sigma=0):
    image = cv2.imread(image_path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      
    title = ''
    
    #gaussian blur image
    if kernel_size is not 0 or sigma is not 0:
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        title = 'Blurred Image kernel size = ' + str(kernel_size) + ' sigma = ' + str(sigma)
    else:
        title = 'Original Image without blur'

    #apply canny edge filter and dilate to observe edges
    edges = cv2.Canny(image,100,200)
    edges = cv2.dilate(edges, None)

    #plot input image
    plt.figure(figsize = (15,15))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title(title)

    #plot edges
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Edge Detection')
    plt.show()

canny_edge_det('image/house.jpg')