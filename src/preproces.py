from PIL import Image, ImageFilter
import numpy as np

def resize_image(image, width, height):
    image = image.resize((width, height))
    return image

def day_night(image): #uses HSV arr
    mean=np.mean(image)

    return (1.0-mean/255)*100.0

def fog_detection(hist):
    total=sum(hist)
    dark=sum(hist[:31])/total

    suma=0
    mean=0
    while suma<total/2:
        suma+=hist[mean]
        mean+=1

    dark=(1.0-(dark/0.15))*50.0
    if dark<0:
        dark=0

    mean=mean/180.0*50.0
    if(mean>50):
        mean=50

    return dark+mean

def rain_detection(img):
    arr=np.array(img)

    img_blur=img.filter(ImageFilter.GaussianBlur(radius=5))

    arr_blur=np.array(img_blur)

    arr_noise=arr-arr_blur

    vert_diff = np.abs(arr[1:, :] - arr[:-1, :])

    rain_mask = (vert_diff > 100) & (arr_noise[:-1, :] > 100)

    rain_ratio = np.count_nonzero(rain_mask) / rain_mask.size

    rain_ratio=(rain_ratio/0.10)*100
    if rain_ratio>100:
        rain_ratio=100

    return rain_ratio

def snow_detection(arr):
    s=arr[:, :, 1]
    v=arr[:, :, 2]

    snow=(v>190) & (s<20)
    snow_ratio=np.count_nonzero(snow)/snow.size

    return (snow_ratio/0.20) * 100

def enhance_image(image):
    image=image.resize((1280,1280))

    img_HSV=image.convert('HSV')
    arr_hsv=np.array(img_HSV)

    img_gray=image.convert('L')
    histogram=img_gray.histogram()
    arr_gray=np.array(img_gray)

    day=day_night(arr_hsv[:, :, 2])
    fog=fog_detection(histogram)
    rain=rain_detection(img_gray)
    snow=snow_detection(arr_hsv)

    print(day, fog, rain, snow)

if "__main__" == __name__:
    #im=Image.open("../images/test/Fog/Fog/haze-013.jpg")

    im = Image.open("../images/test/Rain/Rain/rain_storm-700.jpg")

    #im=Image.open("../images/test/Snow/Snow/snow_storm-001.jpg")

    enhance_image(im)

    im.show()
