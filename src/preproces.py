from PIL import Image, ImageFilter, ImageChops, ImageEnhance, ImageOps
import numpy as np
import os
import scipy.ndimage

def resize_image(image, width, height):
    image = image.resize((width, height))
    return image

def day_night(image):
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

    snow=(v>190) & (s<40)
    snow_ratio=np.count_nonzero(snow)/snow.size

    return min((snow_ratio/0.1) * 100, 100)

def snd_detection(arr):
    h = arr[:, :, 0]
    s = arr[:, :, 1]
    v = arr[:, :, 2]
    sand_mask = (h >= 10) & (h <= 42) & (s > 50) & (v > 100)

    sand_ratio = np.count_nonzero(sand_mask) / sand_mask.size

    sand_percentage = (sand_ratio / 0.20) * 100.0
    return min(100.0, sand_percentage)

def weather_detection(image):
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
    sand=snd_detection(arr_hsv)

    if day<50:
        is_day=1
    else:
        is_day=0


    if fog>90:
        snow=snow-fog*0.05
    scores = {
        0: 50,
        4: sand,
        1: fog,
        3: snow,
        2: rain
    }

    best_weather = max(scores, key=scores.get)

    return is_day, best_weather

def test():
    file = open("../images/preprocesare/data", "r")
    data=file.read()
    data=data.split("\n")

    w_day = 0
    w_rain = 0
    w_snow = 0
    w_sand = 0
    w_fog = 0
    w_clean=0

    total = -1
    rain = 0
    snow = 0
    sand = 0
    fog = 0
    clean=0

    f = ["../images/preprocesare/test/Fog/Fog",
         "../images/preprocesare/test/Rain/Rain",
         "../images/preprocesare/test/Sand/Sand",
         "../images/preprocesare/test/Snow/Snow",
         "../images/preprocesare/test/clean"]

    for i in f:
        for e in os.scandir(i):
            if e.is_file():
                try:
                    total += 1
                    if data[total]!= "-":
                        data[total]=data[total].split(" ")
                        day_i=int(data[total][0])
                        weather_i=int(data[total][1])


                        im = Image.open(e.path)
                        #im.show()
                        day, weather = weather_detection(im)

                        if day == day_i:
                            w_day += 1

                        if weather_i == 1:
                            fog += 1
                        elif weather_i == 2:
                            rain += 1
                        elif weather_i == 3:
                            snow += 1
                        elif weather_i == 4:
                            sand += 1
                        else:
                            clean+=1

                        if weather == weather_i:
                            if weather_i == 1:
                                w_fog += 1
                            elif weather_i == 2:
                                w_rain += 1
                            elif weather_i == 3:
                                w_snow += 1
                            elif weather_i == 4:
                                w_sand += 1
                            else:
                                w_clean += 1

                        # if weather != weather_i:
                        #     im.show()
                        #     print(f"{weather_i}: {weather}")
                        #
                        #     image = im.resize((1280, 1280))
                        #
                        #     img_HSV = image.convert('HSV')
                        #     arr_hsv = np.array(img_HSV)
                        #
                        #     img_gray = image.convert('L')
                        #     histogram = img_gray.histogram()
                        #     arr_gray = np.array(img_gray)
                        #
                        #     dayp = day_night(arr_hsv[:, :, 2])
                        #     fogp= fog_detection(histogram)
                        #     rainp = rain_detection(img_gray)
                        #     snowp = snow_detection(arr_hsv)
                        #     sandp = snd_detection(arr_hsv)
                        #
                        #     print(f"{total} {dayp} {fogp} {rainp} {snowp} {sandp}")
                        #     input("Press Enter to continue...")
                        #     im.close()

                        im.close()
                except(Exception) as e:
                    pass

    print(f"day={w_day / total * 100}")
    print(f"rain={w_rain / rain * 100}")
    print(f"snow={w_snow / snow * 100}")
    print(f"fog={w_fog / fog * 100}")
    print(f"sand={w_sand / sand * 100}")
    print(f"clean={w_clean / clean * 100}")
    print(f"total{(clean + w_rain + w_fog + w_snow + w_sand) / total * 100}")
    file.close()

def fog_enhancing(img):
    equal=ImageOps.equalize(img)
    contrast=ImageEnhance.Contrast(equal)

    result=contrast.enhance(0.7)

    return result

def rain_removal(img):
    result=img.filter(ImageFilter.MedianFilter(9))

    out=fog_enhancing(result)

    return out

def enhance(img):
    day, weather = weather_detection(img)

    if weather==1:
        out=fog_enhancing(img)
    elif weather==2:
        out=rain_removal(img)
    elif weather==4:
        out=fog_enhancing(img)
    else:
        out=img

    return out

if "__main__" == __name__:
    img=Image.open("../images/preprocesare/test/Rain/Rain/rain_storm-002.jpg")

    defog=rain_removal(img)

    img.show()
    defog.show()