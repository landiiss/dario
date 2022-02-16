from datetime import datetime, timedelta
from time import sleep
from fractions import Fraction
from orbit import ISS
import csv
from picamera import PiCamera
from pathlib import Path
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
import os

# Defining directories & models
base_folder = Path(__file__).parent.resolve()
img_folder = base_folder/'images'
night_img_folder = img_folder/'night_images'
ml_file = base_folder/'models/model_edgetpu.tflite'
label_file = base_folder/'models/labels.txt' 
#--------------  
data_dir = base_folder/'data'
#--------------

# Setting variables
start_time = datetime.now()
now_time = datetime.now()
img_number = 1
sleep_time_day = 20
sleep_time_night = 15
sleep_time = sleep_time_day
total_obs_time = 60

# Defining camera settings
camera = PiCamera()
camera.resolution = (1296,972) 

def check_img(image):
# check_img() checks if the image is a day-time picture or a night-time picture
# It uses a machine learning file created from "https://teachablemachine.withgoogle.com/" to check if the image is a day-time or a night-time picture
# It returns "True" if it's a day-time picture and "False" if it is a night-time one
    image_file = img_folder/image
    interpreter = make_interpreter(f"{ml_file}")
    interpreter.allocate_tensors()
    size = common.input_size(interpreter)
    image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)
    labels = read_label_file(label_file)
    for c in classes:
        print(f'{labels.get(c.id, c.id)} {c.score:.5f}')
        status = str({labels.get(c.id, c.id)})
        status = status.replace("{'", "")
        status = status.replace("'}", "")
        if status == "day":
            return True
        else:
            return False

# get_time() returns current time
def get_time():
    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")
    return now

# track_iss() returns ISS coordinates
def track_iss():
    location = ISS.coordinates() 
    latitude = location.latitude.degrees
    longitude = location.longitude.degrees
    coordinates = [latitude, longitude]
    return coordinates

# convert() converts angle coordinates into normal coordinates
# Example: "98° 34' 58.7" to "98/1,34/1,587/10"
def convert(angle):
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle

# my_capture() captures an image
def my_capture(mycam, image): 
    point = ISS.coordinates()
    south, exif_latitude = convert(point.latitude)
    west, exif_longitude = convert(point.longitude)
    mycam.exif_tags['GPS.GPSLatitude'] = exif_latitude
    mycam.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    mycam.exif_tags['GPS.GPSLongitude'] = exif_longitude
    mycam.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"
    mycam.capture(image)


while (now_time < start_time + timedelta(minutes=total_obs_time)):
    sleep(sleep_time) 
    img_name = f"img_{img_number}.jpg"
    my_capture(camera, f"{img_folder}/{img_name}")
    img_status = check_img(img_name)
    print("Picture analyzed")
    camera.close()

    if img_status == False: # Night time image
        os.remove(f"./images/{img_name}") 
      #--------------------------------------
        camera = PiCamera(
            resolution=(1280, 972),
            framerate=Fraction(1, 6),
            sensor_mode=3)
        camera.shutter_speed = 2000000
        camera.iso = 800
        camera.exposure_mode = 'off'
        my_capture(camera, f"{night_img_folder}/{img_name}")
        print("NOTTE")
       #--------------------------------------      
    elif img_status == True: # Day time image
        camera = PiCamera()
        camera.resolution = (1296,972) 
        print("GIORNO")

    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        time = get_time()
        location = track_iss()
        writer.writerow([time, location[0], location[1], img_name])
        
    img_number += 1
    now_time = datetime.now()
