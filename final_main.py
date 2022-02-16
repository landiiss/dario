from datetime import datetime, timedelta
from time import sleep
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

cam = PiCamera()
cam.resolution = (1296,972)
base_folder = Path(__file__).parent.resolve()
img_folder = base_folder/'images'
model_file = base_folder/'models/model_edgetpu.tflite' 
start_time = datetime.now()
now_time = datetime.now()
img_number = 1
img_name = f"img_{img_number}.jpg"


def check_img(image):
    model_file = base_folder/'models/model_edgetpu.tflite' 
    data_dir = base_folder/'data'
    label_file = data_dir/'labels.txt' 
    image_file = img_folder/image

    interpreter = make_interpreter(f"{model_file}")
    interpreter.allocate_tensors()
    size = common.input_size(interpreter)
    image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)
    labels = read_label_file(label_file)

    for c in classes:
        print(f'{labels.get(c.id, c.id)} {c.score:.5f}')
        sheldon = str({labels.get(c.id, c.id)})
        sheldon = sheldon.replace("{'", "")
        sheldon = sheldon.replace("'}", "")
        if sheldon == "day":
            return True
        else:
            return False

def get_time():
    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")
    return now

def track_iss():
    location = ISS.coordinates() 
    latitude = location.latitude.degrees
    longitude = location.longitude.degrees
    coordinates = [latitude, longitude]
    return coordinates

def convert(angle):
    #98° 34' 58.7 to "98/1,34/1,587/10"
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle

def capture(camera, image):
    point = ISS.coordinates()
    south, exif_latitude = convert(point.latitude)
    west, exif_longitude = convert(point.longitude)
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"
    camera.capture(image)

while (now_time < start_time + timedelta(minutes=1)):
    sleep(5)
    img_name = f"img_{img_number}.jpg"
    capture(cam, f"{img_folder}/{img_name}")
    img_status = check_img(img_name)
    print("Picture analyzed")
    if img_status != True:
        #os.remove(f"./images/{img_name}")
        pass

    with open('data.csv', 'a', newline='') as file:
        image_att = ""
        if img_status == True:
            image_att = img_name
        else:
            image_att = ""
        writer = csv.writer(file)
        time = get_time()
        location = track_iss()
        writer.writerow([time, location[0], location[1], image_att])
        
    img_number += 1
    now_time = datetime.now()