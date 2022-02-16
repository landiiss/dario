from time import sleep
from picamera import PiCamera
from pathlib import Path

base_folder = Path(__file__).parent.resolve()

camera = PiCamera()
camera2 = camera
camera.resolution = (1296,972)
camera.start_preview()
camera2.resolution = (720,400)
camera2.start_preview()
sleep(2)
camera.capture(f"{base_folder}/image.jpg")
camera2.capture(f"{base_folder}/image2.jpg")