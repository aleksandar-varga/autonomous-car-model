import argparse
import base64

import cv2
import eventlet
# import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.models import load_model
from io import BytesIO


sio = socketio.Server()
app = Flask(__name__)
model = None


@sio.on('telemetry')
def receive(sid, data):
    if data:
        speed = float(data["speed"])

        imgString = data["image"]

        image = Image.open(BytesIO(base64.b64decode(imgString)))

        img = np.asarray(image)[60:-25, :, :]

        img = cv2.resize(img, (200, 66), cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        img = np.array([img])

        steering_angle = float(model.predict(img, batch_size=1))

        throttle = 1.0 - steering_angle ** 2 - (speed / 15) ** 2

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

def main(path):
    global model
    model = load_model(path)

    global app
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model')

    args = parser.parse_args()

    main(path=args.model)
