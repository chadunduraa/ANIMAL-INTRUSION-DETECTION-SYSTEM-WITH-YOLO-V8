from ultralytics import YOLO
import cv2
import math
import pygame
from twilio.rest import Client

pygame.mixer.init()
siren_sound = pygame.mixer.Sound('sirenn.mp3')
# Running real-time from webcam
cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam, you can change it based on your webcam index
account_sid = 'ACa38c71dfc3b62f5cd310f403455c279d'
auth_token = '449188d0c0e3ce0ddecda481017cc9d2'
twilio_phone_number = '+12512418103'

# Create a Twilio client
client = Client(account_sid, auth_token)

# File to store phone numbers
#phone_numbers_file = "/Users/akhileshv/Liveinlab/phonenumber.txt"
# Load YOLOv5 model (replace 'final.pt' with your actual model path
model = YOLO('best.pt')

def send_sms(message):
        client.messages.create(
            to='+919363066620',
            from_=twilio_phone_number,
            body=message
        )

classnames = [
    'antelope', 'bear', 'cheetah', 'human', 'coyote', 'crocodile', 'deer', 'elephant', 'flamingo',
    'fox', 'giraffe', 'gorilla', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena',
    'kangaroo', 'koala', 'leopard', 'lion', 'meerkat', 'mole', 'monkey', 'moose', 'okapi', 'orangutan',
    'ostrich', 'otter', 'panda', 'pelecaniformes', 'porcupine', 'raccoon', 'reindeer', 'rhino', 'rhinoceros',
    'snake', 'squirrel', 'swan', 'tiger', 'turkey', 'wolf', 'woodpecker', 'zebra'
]
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    
    # Perform inference with YOLOv5
    result = model(frame, stream=True)

    # Process bounding boxes and display results
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            class_index = int(box.cls[0])
            class_name = classnames[class_index]


            if confidence > 50 and classnames[class_index] in classnames:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Display bounding box and class label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(frame, f'{classnames[class_index]} {confidence}%', (x1 + 8, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                 # Play siren sound if detected animal is in the specified list
                if class_name in [ 'bear', 'cheetah',  'coyote', 'crocodile',  'elephant',
                                   'leopard', 'lion', 'snake','tiger', 'wolf']:
                    siren_sound.play()
                    send_sms(f"Animal detected!!!!! WARNING ")


    # Display the frame with detected animals
    cv2.imshow('Animal Detection', frame)

    # Press 'Esc' key to exit the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
