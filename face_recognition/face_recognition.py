import os
import face_recognition
import cv2
import numpy as np
import torch
import pytz
from datetime import datetime
import requests

name_dic = {'id_101 atif':1,
            'id_102 Muzammil':2,
            'id_103 shazaib':3,
            'id_104 taha':4,
            'id_105 hassan':5,
            'id_106 fazal':6,
            'id_107 Bilal':7,
            'id_108 tariq':8,
            'id_109 Farhat':9,
            'id_110 Shaikh Muhammad Zulqarnain':10,
            'id_111 Zain':11,
            'id_112 zubair':12,
            'id_113 Basim':13,
            'id_114 Asim':14,
            'id_115 Muhammad Naseem':15,
            'id_116 Bilal saeed':16,
            'id_117 Mohammad Bilal':17,
            'id_118 Muhammad Yaseen':18,
            'id_119 Faizan Abbasi':19,
            }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to load known faces and their names from a directory
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpeg', '.jpg', '.png')):
            # Load the image file
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            print(image_path)
            
            # Encode the face
            face_encoding = face_recognition.face_encodings(image)
        
            if len(face_encoding) > 0:
                # Move to the appropriate device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                # Convert the encoding to a PyTorch tensor and send it to GPU (if available)
                known_face_encodings.append(torch.tensor(face_encoding[0]).to(device))
                
                # Extract the employee ID from the filename
                #emp_id = filename.split(' ')[0]  # Split by space and take the first part
                #known_face_ids.append(emp_id)  # Store the ID

                name = os.path.splitext(filename)[0]  # Use the full name
                known_face_names.append(name)
    
    return known_face_encodings, known_face_names 
# Load known faces
KNOWN_FACES_DIR = './known_faces'
# known_face_encodings, known_face_names, known_face_ids = load_known_faces(KNOWN_FACES_DIR)

known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)



def post_attendance(employee_id, store_uid, clock_in_type):
    # Get the current date and time in Pakistan Standard Time (PST)
    pk_timezone = pytz.timezone('Asia/Karachi')
    current_time = datetime.now(pk_timezone)
    
    # Format the time and date for the API
    time_str = current_time.strftime('%H:%M:%S.%f')[:-3]
    date_str = current_time.strftime('%Y-%m-%d')
    
    # API URL
    api_url = "http://test.mobilelinkusa.com/AICyberSecurity/api/LocationActivities/postemployeeattendancelog"
    
    # Data to send in the API request
    payload = {
        "EmpID": employee_id,
        "Store_UID": store_uid,
        "Time": time_str,
        "Date": date_str,
        "ClockInType": clock_in_type
    }

    
    # Send the API request
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        print(f"Attendance logged successfully for {store_uid}")
    else:
        print(f"Failed to log attendance for {store_uid}, status code: {response.status_code}")

#Open the video stream
video_capture = cv2.VideoCapture("rtsp://admin:admin123@10.0.0.9:554/cam/realmonitor?channel=6&subtype=0")
THRESHOLD = 0.5
clip_duration = 35  # Duration of the clip in seconds
recording=False
# Dictionary to keep track of frame counts for each detected person
frame_counts = {}
# Process every nth frame
process_interval = 3
frame_count = 0
# Output directory settings
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
# Initialize date handling
current_date = datetime.now().strftime("%Y-%m-%d")
date_dir = os.path.join(output_dir, current_date)
os.makedirs(date_dir, exist_ok=True)
# Create subdirectories for images and videos
image_dir = os.path.join(date_dir, 'images')
video_dir = os.path.join(date_dir, 'videos')
os.makedirs(image_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
logged_names = set()
video_writer = None

# Function to upload files to the API
def upload_file(file_path, url):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(url, files=files)
        print(f'Upload Status Code: {response.status_code}')
        print(f'Upload Response: {response.text}')

# Paths and configurations
base_data_dir = '/home/live/Documents/face_recognition_system/data'
api_url = 'http://test.mobilelinkusa.com/AICyberSecurity/api/LocationActivities/UploadFile/I'  # Update with actual API URL
image_dir = os.path.join(base_data_dir, datetime.now().strftime("%Y-%m-%d"), "images")
video_dir = os.path.join(base_data_dir, datetime.now().strftime("%Y-%m-%d"), "videos")

# Function to upload all images and videos from today's directory
def upload_daily_data():
    today = datetime.now().strftime("%Y-%m-%d")
    today_image_dir = os.path.join(base_data_dir, today, "images")
    today_video_dir = os.path.join(base_data_dir, today, "videos")

    # Upload images
    if os.path.exists(today_image_dir):
        for image_file in os.listdir(today_image_dir):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                upload_file(os.path.join(today_image_dir, image_file), api_url)

    # Upload videos
    if os.path.exists(today_video_dir):
        for video_file in os.listdir(today_video_dir):
            if video_file.endswith(('.mp4', '.mov', '.avi')):
                upload_file(os.path.join(today_video_dir, video_file), api_url)

# Initial call to upload today's data
upload_daily_data()
import time

# Initial timestamp for checking capture
last_success_time = time.time()
while True:

    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to capture video stream.")
        
        # Check if 30 seconds have passed since the last successful capture
        if time.time() - last_success_time >= 10:
            print("Attempting to reconnect to the video stream...")
            video_capture.release()  # Release the current capture
            video_capture = cv2.VideoCapture("rtsp://admin:admin123@10.0.0.9:554/cam/realmonitor?channel=6&subtype=0")
            
            # Wait a moment to allow the camera to initialize
            time.sleep(2)  # Adjust as needed
            ret, frame = video_capture.read()  # Try to read again

            if ret:
                print("Reconnected to the video stream.")
                last_success_time = time.time()  # Update the last success time
            else:
                print("Still unable to capture video stream.")
        continue  # Skip the rest of the loop
    # Update last success time if frame is captured successfully
    last_success_time = time.time()
    frame_count += 1

    if frame_count % process_interval == 0:
        small_frame = cv2.resize(frame, (640, 480))
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            name = "Unknown"

            face_encoding_tensor = torch.tensor(face_encoding).to(device)

            if len(known_face_encodings) > 0:

                face_distances = [torch.norm(known_face - face_encoding_tensor).item() for known_face in known_face_encodings]
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]

                if best_distance < THRESHOLD:
                    name = known_face_names[best_match_index]
                    # print(name)
                    # Check if the name exists in the dictionary
                    if name in name_dic:
                        emp_id = name_dic[name]  # Get the employee ID from the dictionary
                        # print(f"Detected: {name}, Employee ID: {emp_id}")

                        # Prepare the label with name and timestamp
                        current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                        label = f"{name} - {current_time}"

                        # Draw the label on the frame
                        cv2.putText(small_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


                        # Save the complete frame as an image
                        frame_filename = os.path.join(image_dir, f"{label}.jpg")
                        cv2.imwrite(frame_filename, small_frame)  # Save the whole frame
                        # print(f"Saved image: {frame_filename}")
                        upload_file(frame_filename, api_url)

                        if not recording:
                            recording = True
                            start_time = datetime.now()
                            clip_filename = os.path.join(video_dir, f"{name}_{start_time.strftime('%Y_%m_%d_%H_%M_%S')}.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(clip_filename, fourcc, 20.0, (640, 480))
                            # print(f"Started recording: {clip_filename}")

                    # Draw a box around the face
                    cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if recording:
                video_writer.write(small_frame)
                if (datetime.now() - start_time).seconds >= clip_duration:
                    recording = False
                    video_writer.release()
                    # print(f"Stopped recording: {clip_filename}")
                    upload_file(clip_filename, api_url)

                if name != "Unknown" and name in name_dic:
                    if name not in logged_names:
                        pk_timezone = pytz.timezone('Asia/Karachi')
                        current_hour = datetime.now(pk_timezone).hour
                        clock_in_type = "I" if 19 <= current_hour <= 24 else "O"

                        # Log the attendance using the extracted emp_id
                        #print(f"Logging attendance: employee_id={emp_id}, store_uid={name}, clock_in_type={clock_in_type}")

                        post_attendance(employee_id=emp_id, store_uid=name, clock_in_type=clock_in_type)
                        logged_names.add(name)

        #Display the resulting frame
        # cv2.imshow('Video', small_frame)

    #Break the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
# Clean up
if video_writer is not None:
    video_writer.release()  # Ensure the video writer is released if still recording
video_capture.release()

# cv2.destroyAllWindows()
