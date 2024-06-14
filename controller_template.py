"""
This is the template of a python controller script to use with a server-enabled Agent.
"""

import struct
import socket
import time
import math

import numpy as np
import cv2
import pyrealsense2 as rs


CONNECT_SERVER = False  # False for local tests, True for deployment


# ----------- DO NOT CHANGE THIS PART -----------

# The deploy.py script runs on the Jetson Nano at IP 192.168.123.14
# and listens on port 9292
# whereas this script runs on one of the two other Go1's Jetson Nano

SERVER_IP = "192.168.123.14"
SERVER_PORT = 9292

# Maximum duration of the task (seconds):
TIMEOUT = 180

# Minimum control loop duration:
MIN_LOOP_DURATION = 0.1


# Use this function to send commands to the robot:
def send(sock, x, y, r):
    """
    Send a command to the robot.

    :param sock: TCP socket
    :param x: forward velocity (between -1 and 1)
    :param y: side velocity (between -1 and 1)
    :param r: yaw rate (between -1 and 1)
    """
    data = struct.pack('<hfff', code, x, y, r)
    if sock is not None:
        sock.sendall(data)


# Fisheye camera (distortion_model: narrow_stereo):

image_width = 640
image_height = 480

# --------- CHANGE THIS PART (optional) ---------

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("Could not find a depth camera with color sensor")
    exit(0)

# Depht available FPS: up to 90Hz
config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)
# RGB available FPS: 30Hz
config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)
# # Accelerometer available FPS: {63, 250}Hz
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
# # Gyroscope available FPS: {200,400}Hz
# config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

# Start streaming
pipeline.start(config)

# ----------- DO NOT CHANGE THIS PART -----------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.markerBorderBits = 1

RECORD = False
history = []

# ----------------- CONTROLLER -----------------

try:
    # We create a TCP socket to talk to the Jetson at IP 192.168.123.14, which runs our walking policy:

    print("Client connecting...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        if CONNECT_SERVER:
            s.connect((SERVER_IP, SERVER_PORT))
            print("Connected.")
        else:
            s = None

        code = 1  # 1 for velocity commands

        task_complete = False
        start_time = time.time()
        previous_time_stamp = start_time

        # main control loop:
        GOAL_POSITION = (0, 0)

        TAGS = {
            1: (-58, 0),
            2: (32, 117.5),
            3: (203, 117.5),
            4: (293, 0),
            5: (203, -117.5),
            6: (32, -117.5)
        }
        # 0 -> subtract x, 1 -> add x, 2 -> subtract y, 3 -> add y
        TAG_HANDLING = {
            1: 1,
            2: 2,
            3: 2,
            4: 0,
            5: 3,
            6: 3
        }
        
        ROBOT_CAMERA_OFFSET_IN_CM = 26.0

        OBSTACLE_DISTANCE_THRESHOLD_IN_CM = 50.0

        best_global_position_estimate = None
        best_global_yaw_estimate = None

        last_distance_to_tags_in_cm = {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None
        }
        time_since_seen_tags = {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None
        }
        reached_goal = False


        while not task_complete and not time.time() - start_time > TIMEOUT:

            # avoid busy loops:
            now = time.time()
            if now - previous_time_stamp < MIN_LOOP_DURATION:
                time.sleep(MIN_LOOP_DURATION - (now - previous_time_stamp))

            # ---------- CHANGE THIS PART (optional) ----------

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            acc_data = frames[2].as_motion_frame().get_motion_data()

            ax, ay, az = acc_data.x, acc_data.y, acc_data.z
            print(ax,ay,az)

            # robot x = camera z
            # robot y = - camera x
            # robot z = - camera y
            R = np.array([[0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]])

            accel_camera_frame = np.array([ax, ay, az])
            accel_robot_frame = R @ accel_camera_frame
            print(f"Accelerometer data transformed: {ax}, {ay}, {az}")
            
            if not depth_frame or not color_frame:
                continue

            if RECORD:
                history.append((depth_frame, color_frame))

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # --------------- CHANGE THIS PART ---------------

            # --- Detect markers ---

            # Markers detection:
            grey_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            (detected_corners, detected_ids, rejected) = cv2.aruco.detectMarkers(grey_frame, aruco_dict, parameters=arucoParams)

            detected_any_tag_id = False
            if detected_ids is not None:
                for detected_id_ in detected_ids:
                    detected_id = detected_id_[0]
                    if detected_id in TAGS:
                        detected_any_tag_id = True

                        top_left, top_right, bottom_right, bottom_left = detected_corners[0][0]

                        mask = np.zeros(depth_image.shape, dtype=np.uint8)
                        polygon = np.array([[top_left, top_right, bottom_right, bottom_left]], dtype=np.int32)
                        cv2.fillPoly(mask, polygon, 255)

                        # Mask the depth image
                        masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)

                        # Calculate the mean depth value
                        mean_depth = (cv2.mean(depth_image, mask=mask)[0]) / 10.0

                        mean_depth += ROBOT_CAMERA_OFFSET_IN_CM

                        last_distance_to_tags_in_cm[detected_id] = mean_depth
                        time_since_seen_tags[detected_id] = time.time()

                        best_global_position_estimate = TAGS[detected_id]
                        handling_case = TAG_HANDLING[detected_id]
                        if handling_case == 0:
                            best_global_position_estimate = (best_global_position_estimate[0] - mean_depth, best_global_position_estimate[1])
                            best_global_yaw_estimate = math.pi
                        elif handling_case == 1:
                            best_global_position_estimate = (best_global_position_estimate[0] + mean_depth, best_global_position_estimate[1])
                            best_global_yaw_estimate = 0.0
                        elif handling_case == 2:
                            best_global_position_estimate = (best_global_position_estimate[0], best_global_position_estimate[1] - mean_depth)
                            best_global_yaw_estimate = -math.pi / 2
                        elif handling_case == 3:
                            best_global_position_estimate = (best_global_position_estimate[0], best_global_position_estimate[1] + mean_depth)
                            best_global_yaw_estimate = math.pi / 2
            
            # Update the global position and yaw estimate with the IMU if we haven't seen a tag this frame
            if detected_ids is None or not detected_any_tag_id:
                ...
            
            if reached_goal:
                x_velocity = 0.0
                y_velocity = 0.0
                yaw_velocity = 0.0
            else:
                # We haven't seen any tag yet
                if not best_global_position_estimate:
                    x_velocity = 0.0
                    y_velocity = 0.0
                    yaw_velocity = 1.0
                # We have an estimate of the robot's position
                else:
                    heading_to_goal = math.atan2(best_global_position_estimate[1] - GOAL_POSITION[1], best_global_position_estimate[0] - GOAL_POSITION[0])
                    difference_yaw = best_global_yaw_estimate - heading_to_goal
                    heading_vel_to_goal = max(min(1.0, difference_yaw), -1.0)

                    upper_depth_image = depth_image[:int(image_height * 2 / 3), :]
                    left_upper_depth_image = upper_depth_image[:, :int(image_width / 2)]
                    right_upper_depth_image = upper_depth_image[:, int(image_width / 2):]

                    median_depth_in_front_left = np.median(left_upper_depth_image) / 10.0
                    median_depth_in_front_right = np.median(right_upper_depth_image) / 10.0
                    obstacle_in_front_left = median_depth_in_front_left < OBSTACLE_DISTANCE_THRESHOLD_IN_CM
                    obstacle_in_front_right = median_depth_in_front_right < OBSTACLE_DISTANCE_THRESHOLD_IN_CM

                    if obstacle_in_front_left or obstacle_in_front_right:
                        if (obstacle_in_front_left and not obstacle_in_front_right) or (obstacle_in_front_left and obstacle_in_front_right and median_depth_in_front_left < median_depth_in_front_right):
                            x_velocity = 0.0
                            y_velocity = -1.0
                            yaw_velocity = heading_vel_to_goal
                        elif (not obstacle_in_front_left and obstacle_in_front_right) or (obstacle_in_front_left and obstacle_in_front_right and median_depth_in_front_left >= median_depth_in_front_right):
                            x_velocity = 0.0
                            y_velocity = 1.0
                            yaw_velocity = heading_vel_to_goal
                    else:
                        difference_to_goal_in_cm = (best_global_position_estimate[0] - GOAL_POSITION[0], best_global_position_estimate[1] - GOAL_POSITION[1])
                        distance_to_goal_in_cm = math.sqrt(difference_to_goal_in_cm[0] ** 2 + difference_to_goal_in_cm[1] ** 2)
                        if distance_to_goal_in_cm > 10:
                            x_velocity = min(1.0, difference_to_goal_in_cm[0] / 100.0)
                            y_velocity = -max(min(1.0, difference_to_goal_in_cm[1] / 100.0), -1.0)
                            yaw_velocity = heading_vel_to_goal
                        else:
                            reached_goal = True
                            x_velocity = 0.0
                            y_velocity = 0.0
                            yaw_velocity = 0.0

            # --- Send control to the walking policy ---
            send(s, x_velocity, y_velocity, yaw_velocity)

        print(f"End of main loop.")

        if RECORD:
            import pickle as pkl
            with open("frames.pkl", 'wb') as f:
                pkl.dump(frames, f)
finally:
    # Stop streaming
    pipeline.stop()
