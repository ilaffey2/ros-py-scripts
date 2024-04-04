import rospy
import rosbag
import cv2
import numpy as np
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
from ahrs import Madgwick

# Initialize ROS node
rospy.init_node("gravity_visualization")

# Load the rosbag
bag = rosbag.Bag("~/data/MH_01_easy.bag")

# Create a CvBridge instance
bridge = CvBridge()

# Create an AHRS filter instance
ahrs = Madgwick()

# Initialize variables for gravity direction
gravity_direction = None

# Iterate through the rosbag messages
for topic, msg, t in bag.read_messages(topics=["/camera/image_raw", "/imu0"]):
    if topic == "/camera/image_raw":
        # Convert the image message to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Display gravity direction on the image
        if gravity_direction is not None:
            # Convert gravity direction to pixel coordinates
            center = (cv_image.shape[1] // 2, cv_image.shape[0] // 2)
            gravity_pixel = (
                int(center[0] + gravity_direction[0] * 100),
                int(center[1] - gravity_direction[1] * 100),
            )

            # Draw an arrow representing the gravity direction
            cv2.arrowedLine(cv_image, center, gravity_pixel, (0, 255, 0), 2)

        # Display the image
        cv2.imshow("Camera Feed with Gravity Direction", cv_image)
        cv2.waitKey(1)

    elif topic == "/imu0":
        # Update the AHRS filter with IMU data
        ahrs.update(
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        )

        # Get the gravity direction in the camera frame
        gravity_direction = ahrs.quaternion.to_euler()[:2]

# Close the rosbag
bag.close()

# Destroy OpenCV windows
cv2.destroyAllWindows()
