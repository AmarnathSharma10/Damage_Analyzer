import cv2
import os
import random
import time
import numpy as np


class DroneToDrone:
    def __init__(self, output_folder):
        self.input = []
        self.output = []
        self.report = []
        self.output_folder = output_folder
        self.number = 0


    def get_input(self, image_folder):
        for filename in os.listdir(image_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(image_folder, filename)
                self.input.append(image_path)

    def get_number(self, n):
        self.number = n % len(self.input)
        if self.number == 0: self.number += 1

    def classify_debris(self, image_path, i):
        # Sample parameters for GSD calculation (these should be adjusted as per your scenario)
        height = 100
        sensor_width = 36
        sensor_height = 24
        focal_length = 50

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return {}

        gsd_width = calculate_gsd(height, sensor_width, image.shape[1], focal_length)
        gsd_height = calculate_gsd(height, sensor_height, image.shape[0], focal_length)
        gsd = (gsd_width + gsd_height) / 2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))

        overlap_counts = np.zeros_like(image[:, :, 0], dtype=int)
        for (x, y, w, h) in bounding_boxes:
            overlap_counts[y:y + h, x:x + w] += 1

        output_image = image.copy()
        total_area_70 = total_area_80 = total_area_90 = total_area_100 = 0
        count_70 = count_80 = count_90 = count_100 = 0

        for (x, y, w, h) in bounding_boxes:
            region_overlap_count = overlap_counts[y:y + h, x:x + w].max()
            if region_overlap_count >= 4:
                color = (100, 100, 255)
                total_area_100 += w * h * (gsd ** 2)
                count_100 += 1
            elif region_overlap_count == 3:
                color = (100, 200, 255)
                total_area_90 += w * h * (gsd ** 2)
                count_90 += 1
            elif region_overlap_count == 2:
                color = (100, 255, 100)
                total_area_80 += w * h * (gsd ** 2)
                count_80 += 1
            else:
                color = (255, 100, 100)
                total_area_70 += w * h * (gsd ** 2)
                count_70 += 1

            overlay = output_image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)
        op_name = f"OP{i}.jpg"
        self.output.append(output_image)
        save_image(output_image, self.output_folder, op_name)

        return {
            "LOW  damage " + " " * 5: {"regions": count_70, "total_area(in m*m)": total_area_70},
            "MEDIUM damage": {"regions": count_80, "total_area(in m*m)": total_area_80},
            "HIGH damage ": {"regions": count_90, "total_area(in m*m)": total_area_90},
            "SEVERE  damage ": {"regions": count_100, "total_area(in m*m)": total_area_100}

        }

    def mainProcess(self):
        randomInput = random.sample(self.input, self.number)
        i = 0
        for image in randomInput:
            i += 1
            self.report.append(self.classify_debris(image, i))
            time.sleep(1)






def remove_files(directory):
    # List all entries in the directory
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isfile(path):
            # Delete the file
            os.remove(path)
    # os.rmdir(directory)


def calculate_gsd(height, sensor_size, image_dimension, focal_length):
    return (height * sensor_size) / (image_dimension * focal_length)


def save_image(image, directory, filename):
    file_path = os.path.join(directory, filename)
    if image is not None:
        if os.path.exists(file_path):
            os.remove(file_path)
        cv2.imwrite(file_path, image)
        print(f"Best image saved as {filename}")
    else:
        print("No image to save.")


