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
        if self.number == 0: self.number += len(self.input)

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


        connected_regions_70 = []
        connected_regions_80 = []
        connected_regions_90 = []
        connected_regions_100 = []

        for (x, y, w, h) in bounding_boxes:
            region_overlap_count = overlap_counts[y:y + h, x:x + w].max()
            if region_overlap_count >= 4:
                color = (0, 0, 255)
                total_area_100 += w * h * (gsd ** 2)
                count_100 += 1
                connected_regions_100.append((x, y, w, h))
            elif region_overlap_count == 3:
                color = (0, 165, 255)
                total_area_90 += w * h * (gsd ** 2)
                count_90 += 1
                connected_regions_90.append((x, y, w, h))
            elif region_overlap_count == 2:
                color = (0, 255, 255)
                total_area_80 += w * h * (gsd ** 2)
                count_80 += 1
                connected_regions_80.append((x, y, w, h))
            else:
                color = (0, 255, 0)
                total_area_70 += w * h * (gsd ** 2)
                count_70 += 1
                connected_regions_70.append((x, y, w, h))

            overlay = output_image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

        op_name = f"OP{i}.jpg"
        self.output.append(output_image)
        save_image(output_image, self.output_folder, op_name)


        def find_connected_regions(regions):
            if not regions:
                return []
            regions = sorted(regions, key=lambda x: (x[0], x[1]))
            connected = []
            current_group = [regions[0]]

            for i in range(1, len(regions)):
                prev = regions[i - 1]
                curr = regions[i]
                if curr[0] <= prev[0] + prev[2] and curr[1] <= prev[1] + prev[3]:
                    current_group.append(curr)
                else:
                    connected.append(len(current_group))
                    current_group = [curr]

            connected.append(len(current_group))
            return connected

        # Compute the connected regions for each damage level
        connected_70 = find_connected_regions(connected_regions_70)
        connected_80 = find_connected_regions(connected_regions_80)
        connected_90 = find_connected_regions(connected_regions_90)
        connected_100 = find_connected_regions(connected_regions_100)

        return {
            "LOW  damage  GREEN" + " " * 5: {"regions": count_70, "total_area(in m*m)": total_area_70,
                                       "connected_regions": connected_70},
            "MEDIUM damage YELLOW": {"regions": count_80, "total_area(in m*m)": total_area_80,
                              "connected_regions": connected_80},
            "HIGH damage ORANGE": {"regions": count_90, "total_area(in m*m)": total_area_90,
                             "connected_regions": connected_90},
            "SEVERE  damage RED": {"regions": count_100, "total_area(in m*m)": total_area_100,
                                    "connected_regions": connected_100}
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


if __name__ == "__main__":
    image_folder = 'input_images'
    output_folder = 'OUTPUT_IMAGES'
    if os.path.exists(output_folder):
        remove_files(output_folder)
        os.rmdir(output_folder)
    os.makedirs(output_folder)
    d = DroneToDrone(output_folder)
    d.get_input(image_folder)
    d.get_number(6)
    d.mainProcess()
    for report in d.report:
        for key,val in report.items():
            print(key)
            for i,j in val.items():
                print(f"{i} {j}")
            print("____________________________________")

        print("________________________________________")

