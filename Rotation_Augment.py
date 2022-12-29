from multiprocessing import Process
from PIL import Image, ImageDraw
from typing import Tuple, List
import numpy as np
import argparse
import os

def get_diff_origin(mid_point: Tuple[float, float], input: Tuple[float, float]
                    ) -> Tuple[float, float]:
    new_x = input[0] - mid_point[0]
    new_y = -input[1] + mid_point[1]
    return (new_x, new_y)

def rotate_point(mid_point: Tuple[float, float], input: Tuple[float, float],
                 degree: float) -> Tuple[float, float]:
    radian = degree * np.pi / 180
    np_input = np.array([[input[0]],
                         [input[1]]])
    rotation_matrix = np.array([[np.cos(radian), -np.sin(radian)],
                                [np.sin(radian), np.cos(radian)]])
    new_np_vector = rotation_matrix @ np_input
    return (new_np_vector[0][0] + mid_point[0], mid_point[1] - new_np_vector[1][0])

def clamp(x: float) -> float:
    return min(1.0, max(0.0, x))

def draw_point(draw, point: Tuple[float, float], radius: float) -> None:
    draw.ellipse((point[0] - radius, point[1] - radius,
                  point[0] + radius, point[1] + radius), fill="red")

def main_thread(images_list: List[str], images_path: str, labels_path: str,
                degree_interval: int, inner_offset: int) -> None:
    for image in images_list:
        for degree in range(degree_interval, 360, degree_interval):
            img = Image.open(os.path.join(images_path, image)).rotate(degree)
            img_max_width = img.width
            img_max_height = img.height
            mid_point = (img_max_width / 2, img_max_height / 2)
            inner_offset = inner_offset if (degree != 90 and degree != 180 and
                                            degree != 270) else 0.0
            filename = f"{os.path.join(labels_path, os.path.splitext(image)[0])}.txt"
            with open(filename, "r") as f:
                annotations = f.readlines()
                content = ""
                for annotation in annotations:
                    elements = annotation.split(" ")
                    class_ = elements[0]
                    x = float(elements[1]) * img_max_width
                    y = float(elements[2]) * img_max_height
                    width = float(elements[3]) * img_max_width
                    height = float(elements[4]) * img_max_height
                    width_half, height_half = width / 2, height / 2
                    point1 = rotate_point(mid_point, get_diff_origin(
                        mid_point, (x - width_half, y - height_half)), degree)
                    point2 = rotate_point(mid_point, get_diff_origin(
                        mid_point, (x + width_half, y - height_half)), degree)
                    point3 = rotate_point(mid_point, get_diff_origin(
                        mid_point, (x - width_half, y + height_half)), degree)
                    point4 = rotate_point(mid_point, get_diff_origin(
                        mid_point, (x + width_half, y + height_half)), degree)
                    obj_mid_point = rotate_point(mid_point, get_diff_origin(
                        mid_point, (x, y)), degree)
                    x_points = [point1[0], point2[0], point3[0], point4[0]]
                    y_points = [point1[1], point2[1], point3[1], point4[1]]
                    left = np.min(x_points) + inner_offset
                    right = np.max(x_points) - inner_offset
                    top = np.min(y_points) + inner_offset
                    bottom = np.max(y_points) - inner_offset
                    content += f"{class_} {clamp(obj_mid_point[0] / img_max_width)} {clamp(obj_mid_point[1] / img_max_height)} {clamp((right - left) / img_max_width)} {clamp((bottom - top) / img_max_height)}\n"
                with open(f"{os.path.join(labels_path, os.path.splitext(image)[0])}_{degree}.txt", "w") as f:
                    f.write(content)
                img.save(f"{os.path.join(images_path, os.path.splitext(image)[0])}_{degree}.jpg")



if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path")
    parser.add_argument("--labels_path")
    parser.add_argument("--degree_interval")
    parser.add_argument("--inner_offset")
    args = parser.parse_args()

    images_path = args.images_path
    labels_path = args.labels_path
    degree_interval = int(args.degree_interval)
    inner_offset = float(args.inner_offset)

    images_list = os.listdir(images_path)
    n = len(images_list)
    middle = n // 2
    list_a = images_list[:middle]
    list_b = images_list[middle:]

    proc1 = Process(target=main_thread, args=(list_a, images_path, labels_path,
                                              degree_interval, inner_offset))
    proc2 = Process(target=main_thread, args=(list_b, images_path, labels_path,
                                              degree_interval, inner_offset))
    proc1.start()
    proc2.start()
    proc1.join()
    proc2.join()
    # Code to show image with bounding box

    # img = Image.open("path/to/image")
    # img_max_width = img.width
    # img_max_height = img.height
    # mid_point = (img_max_width / 2, img_max_height / 2)
    # draw = ImageDraw.Draw(img)
    # with open("path/to/label", "r") as f:
    #     annotations = f.readlines()
    #     for annotation in annotations:
    #         elements = annotation.split(" ")
    #         class_ = elements[0]
    #         x = float(elements[1]) * img_max_width
    #         y = float(elements[2]) * img_max_height
    #         width = float(elements[3]) * img_max_width
    #         height = float(elements[4]) * img_max_height
    #         point1 = (x - width / 2, y - height / 2)
    #         point2 = (x + width / 2, y - height / 2)
    #         point3 = (x - width / 2, y + height / 2)
    #         point4 = (x + width / 2, y + height / 2)
    #         draw_point(draw, point1, 9)
    #         draw_point(draw, point2, 9)
    #         draw_point(draw, point3, 9)
    #         draw_point(draw, point4, 9)
    #         draw.rectangle((*point1, *point4), outline="red")
    #     img.show()
