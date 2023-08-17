from multiprocessing import Process, cpu_count
from PIL import Image, ImageDraw
from typing import Tuple, List
import numpy as np
import argparse
import os


def rotate_point(mid_point: Tuple[float, float], input: Tuple[float, float],
                 degree: float) -> Tuple[float, float]:
    radian = degree * np.pi / 180
    np_input = np.array([[input[0]],
                         [input[1]]])
    rotation_matrix = np.array([[np.cos(radian), -np.sin(radian)],
                                [np.sin(radian), np.cos(radian)]])
    new_np_vector = rotation_matrix @ np_input
    return (new_np_vector[0][0] + mid_point[0], mid_point[1] - new_np_vector[1][0])


def get_diff_origin(mid_point: Tuple[float, float], input: Tuple[float, float]
                    ) -> Tuple[float, float]:
    new_x = input[0] - mid_point[0]
    new_y = -input[1] + mid_point[1]
    return (new_x, new_y)


def clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def main_thread(images_list: List[str], images_path: str, labels_path: str,
                degree_interval: int, inner_offset: int) -> None:
    for image in images_list:
        for degree in range(degree_interval, 360, degree_interval):
            img = Image.open(os.path.join(images_path, image)).rotate(degree)
            inner_offset = inner_offset if (degree != 90 and degree != 180 and
                                            degree != 270) else 0
            img_max_width = img.width
            img_max_height = img.height
            mid_point = (img_max_width / 2, img_max_height / 2)
            filename = f"{os.path.join(labels_path, os.path.splitext(image)[0])}.txt"
            with open(filename, "r") as f:
                annotations = f.readlines()
                content = ""
                for annotation in annotations:
                    elements = annotation.split(" ")
                    class_ = elements[0]
                    points = list(map(float, elements[1:]))
                    content += f"{class_} "
                    for i in range(0, len(points), 2):
                        x = points[i] * img_max_width
                        y = points[i + 1] * img_max_height
                        new_point = rotate_point(mid_point, get_diff_origin(mid_point, (x, y)), degree)
                        content += f"{clamp(new_point[0] / img_max_width)} {clamp(new_point[1] / img_max_height)} "
                    content += "\n"
                content.strip()
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
    inner_offset = int(args.inner_offset)

    images_list = os.listdir(images_path)
    n_cpu = cpu_count()
    n = len(images_list)

    if n < n_cpu:
        main_thread(images_list, images_path, labels_path, degree_interval, inner_offset)
    else:
        batches = []
        procs = []
        sub_n = n // n_cpu
        i = 0
        while (i + sub_n) < n:
            batches.append(images_list[i:i+sub_n])
            i += sub_n
        if i != n:
            batches.append(images_list[i:])

        for batch in batches:
            proc = Process(target=main_thread, args=(batch, images_path, labels_path,
                                                     degree_interval, inner_offset))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
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
