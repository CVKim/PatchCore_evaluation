from PIL import Image
import numpy as np
import os
import json

def create_dummy_images(num_images, image_size, save_dir, create_json=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_images):
        image = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        image_path = os.path.join(save_dir, f'dummy_{i}.bmp')
        Image.fromarray(image).save(image_path)

        if create_json:
            create_dummy_labelme_json(image_path, image_size)


def create_dummy_labelme_json(image_path, image_size):
    # Smaller margin for blob area
    margin = image_size // 20  # 5% of the image size as margin
    center_x, center_y = image_size // 2, image_size // 2
    blob_size = image_size // 10  # Blob size is 10% of the image size

    points = [
        [center_x - blob_size, center_y - blob_size], 
        [center_x + blob_size, center_y - blob_size], 
        [center_x + blob_size, center_y + blob_size], 
        [center_x - blob_size, center_y + blob_size]
    ]

    json_path = os.path.splitext(image_path)[0] + '.json'
    data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [
            {
                "label": "dummy_blob",
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image_size,
        "imageWidth": image_size
    }

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Create 20 dummy images of size 8192x8192 and corresponding JSON files
create_dummy_images(num_images=2, image_size=8192, save_dir='dummy_images', create_json=True)
