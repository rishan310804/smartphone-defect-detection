import json
import os
from pathlib import Path
import argparse


def convert_coco_to_yolo(json_path, image_path, output_path):
    """
    Converts COCO JSON annotation file to YOLO format .txt files.

    Args:
        json_path (str): Path to the COCO JSON annotation file.
        image_path (str): Path to the directory containing the images.
        output_path (str): Path to the directory where YOLO .txt files will be saved.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {image['id']: image for image in data['images']}
    annotations = data['annotations']

    # Create a mapping from category ID to a continuous 0-based index
    categories = {cat['id']: i for i, cat in enumerate(data['categories'])}

    # Group annotations by image
    image_annots = {image_id: [] for image_id in images}
    for annot in annotations:
        image_annots[annot['image_id']].append(annot)

    converted_count = 0
    for image_id, annots in image_annots.items():
        image = images[image_id]
        img_h = image['height']
        img_w = image['width']
        file_name = image['file_name']

        yolo_lines = []
        for annot in annots:
            category_id = categories[annot['category_id']]

            # COCO format: [x_min, y_min, width, height]
            x_min, y_min, bbox_w, bbox_h = annot['bbox']

            # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
            x_center = (x_min + bbox_w / 2) / img_w
            y_center = (y_min + bbox_h / 2) / img_h
            w = bbox_w / img_w
            h = bbox_h / img_h

            yolo_lines.append(
                f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            # Save the .txt file
            txt_path = output_path / f"{Path(file_name).stem}.txt"
            with open(txt_path, 'w') as f_out:
                f_out.write('\n'.join(yolo_lines))
            converted_count += 1

    print(
        f"✅ Converted {converted_count} annotations to YOLO format from '{Path(json_path).name}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert COCO JSON to YOLO format.')
    parser.add_argument('--json_path', type=str,
                        required=True, help='Path to COCO JSON file.')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image directory.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save YOLO annotation files.')

    args = parser.parse_args()

    convert_coco_to_yolo(args.json_path, args.image_path, args.output_path)
