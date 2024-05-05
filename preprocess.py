import json
import os
import random
from PIL import Image, ImageDraw

def load_annotations(annotation_paths):
    all_annotations = []
    all_categories = {}
    for path in annotation_paths:
        with open(path) as file:
            data = json.load(file)
            all_annotations.extend(data['annotations'])
            for category in data['categories']:
                all_categories[category['id']] = category['name']
    return all_annotations, all_categories

def filter_and_sample_classes(annotations, categories, min_images=50, total_sample_size=500):
    category_count = {}
    for ann in annotations:
        category_count[ann['category_id']] = category_count.get(ann['category_id'], 0) + 1
    
    valid_categories = {cat_id for cat_id, count in category_count.items() if count >= min_images}
    valid_categories = list(valid_categories.intersection(set(categories.keys())))

    random.shuffle(valid_categories)
    selected_categories = valid_categories[:5]  # Select 5 random valid categories

    valid_annotations = [ann for ann in annotations if ann['category_id'] in selected_categories]
    
    random.shuffle(valid_annotations)
    return valid_annotations[:total_sample_size], selected_categories

def split_data(annotations, ratio=(0.7, 0.2, 0.1)):
    random.shuffle(annotations)
    n = len(annotations)
    train_end = int(n * ratio[0])
    val_end = train_end + int(n * ratio[1])
    return annotations[:train_end], annotations[train_end:val_end], annotations[val_end:]

def create_mask(annotation, img_width, img_height):
    mask = Image.new('L', (img_width, img_height), 0)  # Black mask
    draw = ImageDraw.Draw(mask)
    for polygon in annotation['segmentation']:
        if polygon:
            reshaped_polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            draw.polygon(reshaped_polygon, outline=255, fill=255)
    return mask

def convert_to_yolo(ann, img_width, img_height):
    x_min, y_min, width, height = ann['bbox']
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    normalized_polygons = []
    for polygon in ann['segmentation']:
        normalized_polygon = ' '.join(f"{x / img_width} {y / img_height}" for x, y in zip(polygon[::2], polygon[1::2]))
        normalized_polygons.append(normalized_polygon)
    all_polygons = ' '.join(normalized_polygons)
    return f"{ann['category_id']} {x_center} {y_center} {width} {height} {all_polygons}"

def save_annotations_and_masks(annotations, img_dir, ann_dir, mask_dir, img_final_dir):
    if not os.path.exists(ann_dir):
        os.makedirs(ann_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(img_final_dir):
        os.makedirs(img_final_dir)
    for ann in annotations:
        img_path = os.path.join(img_dir, f"{ann['image_id']}.jpg")
        ann_path = os.path.join(ann_dir, f"{ann['image_id']}.txt")
        mask_path = os.path.join(mask_dir, f"{ann['image_id']}.png")
        final_img_path = os.path.join(img_final_dir, f"{ann['image_id']}.jpg")
        if not os.path.exists(img_path) or 'segmentation' not in ann:
            continue
        img = Image.open(img_path)
        img_width, img_height = img.size
        yolo_format = convert_to_yolo(ann, img_width, img_height)
        with open(ann_path, 'a') as file:
            file.write(yolo_format + '\n')
        mask = create_mask(ann, img_width, img_height)
        mask.save(mask_path)
        img.save(final_img_path)

def remap_labels(label_dir, id_mapping):
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            with open(filepath, 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    if class_id in id_mapping:
                        parts[0] = str(id_mapping[class_id])
                    file.write(" ".join(parts) + "\n")

if __name__ == "__main__":
    annotation_files = ['annotations/instances_train2024.json', 'annotations/instances_val2024.json']
    annotations, categories = load_annotations(annotation_files)
    filtered_anns, selected_cats = filter_and_sample_classes(annotations, categories)

    print("Selected categories and their IDs:", {cat: categories[cat] for cat in selected_cats})

    image_dirs = {'train': 'images/train', 'val': 'images/val', 'test': 'images/test'}
    mask_dirs = {'train': 'masks/train', 'val': 'masks/val', 'test': 'masks/test'}
    image_final_dirs = {'train': 'images_final/train', 'val': 'images_final/val', 'test': 'images_final/test'}

    train, val, test = split_data(filtered_anns)
    save_annotations_and_masks(train, image_dirs['train'], 'annotations/train', mask_dirs['train'], image_final_dirs['train'])
    save_annotations_and_masks(val, image_dirs['val'], 'annotations/val', mask_dirs['val'], image_final_dirs['val'])
    save_annotations_and_masks(test, image_dirs['test'], 'annotations/test', mask_dirs['test'], image_final_dirs['test'])

    # Adjust code based on your mapping
    # id_mapping = {6: 0, 8: 1, 9: 2, 5: 3, 10: 4}
    # remap_labels('final_data/val/labels', id_mapping)
