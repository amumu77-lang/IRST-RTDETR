import cv2
import numpy as np
import os
import json
from scipy import ndimage
from pathlib import Path


class GCP_TEAS:
    def __init__(self, lambda_val=0.4, tau=0.6, a=0.2, pyramid_levels=4):
        self.lambda_val = lambda_val
        self.tau = tau
        self.a = a
        self.pyramid_levels = pyramid_levels

    def global_contrast_estimation(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        pyramid = [gray.astype(np.float32)]
        for i in range(1, self.pyramid_levels):
            pyramid.append(cv2.pyrDown(pyramid[i - 1]))

        global_contrast = 0
        for i, level_img in enumerate(pyramid):
            contrast = np.var(level_img)
            weight = 1.0 / (2 ** i)
            global_contrast += weight * contrast

        return global_contrast / self.pyramid_levels

    def target_region_extraction(self, image, bbox):
        x_min, y_min, x_max, y_max = bbox

        h, w = image.shape[:2]
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))

        if x_min >= x_max or y_min >= y_max:
            return np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image), None

        target_region = image[y_min:y_max, x_min:x_max]

        if len(target_region.shape) == 3:
            target_gray = cv2.cvtColor(target_region, cv2.COLOR_RGB2GRAY)
        else:
            target_gray = target_region

        median_intensity = np.median(target_gray)
        lower_threshold = int(max(0, 0.7 * median_intensity))
        upper_threshold = int(min(255, 1.3 * median_intensity))

        edges = cv2.Canny(target_gray, lower_threshold, upper_threshold)

        edge_mask = np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image)
        edge_mask[y_min:y_max, x_min:x_max] = edges

        return edge_mask, target_region

    def local_saliency_scoring(self, image, edge_mask):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        local_contrast = ndimage.generic_filter(gray.astype(np.float32), np.std, size=5)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_energy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        local_contrast_norm = (local_contrast - np.min(local_contrast)) / (
                    np.max(local_contrast) - np.min(local_contrast) + 1e-8)
        gradient_energy_norm = (gradient_energy - np.min(gradient_energy)) / (
                    np.max(gradient_energy) - np.min(gradient_energy) + 1e-8)

        saliency_score = (self.lambda_val * local_contrast_norm +
                          (1 - self.lambda_val) * gradient_energy_norm)

        return saliency_score

    def selective_edge_enhancement(self, image, saliency_score, edge_mask):
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])

        weak_edge_mask = (saliency_score < self.tau) & (edge_mask > 0)

        if len(image.shape) == 3:
            enhanced_image = image.copy().astype(np.float32)
            for channel in range(image.shape[2]):
                channel_img = image[:, :, channel].astype(np.float32)
                sharpened = cv2.filter2D(channel_img, -1, sharpen_kernel)
                enhanced_channel = np.where(weak_edge_mask,
                                            channel_img + self.a * (sharpened - channel_img),
                                            channel_img)
                enhanced_image[:, :, channel] = enhanced_channel
        else:
            sharpened = cv2.filter2D(image.astype(np.float32), -1, sharpen_kernel)
            enhanced_image = np.where(weak_edge_mask,
                                      image.astype(np.float32) + self.a * (sharpened - image.astype(np.float32)),
                                      image.astype(np.float32))

        return np.clip(enhanced_image, 0, 255).astype(np.uint8)

    def process_single_image(self, image, bbox_list):
        global_contrast = self.global_contrast_estimation(image)

        combined_edge_mask = np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image)

        for bbox in bbox_list:
            edge_mask, target_region = self.target_region_extraction(image, bbox)
            combined_edge_mask = cv2.bitwise_or(combined_edge_mask, edge_mask)

        saliency_score = self.local_saliency_scoring(image, combined_edge_mask)

        enhanced_image = self.selective_edge_enhancement(image, saliency_score, combined_edge_mask)

        output_image = np.where(
            combined_edge_mask[:, :, np.newaxis] > 0 if len(image.shape) == 3 else combined_edge_mask > 0,
            enhanced_image, image)

        return output_image


def load_annotations(annotation_path, annotation_format='coco'):
    bbox_list = []

    if annotation_format == 'coco':
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)

        for annotation in coco_data.get('annotations', []):
            bbox = annotation['bbox']
            x_min, y_min, width, height = bbox
            x_max, y_max = x_min + width, y_min + height
            bbox_list.append([x_min, y_min, x_max, y_max])

    elif annotation_format == 'yolo':
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                pass

    elif annotation_format == 'txt_bbox':
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            coords = line.strip().split(',')
            if len(coords) == 4:
                x_min, y_min, x_max, y_max = map(float, coords)
                bbox_list.append([x_min, y_min, x_max, y_max])

    return bbox_list


def batch_process_gcp_teas(input_image_dir, input_label_dir, output_dir,
                           annotation_format='txt_bbox', image_ext='.jpg'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    processor = GCP_TEAS(lambda_val=0.4, tau=0.6, a=0.2)

    image_files = list(Path(input_image_dir).glob(f'*{image_ext}'))

    processed_count = 0
    error_count = 0

    for image_path in image_files:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label_path = Path(input_label_dir) / f"{image_path.stem}.txt"

            if not label_path.exists():
                print(f"Annotation file not found: {label_path}")
                continue

            bbox_list = load_annotations(str(label_path), annotation_format)

            if not bbox_list:
                print(f"No valid bounding boxes in annotation file: {label_path}")
                continue

            enhanced_image = processor.process_single_image(image_rgb, bbox_list)

            enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
            output_path = Path(output_dir) / f"{image_path.stem}_enhanced{image_path.suffix}"
            cv2.imwrite(str(output_path), enhanced_bgr)

            processed_count += 1
            print(f"Processed: {image_path.name} -> {output_path.name}")

        except Exception as e:
            error_count += 1
            print(f"Processing failed {image_path.name}: {str(e)}")

    print(f"Processing completed! Successful: {processed_count}, Failed: {error_count}")


if __name__ == "__main__":
    input_image_dir = "path/to/your/images"
    input_label_dir = "path/to/your/labels"
    output_dir = "path/to/your/output"

    batch_process_gcp_teas(
        input_image_dir=input_image_dir,
        input_label_dir=input_label_dir,
        output_dir=output_dir,
        annotation_format='txt_bbox',
        image_ext='.jpg'
    )