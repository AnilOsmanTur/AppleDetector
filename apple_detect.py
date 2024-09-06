import os
import sys
import cv2
import clip
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import PIL.ImageOps as iops
import matplotlib.pyplot as plt
sys.path.append('./FastSAM/')
from FastSAM.fastsam import FastSAM, FastSAMPrompt


class AppleDetector:
    def __init__(self, output_dir, in_size, conf_th, iou_th, device, verbose=False):
        # model parameters
        self.device = f'cuda:{device}' if device > -1 else 'cpu'
        self.in_size = in_size
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.verbose = verbose
        # model initializations
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.sam_model = FastSAM('FastSAM/weights/FastSAM-s.pt')
        self.prompt_process = None
        # result collections
        self.bboxes = None
        self.centers = None
        self.objects = None
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def load_image(self, image_path):
        self.image_path = image_path
        self.image_org = Image.open(image_path)
        if self.image_org:
            print('Image loaded successfully.')
        else:
            print('Error loading image.')
            return None
        
        self.wpercent = (self.in_size / float(self.image_org.size[0]))
        self.hsize = int((float(self.image_org.size[1]) * float(self.wpercent)))
        input_image = self.image_org.resize((self.in_size, self.hsize), Image.Resampling.LANCZOS)
        return input_image

    def get_segments(self, input_image):
        # run the model
        iou_th = self.iou_th
        conf_th = self.conf_th
        everything_results = self.sam_model(input_image,
                                            imgsz=self.in_size,
                                            conf=iou_th,
                                            iou=conf_th,
                                            retina_masks=True,
                                            device=self.device,
                                            verbose=self.verbose)
        self.prompt_process = FastSAMPrompt(input_image, everything_results, device=self.device)
        self.ann = self.prompt_process.everything_prompt()
        
        while len(self.ann) == 0:
            iou_th -= 0.1
            # conf_th -= 0.1
            everything_results = self.sam_model(input_image,
                                            imgsz=self.in_size,
                                            conf=conf_th,
                                            iou=iou_th,
                                            retina_masks=True,
                                            device=self.device,
                                            verbose=self.verbose)
            self.prompt_process = FastSAMPrompt(input_image, everything_results, device=self.device)
            self.ann = self.prompt_process.everything_prompt()
        
        # gather the results
        format_results = self.prompt_process._format_results(self.prompt_process.results[0], 0)
        (
            _,
            self.bboxes,
            _,
            _,
            _
        ) = self.prompt_process._crop_image(format_results)
        
        img_arr = np.array(input_image)
        clip_images = []
        for i in range(len(self.bboxes)):
            bbox = self.bboxes[i]
            im = Image.fromarray(img_arr[bbox[1]:bbox[3], bbox[0]:bbox[2], :])
            clip_img = iops.pad(im, (256, 256), color='white')
            clip_images.append(clip_img)

        return clip_images
    
    def predict_apples(self, clip_images):
        classes = ['an apple',
                    'a leaf',
                    'a branch',
                    'a tree',
                    'a net']
        text_descriptions = ['a photo of ' + item for item in classes]
        with torch.no_grad():
            preprocessed_images = [self.preprocess(image).to(self.device) for image in clip_images]
            tokenized_text = clip.tokenize(text_descriptions).to(self.device)
            stacked_images = torch.stack(preprocessed_images)
            image_features = self.clip_model.encode_image(stacked_images)
            text_features = self.clip_model.encode_text(tokenized_text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probs = 100.0 * image_features @ text_features.T
            logits = probs.softmax(dim=-1)
            predicted_idx = torch.argwhere(torch.argmax(logits, dim=1) < 1).squeeze().cpu().numpy()
        return predicted_idx

    def gather_results(self, preds):
        # gathering the predicted bboxes
        self.bboxes = [self.bboxes[i] for i in preds]
        # calculating original bboxes values
        self.bboxes = np.array(self.bboxes)
        self.bboxes = (self.bboxes / self.wpercent).astype(int)
        img_arr = np.array(self.image_org)
        center_pos = []
        objects = []
        for i in range(len(self.bboxes)):
            bbox = self.bboxes[i]
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            center_pos.append([center_x, center_y])
            im = Image.fromarray(img_arr[bbox[1]:bbox[3], bbox[0]:bbox[2], :])
            objects.append(im)
        self.centers = center_pos
        self.objects = objects

    def draw_bboxes(self):
        image_arr = np.array(self.image_org)
        for bbox in self.bboxes:
            cv2.rectangle(image_arr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 5)
        in_name = self.image_path.split('/')[-1]
        output_path = f'{self.output_dir}/{in_name}_bboxes.png'
        pil_img = Image.fromarray(image_arr)
        pil_img.save(output_path)

    def __call__(self, input_image, visualize=True):
        if type(input_image) == str:
            input_image = self.load_image(input_image)
        clip_images = self.get_segments(input_image)
        predicted_idx = self.predict_apples(clip_images)

        self.gather_results(predicted_idx)
        if visualize:
            self.draw_bboxes()
        # save the centers    
        if len(self.centers) > 0:
            center_points = np.array(self.centers)
            in_name = self.image_path.split('/')[-1]
            pd.DataFrame(center_points, columns=['x', 'y']).to_csv(
                f'{self.output_dir}/{in_name}_centers.csv',
                index=False,
            )
        return self.centers, self.objects


def arg_parser_init():
    # arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='images folder')
    parser.add_argument('--file', type=str, help='images file path')
    parser.add_argument('--output', default='./outputs', type=str, help='output folder')
    parser.add_argument('--device', default=0, type=int, help='device to run the model -1 for cpu')
    parser.add_argument('--iou', default=0.8, type=float, help='iou threshold')
    parser.add_argument('--conf', default=0.4, type=float, help='confidence threshold')
    parser.add_argument('--size', default=1024, type=int, help='input size to run the model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parser_init()

    detector = AppleDetector(args.output, args.size, args.conf, args.iou, args.device, True)
    if args.file:
        centers, objects = detector(args.file)
        print(f'Found {len(centers)} apples in {args.file}')
    elif args.dataset:
        file_names = sorted(os.listdir(args.dataset))
        for img_name in tqdm(file_names):
            img_path = os.path.join(args.dataset, img_name)
            centers, objects = detector(img_path)
            print(f'Found {len(centers)} apples in {img_name}')
    else:
        print('No input provided.')