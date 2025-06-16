import torch
import torchvision.transforms as transforms

import cv2
import face_alignment
import numpy as np
from PIL import Image

from preprocessing.eg3d_alignment import EG3DImageAlignment
from preprocessing.pose_estimation import PoseEstimator
from preprocessing.crop_image import ImageCropper
from preprocessing.foreground_extraction import ForegroundExtractor


class TestDataPreprocessor:
    def __init__(self, device):
        self.device = device

        self.eg3d_alignment = EG3DImageAlignment(output_size=1024, transform_size=1024)
        self.pose_estimator = PoseEstimator(device)
        self.cropper = ImageCropper()
        self.foreground_extractor = ForegroundExtractor(device)
        self.landmark_estimator = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
        )

        self.transform = transforms.ToTensor()

    def __call__(self, img, keep_bg=False):
        lm = self.landmark_estimator.get_landmarks(np.array(img))
        if lm is None:
            detected_face = [0, 0, img.size[0], img.size[1]]
            lm = self.landmark_estimator.get_landmarks(img, detected_faces=[detected_face])[0]
        else:
            lm = lm[0]

        img, aligned_lm = self.eg3d_alignment(img, lm)
        intrinsics, pose = self.pose_estimator.predict_pose(img, aligned_lm)

        img, aligned_lm = self.cropper(img, aligned_lm)
        img = np.array(img)

        if not keep_bg:
            matte = self.foreground_extractor(img)
            img = (img * matte).astype(np.uint8)

        crop_params = cv2.estimateAffine2D(aligned_lm, lm)[0]

        return img, intrinsics, pose, crop_params

    def from_path(self, image_path, device, keep_bg=False):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img, intrinsics, pose, crop_params = self(img, keep_bg=keep_bg)

        img = np.transpose(img, (2, 0, 1))[None, :, :, :] / 255.
        img = (img * 2 - 1)
        img = torch.from_numpy(img).float()

        pose = torch.from_numpy(pose).unsqueeze(0).float()
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).float()

        return {
            'image': img.to(device),
            'cam2world': pose.to(device),
            'intrinsics': intrinsics.to(device),
            'crop_params': crop_params
        }

    @staticmethod
    def undo_alignment(img, T):
        out = cv2.warpAffine(img, T, (img.shape[1], img.shape[0]))
        return out
