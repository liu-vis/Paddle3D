#  !/usr/bin/env python3
"""
Loss functions for stylization
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

import cv2
from PIL import Image
import paddle.vision.transforms as transforms


imagenet_templates = [
    'a bad photo of a {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


class DirectionLoss(nn.Layer):
    """
    DirectionLoss functions
    """

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type
    
        #Define multiple losses, in practice cosine similarity is used
        self.loss_func = {
            'mse':    nn.MSELoss,
            'cosine': nn.CosineSimilarity,
            'mae':    nn.L1Loss
        }[loss_type]()


    def forward(self, I_direct, t_direct):
        """
        compute DirectionLoss 
        """
        if self.loss_type == "cosine":
            return 1. - self.loss_func(I_direct, t_direct)
        
        return self.loss_func(I_direct, t_direct)


class CLIPLoss(nn.Layer):
    """
    CLIPLoss functions
    """

    def __init__(self, 
        direction_loss_type='cosine', 
        distance_loss_type='mae', 
        use_distance=False, 
        src_img_list=None, 
        tar_img_list=None):
        super(CLIPLoss, self).__init__()

        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        #Normalize images to match clip input
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])])# + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
        self.preprocess2 = transforms.Compose([
            transforms.Resize(size=224, interpolation="bicubic"),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

        self.direction_loss = DirectionLoss(direction_loss_type)
        self.target_direction = None

        self.device = paddle.CUDAPlace(0) if paddle.device.is_compiled_with_cuda() else paddle.CPUPlace

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        """
        Utility to compose string starting from templates
        """
        return [template.format(text) for template in templates]

    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> paddle.Tensor:
        """
        Get Features of a string or template (defaults to only the class_str)
        """
        # templates = ["{}"]    
        template_text = self.compose_text_with_templates(class_str, templates)
        tokens = self.clip_tokenizer(template_text, padding=True, return_tensors="pd")

        text_features = self.clip_model.get_text_features(**tokens)

        if norm:
            text_features /= text_features.norm(axis=-1, keepdim=True)

        return text_features

    def encode_images(self, images: paddle.Tensor) -> paddle.Tensor:
        """
        Encode a list of images
        """
        images = self.preprocess(images)
        images = self.preprocess2(images[0])
        images = images[None, ...]
        
        images = {"pixel_values": images}
        return self.clip_model.get_image_features(**images)

    def get_image_features(self, img: paddle.Tensor, norm: bool = True) -> paddle.Tensor:
        """
        Get features of an Image
        """
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(axis=-1, keepdim=True)

        return image_features

    def oriforward(self, src_img: paddle.Tensor, src_class: str, tgt_img: paddle.Tensor, tgt_class: str):
        """
        compute CLIPLoss and combines them
        """

        src_txt_f = self.get_text_features(src_class)
        tgt_txt_f = self.get_text_features(tgt_class)

        txt_direction  = tgt_txt_f - src_txt_f
        txt_direction /= txt_direction.norm(axis=-1, keepdim=True)

        src_img_f = self.get_image_features(src_img)
        tgt_img_f = self.get_image_features(tgt_img)

        img_direction = tgt_img_f - src_img_f
        img_direction /= img_direction.norm(axis=-1, keepdim=True)

        return self.direction_loss(img_direction, txt_direction).mean()

    def forward(self, src_img: paddle.Tensor, src_class: str, tgt_img: paddle.Tensor, tgt_class: str):
        tgt_txt_token = self.clip_tokenizer([tgt_class], padding=True, return_tensors="pd")

        image = F.upsample(tgt_img, (224, 224), mode='bilinear')
        inputs = {
            "input_ids": tgt_txt_token['input_ids'],
            "pixel_values": image
        }

        output  = self.clip_model(**inputs)
        logits_per_image = output[0]

        similarity = 1 - logits_per_image / 100
        return similarity.mean()


    def get_img_f(self, img: paddle.Tensor, norm: bool = True):
        # images = F.upsample(img, (224, 224), mode='bilinear')
        images = self.preprocess(img)
        images = self.preprocess2(images[0])
        images = images[None, ...]

        vision_outputs = self.clip_model.vision_model(pixel_values=images)
        image_embeds = vision_outputs[1]
        image_embeds = paddle.matmul(image_embeds, self.clip_model.vision_projection)
        
        if norm:
            image_embeds = image_embeds / image_embeds.norm(axis=-1, keepdim=True)

        return image_embeds

    def get_txt_f(self, txt: paddle.Tensor, norm: bool = True):
        txt_token = self.clip_tokenizer([txt], padding=True, return_tensors="pd")
        text_outputs = self.clip_model.text_model(**txt_token)

        text_embeds = text_outputs[1]
        text_embeds = paddle.matmul(text_embeds, self.clip_model.text_projection)

        if norm:
            text_embeds = text_embeds / text_embeds.norm(axis=-1, keepdim=True)

        return text_embeds

    def artforward(self, src_img: paddle.Tensor, src_class: str, tgt_img: paddle.Tensor, tgt_class: str):
        src_txt_embeds = self.get_txt_f(src_class)
        tgt_txt_embeds = self.get_txt_f(tgt_class)

        src_img_embeds = self.get_img_f(src_img)
        tgt_img_embeds = self.get_img_f(tgt_img)

        img_dir = tgt_img_embeds - src_img_embeds
        txt_dir = tgt_txt_embeds - src_txt_embeds

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = paddle.matmul(txt_dir, img_dir, transpose_y=True) * logit_scale
        logits_per_image = logits_per_text.t()

        return (1 - logits_per_image / 100).sum()

    def d_forward(self, src_img: paddle.Tensor, src_class: str, tgt_img: paddle.Tensor, tgt_class: str):
        
        image = F.upsample(tgt_img, (224, 224), mode='bilinear')

        vision_outputs = self.clip_model.vision_model(pixel_values=image)
        image_embeds = vision_outputs[1]
        image_embeds = paddle.matmul(image_embeds, self.clip_model.vision_projection)


        tgt_txt_token = self.clip_tokenizer([tgt_class], padding=True, return_tensors="pd")
        text_outputs = self.clip_model.text_model(**tgt_txt_token)

        text_embeds = text_outputs[1]
        text_embeds = paddle.matmul(text_embeds, self.clip_model.text_projection)

        image_embeds = image_embeds / image_embeds.norm(axis=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(axis=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = paddle.matmul(text_embeds, image_embeds, transpose_y=True) * logit_scale
        logits_per_image = logits_per_text.t()

        return (1 - logits_per_image / 100).sum()


if __name__ == "__main__":
    import numpy as np

    clip_loss = CLIPLoss()

    I_src = paddle.ones([1, 3, 64, 64])
    I_tgt = paddle.zeros([1, 3, 64, 64])

    I_tgt = cv2.imread("/home/mengqingyue/Work/Nerf/baidu/ar/paddle-nerf/source/NeRF-Art/criteria/../logs/20230319_1.0nadaclip_2.0contrast_0.4perp_0.1patchnce64x5_0.1reg_0.1eikonal_fl3_true_vvv/imgs/val/predicted_rgb/00000020_0.png").transpose((2, 0, 1))
    I_tgt = paddle.to_tensor(I_tgt[None, ...].astype("float32"))
    print(I_tgt.shape)


    t_src = "human"
    t_tgt = "zombie"

    loss1 = clip_loss.nadaforward(I_src, t_src, I_tgt, t_tgt)
    print(loss1)

    loss2 = clip_loss.diyforward(I_src, t_src, I_tgt, t_tgt)
    print(loss2)
