#  !/usr/bin/env python3
"""
ContrastiveLoss methods
"""
from paddlenlp.transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from PIL import Image
import paddle
import paddle.vision.transforms as transforms
import paddle.nn.functional as F
import paddle.nn as nn

import random

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

class ContrastiveLoss(nn.Layer):
    """
    ContrastiveLoss Class
    """

    def __init__(self, margin=0.07, distance_type='euclidean', neg_texts=None):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.eps = 1e-9
        self.dis_type = distance_type
        if neg_texts is not None:
            self.neg_texts = open(neg_texts).readlines()
        else:
            self.neg_texts = open("criteria/neg_text.txt").readlines()
        self.neg_texts = [line.strip() for line in self.neg_texts]

        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        #Normalize images to match clip input
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])])
        self.preprocess2 = transforms.Compose([
            transforms.Resize(size=224, interpolation="bicubic"),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        """
        Utility to compose string starting from templates
        """
        return [template.format(text) for template in templates]

    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> paddle.Tensor:
        """
        Get Features of a string or template (defaults to only the class_str)
        """
        templates = ["{}"]
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

    def forward(self, I_tgt, t_tgt):
        """
        Args:
            I_tgt: the shape should be B[F].
            t_tgt: the shape should be B[F].
        """

        margin_tensor = paddle.to_tensor(self.margin)
        batch_size = I_tgt.shape[0]

        f_I_tgt = self.get_image_features(I_tgt)
        f_t_tgt = self.get_text_features(t_tgt)

        nominator = paddle.exp(paddle.nn.CosineSimilarity()(f_I_tgt, f_t_tgt) / margin_tensor)
        denominator_p2 = 0
        
        for t_neg in self.neg_texts:
            if t_neg != t_tgt:
                f_t_neg = self.get_text_features(t_neg, templates=["{}"])
                denominator_p2 += paddle.exp(paddle.nn.CosineSimilarity()(f_I_tgt, f_t_neg) / margin_tensor)
        loss = - paddle.log(nominator / (nominator + denominator_p2))
        return paddle.mean(loss)


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
        if type(txt) != list:
            txt = [txt]

        txt_token = self.clip_tokenizer(txt, padding=True, return_tensors="pd")
        text_outputs = self.clip_model.text_model(**txt_token)

        text_embeds = text_outputs[1]
        text_embeds = paddle.matmul(text_embeds, self.clip_model.text_projection)

        if norm:
            text_embeds = text_embeds / text_embeds.norm(axis=-1, keepdim=True)

        return text_embeds

    def newforward(self, I_tgt, t_tgt):
        """
        Args:
            I_tgt: the shape should be B[F].
            t_tgt: the shape should be B[F].
        """

        margin_tensor = paddle.to_tensor(self.margin)

        f_t_tgt = self.get_txt_f(t_tgt)
        f_I_tgt = self.get_img_f(I_tgt)

        nominator = paddle.exp(paddle.nn.CosineSimilarity()(f_I_tgt, f_t_tgt) / margin_tensor)
        
        denominator_p2 = 0
        for t_neg in self.neg_texts:
            if t_neg != t_tgt:
                f_t_neg = self.get_txt_f(t_neg)
                denominator_p2 += paddle.exp(paddle.nn.CosineSimilarity()(f_I_tgt, f_t_neg) / margin_tensor)
        loss = - paddle.log(nominator / (nominator + denominator_p2))
        return paddle.mean(loss)

if __name__ == "__main__":

    import cv2

    contrast_loss = ContrastiveLoss(neg_texts="neg_text.txt")

    I_tgt = paddle.zeros([1, 3, 64, 64])

    I_tgt = cv2.imread("../logs/20230406_neus_fangzhou_1.0artclip_pre12_0.2percep_fangzhou_painting_oil_on_canvas_Vincent_van_gogh_self-portrait_style/imgs/val/predicted_rgb/00000400_0.png").transpose((2, 0, 1))
    I_tgt = paddle.to_tensor(I_tgt[None, ...].astype("float32"))

    t_tgt = "zombie"

    loss = contrast_loss(I_tgt, t_tgt)
    print(loss)


