from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import os

path = './ofa_image-caption_coco_large_en'

img_dir = './val2017/'

img_caption = pipeline(Tasks.image_captioning, model=path, device='gpu:0')

list_caption = []

for name in os.listdir(img_dir):
    img = img_dir + name
    list_caption.append([img,img_caption(img)[OutputKeys.CAPTION][0]])

with open('./coco_captions.txt', encoding='utf-8', mode='w') as f:
    for i in list_caption:
        for j in i:
            f.write(str(j)+' ')
        f.write('\n')
