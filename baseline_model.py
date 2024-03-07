import PIL
from PIL import Image

def	baseline_bicubic(img):
		new_img = img.resize((128,128),Image.BICUBIC)
		return new_img