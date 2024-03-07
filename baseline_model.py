import PIL
from PIL import Image
import numpy as np

def	baseline_bicubic(img_arr, scale=3):
		# Convert array back into PIL Image
		img = Image.fromarray(np.uint8(np.transpose(img_arr, (2, 1, 0)) * 255))

		detected_width = img_arr.shape[2]
		detected_height = img_arr.shape[1]
		new_img = img.resize((detected_width * scale,detected_height * scale), Image.BICUBIC)

		# Convert PIL Image back into array
		return np.transpose(np.array(new_img) / 255.0, (2, 1, 0))