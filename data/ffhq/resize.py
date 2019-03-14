import cv2, time, random, sys, gc
import os
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image



def resize(image_file):
	if ".png" not in image_file:
		return
	im = Image.open("images512x512/"+image_file)
	im_resized = im.resize((256,256), Image.ANTIALIAS)
	im_resized.save("images256x256/"+image_file, "PNG")
	return

if __name__ == "__main__":
	imgList = os.listdir("images512x512/")
	imgList.sort()


	threads = 28
	chunking = 20
	p = Pool(threads)
	#gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
	fh = open("save_resize.txt",'r')
	start = int(fh.readlines()[0])
	fh.close()
	for n,i in enumerate(tqdm(range(start,len(imgList),threads*chunking))):
		p.map(resize, imgList[i:i+threads*chunking])
		with open("save_resize.txt",'w') as save:
			save.writelines([str(i+threads*chunking)])
