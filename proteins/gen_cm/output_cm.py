import numpy as np
from math import sqrt
import os, re
from matplotlib import pyplot as plt
from PIL import Image

ones = {}
pwd = os.path.dirname(os.path.realpath(__file__)) + '/'

with open(pwd + 'sequences.txt', 'r') as r_open:
	sequences = [line.strip() for line in r_open.readlines()]

def make_txt(sample, isSample=False):
	a = np.load(sample)[0,0,:,:]

	if not isSample:
		a = a.astype(int)

	a1, a2 = os.path.splitext(sample)
	yolo = a1[-1]

	if not isSample:
		ones[yolo] = np.sum(a)
		threshold = 0.5
	else:
		one = ones[yolo]
		threshold = (sorted(a.flatten().tolist())[-one]+sorted(a.flatten().tolist())[-one-1])/2
	a = (a>threshold).astype(int)
	np.savetxt(pwd + a1 + '.txt', a, '%d')

def make_imgs(file):
	im = Image.fromarray((-np.loadtxt(file.replace('.npy', '.txt'))+1).astype('uint8')*255)
	im.save(pwd + 'outputs/' + file.replace('.npy', '.jpeg'))


references = [filename for filename in os.listdir(pwd) if re.search('reference_.\.npy', filename)]
samples = [reference.replace('reference', 'sample') for reference in references]

for reference in references:
	make_txt(reference)
for sample in samples:
	make_txt(sample, True)
for refsam in zip(references, samples):
	ref, sam = refsam
	make_imgs(ref)
	make_imgs(sam)
