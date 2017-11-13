import requests
import re
import base64
import numpy as np
import os
from bs4 import BeautifulSoup

r = requests.get('https://unicode.org/emoji/charts/full-emoji-list.html')
html = r.text

soup = BeautifulSoup(html, 'html.parser')

emojis = soup.find_all('img', {'class': 'imga'})

if not os.path.exists('data/'):
	os.makedirs('data/')

i = 0
for emoji in emojis:
	image_data = emoji.get('src')
	imgstr = re.search(r'base64,(.*)', image_data).group(1)
	output = open('data/{}.png'.format(i), 'wb')
	output.write(base64.b64decode(imgstr.encode()))
	output.close()
	i += 1