import requests
from bs4 import BeautifulSoup
import urllib
import os

# 目標網站的 URL
url = 'https://www.google.com/search?q=ct+scan+of+brain&rlz=1C1VDKB_zh-TWTW952TW952&sxsrf=APwXEdee0IzbCvxI5V8fwl1Lmrdft5iDkQ:1685018701684&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjflJ3pv5D_AhW2SfUHHSoGDbQQ_AUoAXoECAEQAw&biw=1552&bih=736&dpr=1.65'

folder = '.\\CT_scan_of_brain'
os.makedirs(folder, exist_ok=True)

response = requests.get(url)
cnt = 0
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    image_elements = soup.find_all('img')
    # print(folder)
    for image in image_elements:
        image_url = urllib.parse.urljoin(url, image['src'])
        
        image_data = requests.get(image_url).content
        image_name = image_url.split('/')[-1]
        image_path = os.path.join(folder, "Image" + str(cnt) + ".jpg")
        cnt += 1
        # print(image_path)
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        # print('Download done')
        

else:
    print('Request Error: ', response.status_code)
