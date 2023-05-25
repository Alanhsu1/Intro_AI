import requests
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/joke/index.html"

for i in range(3):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    sel = soup.select('div.title a')
    u = soup.select('div.btn-group.btn-group-paging a')
    # print(u)
    print("本頁的URL為" + url)
    url = 'https://www.ptt.cc' + u[1]['href']
    print(sel)

for s in sel:
    print(s['href'], s.text)