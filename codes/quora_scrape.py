import requests
from bs4 import BeautifulSoup

url = 'https://www.quora.com/answer'
html = requests.get(url)

s = BeautifulSoup(html.content, 'html.parser')

que = s.find('div', class_="q-box qu-borderBottom")

i = 0
j = len(que)
print(j)

while i < j:
    print(que[i].text)
