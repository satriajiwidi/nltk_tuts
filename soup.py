import bs4 as bs
import urllib.request as req

source = req.urlopen('https://satriajiwidi.com').read()
soup = bs.BeautifulSoup(source, 'lxml')

print(soup.title.string)