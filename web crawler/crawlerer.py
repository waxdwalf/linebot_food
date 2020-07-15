import codecs
import csv
import random
import threading
import time
import requests
from bs4 import BeautifulSoup
# 葷食食譜末頁: https://www.ytower.com.tw/recipe/pager.asp?VEGETARIA=%B8%A7%AD%B9&IsMobile=0&page=1910
# 素食食譜末頁: https://www.ytower.com.tw/recipe/pager.asp?VEGETARIA=%AF%C0%AD%B9&IsMobile=0&page=447
csvFile = codecs.open('楊桃美食網葷食食譜.csv', 'w', encoding='utf-8-sig')
csvWriter = csv.writer(csvFile)
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
}
page_number = 1910

for i in range(1,page_number+1):
    url_V = "https://www.ytower.com.tw/recipe/pager.asp?VEGETARIA=%B8%A7%AD%B9&IsMobile=0&page=" + str(i)
    # print(url_V)
    page_res = requests.get(url_V, headers=headers)
    page_soup = BeautifulSoup(page_res.text, 'html.parser')
    # 起手式，爬取目錄頁面

    title = page_soup.select('li')
    # 爬取目錄頁面的標題

    time.sleep(random.randint(1, 5))
    # 每爬一頁休息1~5秒

    for t in title:
        # print('--------')
        try:
            recipe_title = t.select('li a')[2]
            # print(recipe_title.text)
            # 取得該食譜名稱
            recipe_title_html = "https://www.ytower.com.tw/" + t.select('a')[0]['href']
            # print(recipe_title_html)
            # 取得該食譜網址

            recipe_picture_html = t.img['src']
            # 取得該食譜圖片網址

            recipe_title_res = requests.get(recipe_title_html, headers=headers)
            recipe_title_res.encoding = 'big5'
            # 該網頁為BIG5 編碼，需先解碼
            recipe_title_soup = BeautifulSoup(recipe_title_res.text, 'html.parser')
            # 取得該食譜頁面

            recipe_ingredient_name = recipe_title_soup.select('span[class="ingredient_name"] a')
            recipe_ingredient_amount = recipe_title_soup.select('span[class="ingredient_name"] span')
            itgredient = {}
            for t, i in zip(recipe_ingredient_name, recipe_ingredient_amount):
                itgredient[t.text] = i.text
            rowData = [recipe_title.text, recipe_title_html, itgredient, recipe_picture_html]
            csvWriter.writerow(rowData)
        except :
            print("ERROR")

            # csvFile.close()


            # with open('article_title.txt', 'w', encoding='utf-8') as w:
            #     w.write(ingredient_name)






