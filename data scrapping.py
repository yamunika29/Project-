Data scrapping:

input-
import requests
from bs4 import BeautifulSoup
import csv
import os
import subprocess
response = requests.get('https://books.toscrape.com/catalogue/category/books/travel_2/index.html')
soup = BeautifulSoup(response.text, 'html.parser')
title = soup.find('title')
course_name = title.get_text().strip().split('|')[0].strip()
file_name = 'travel.csv'
with open(file_name, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Book Name', 'Rating', 'Price'])
    travel_books = soup.find_all('article', attrs={'class': 'product_pod'})
    print(f"Total books found: {len(travel_books)}")

    for book in travel_books:
        travel_book_name = book.find('h3').get_text().strip()
        rates = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
        travel_book_rating = rates[book.find('p', attrs={'class': 'star-rating'}).get('class')[1]]
        travel_book_price = book.find('div', attrs={'class': 'product_price'}).find('p', {'class': "price_color"})
        travel_book_price = float(travel_book_price.get_text().split('Â£')[1])
        writer.writerow([travel_book_name, travel_book_rating, travel_book_price])
if os.name == 'nt':
    os.startfile(file_name)
elif os.name == 'posix':
    try:
        subprocess.call(['open', file_name])
    except FileNotFoundError:
        subprocess.call(['xdg-open', file_name])
