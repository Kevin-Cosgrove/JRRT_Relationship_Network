import functions
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By


#Initialize the Chrome driver with ChromeDriverManager
driver_service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()

#Create driver object for webscraping
driver = webdriver.Chrome(service=driver_service, options=options)

#Go to the wiki page for Tolkien universe
page_url = functions.page_url
driver.get(page_url)

#Store each book wiki page to a unique variable
#Searching page_url by link text
hobbit_page = driver.find_element(By.LINK_TEXT, functions.hobbit_character_page)
lotr_page = driver.find_element(By.LINK_TEXT, functions.lotr_character_page)
silmarillion_page = driver.find_element(By.LINK_TEXT, functions.silmarillion_character_page)

#Store book character pages in a list for later looping
book_pages =[hobbit_page, lotr_page, silmarillion_page]

#Store meta data for the book character pages to a list of dictionaries
books=[]
for book in book_pages:
    book_url = book.get_attribute('href')
    book_name = book.text
    books.append({'book_name': book_name, 'url': book_url})
    

#Store characters from each book to a list of dictionaries
characters_list = []

#Loop through each book to scrape character names for the respective book's webpage
for book in books:
    driver.get(book['url'])
    character_elems = driver.find_elements(By.CLASS_NAME, 'category-page__member-link')
    for elem in character_elems:
        characters_list.append({'book': book['book_name'], 'character': elem.text})
        
#Close driver
driver.close()

#Quit driver
driver.quit()

#Store character_list to a pandas dataframe
df = pd.DataFrame(characters_list)

#Save df to a pickle file for later use
df.to_pickle(functions.work_dir + '/data/characters.pkl')