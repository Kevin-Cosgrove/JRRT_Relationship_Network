# Tolkien Universe Text Analytics Project

## A **Python** project that leverages Selenium, Pandas, Spacy, Networkx, Pyvis, Community, Matplotlib, Seaborn, and others to extract character relationships, communities, and insights from J.R.R. Tolkien's literary masterpieces: *The Silmarillion*, *The Hobbit*, and *The Lord of the Rings*

This project has 3 main python files:
- main.py is the code that runs the project from start to finish
- scrape.py is the code that web scrapes book and character data
- functions.py is a custom script that contains all of the methods that control how the data is cleaned, tidied, modified, visualized, etc.

This project also has 2 Jupyter Notebooks files:
- Web_Scraping_Characters.ipynb is the notebook that contains the code to scrape the book names and their respective character names from the internet
- LOTR_Relationship_Network_Ex.ipynb is the notebook that contains a step by step, very detailed documentation on how to perform NER, data cleaning, and identifying relationships, communities, and character insights using *The Lord of the Rings* as an example text.

This project accomplishes the following:
- Writes a web scraping script using **Selenium** to pull book information and their respective characters from the [Tolkien Wiki](https://lotr.fandom.com/wiki/Category:Characters)
- Cleans and tidies character data using **Pandas**
- Extracts entities using **Spacy**'s small English language model
- Cleans and tidies entities from the text using **Pandas**
- Defines character relationships from the text using entities extracted from previous NER tasks
- Builds network graphs and explores character insights from character and relationship data using **Networkx**, **Pyvis**, **Community**, **Matplotlib**, and **Seaborn**  

![LOTR Network Graph](https://github.com/Kevin-Cosgrove/JRRT_Relationship_Network/blob/main/assets/img/read_me_graph.png)

## How to tweak this project for your own uses
Since Tolkien's works are a passion of mine, I'd encourage you to clone and rename this project to use for your own purposes. It's a good introduction to text analytics and network graph visualization.
I also wrote this code in a way where a user only needs to edit the variables in functions.py if they want to webscrape and process book series from other wikis.

## Find a bug?
If you found an issue or would like to submit an improvement to this project, please submit an issue using the issues tab above. If you would like to submit a PR with a fix, reference the issue you created!

## Known issues (Work in progress)
This project is no longer ongoing and to my knowledge does not have any issues.

## Data sources/inspiration
Below is where I was able to get the books in text format:
- *[The Silmarillion](https://archive.org/stream/TheSilmarillionIllustratedJ.R.R.TolkienTedNasmith/The%20Silmarillion%20%28Illustrated%29%20-%20J.%20R.%20R.%20Tolkien%3B%20Ted%20Nasmith%3B_djvu.txt)*
- *[The Hobbit](https://archive.org/stream/dli.ernet.474126/474126-The%20Hobbit%281937%29_djvu.txt)*
- *[The Lord of the Rings](https://www.kaggle.com/datasets/ashishsinhaiitr/lord-of-the-rings-text)*

I was inspired to do this project after seeing a youtube video series on text analysis of *The Witcher* book series. Please see links to the videos and the creator's github profile below:
- [Youtube ep. 1](https://www.youtube.com/watch?v=RuNolAh_4bU)
- [Youtube ep. 2](https://www.youtube.com/watch?v=fAHkJ_Dhr50)
- [Github](https://github.com/thu-vu92)

## Like this project?
Please consider following me on [Github](https://github.com/Kevin-Cosgrove) and link with me on [LinkedIn](https://www.linkedin.com/in/kevin-j-cosgrove)
