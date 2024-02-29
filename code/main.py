import functions
import subprocess
import pandas as pd
import numpy as np
import chardet
import spacy
from spacy import displacy
import unicodedata
import networkx as nx
import os
import re
from pyvis.network import Network
import community.community_louvain as cl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

print('Scraping character and book data from Tolkien wiki: \'https://lotr.fandom.com/wiki/Main_Page\'')
print('Please wait...')
print('----------------------------------------------------------------------------------------------')

#Scrape character and book data from the Tolkien Wiki
subprocess.run(['python', 'scrape.py'])

print('Webscraping has completed...')
print('Book and character data has been saved.')
print('----------------------------------------------------------------------------------------------')


print('Downloading Spacy\'s small english language model...')
print('----------------------------------------------------------------------------------------------')
#Download the spacy small english language model
spacy.cli.download('en_core_web_sm')

print('Loading Spacy\'s small english language model...')
print('----------------------------------------------------------------------------------------------')
#Load spacy's english language model
NER = spacy.load('en_core_web_sm')

#Increase the max_length limit of spacy's NER model
NER.max_length = 3000000  # or any value greater than the length of your text

#Store the directory that the book text files are in
data_directory = os.path.join(functions.work_dir, 'data')

#Store the book text file names without extentions to a list
book_titles = functions.get_text_file_names(data_directory)

#Store the book text files and their respective contents to a dictionary
books = functions.read_text_files_with_encoding(data_directory)

print('Looping through books for NER, relationship extraction, character insights, and visualizations')
print('----------------------------------------------------------------------------------------------')
#Loop through the books
for index, book in enumerate(books):
    #Store contents of the book
    book_text = books.get(book)
    
    #Store the title of book
    book_title = book_titles[index]
    
    #Process the text with model
    book_doc = NER(book_text)
    
    #Load character data from pickle file to pandas df
    character_df = pd.read_pickle(functions.work_dir + '/data/characters.pkl')
    
    
    clean_character_df = functions.clean_tidy_book_characters(character_df)
    print('Cleaned character names for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    functions.modify_add_character_names_for_NER(clean_character_df)
    print('Modified character names for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
   
    sent_entity_df = functions.extract_book_entities(book_doc)
    print('Extracted entities from book sentences for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    sent_entity_filtered_df = functions.clean_modify_entities(sent_entity_df, clean_character_df)
    print('Cleaned, modified, and filtered entities for character names only from book sentences for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    
    relationship_df = functions.extract_relationships(sent_entity_filtered_df, clean_character_df)
    print('Extracted character relationships for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    relationship_filtered_df =  functions.clean_modify_relationships(relationship_df)
    print('Cleaned, modified, and filtered  character relationships data for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    

    #Build the network
    G = nx.from_pandas_edgelist(relationship_filtered_df,
                                source = 'source',
                                target = 'target',
                                edge_attr = 'value',
                                create_using = nx.Graph())
    
    print('Built graph network for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    functions.visualize_basic_network(G, book_title)
    print('Saved basic graph network for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    adv_network = functions.build_adv_network(G, clean_character_df)
    
    print('Built advanced graph network for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    adv_network.write_html(functions.work_dir + '/assets/img/' + book_title +'_adv_network.html')
    print('Saved advanced graph network for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    
    #Use the louvain community algorithm to determine the optimal community partitions
    communities = cl.best_partition(G)
    
    adv_comm_network = functions.build_adv_comm_net(G, clean_character_df, communities)
    print('Built advanced community graph network for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    #Save figure
    adv_comm_network.write_html(functions.work_dir + '/assets/img/' + book_title +'_adv_comm_network.html')
    print('Saved advanced community graph network for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    
    #Degree centrality
    degree_dict = nx.degree_centrality(G)
    degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['centrality'])
    functions.visualize_character_insights(degree_df, 'degree', book_title)
    print('Saved degree centrality visual for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    
    # Betweenness centrality
    betweenness_dict = nx.betweenness_centrality(G)
    betweenness_df = pd.DataFrame.from_dict(betweenness_dict, orient='index', columns=['centrality'])
    functions. visualize_character_insights(betweenness_df, 'betweenness', book_title)
    print('Saved betweeness centrality visual for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    # Closeness centrality
    closeness_dict = nx.closeness_centrality(G)
    closeness_df = pd.DataFrame.from_dict(closeness_dict, orient='index', columns=['centrality'])
    functions.visualize_character_insights(closeness_df, 'closeness', book_title)
    print('Saved closeness centrality visual for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    
    #Set the style of the plot using seaborn
    sns.set_style('darkgrid')

    #Plot histogram of the distribution of values
    plt.figure(figsize=(10, 6))
    plt.hist(relationship_filtered_df['value'], bins=100, color='skyblue', edgecolor='black')

    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Relationship Strength', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    #Add a vertical line for the mean value
    mean_value = relationship_filtered_df['value'].mean()
    median_value = relationship_filtered_df['value'].median()
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='yellow', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(functions.work_dir + '/assets/img/' + book_title + '_Relationship_Strength.png')
    
    print('Saved relationship strength visual for ' + book_title)
    print('----------------------------------------------------------------------------------------------')
    
    
    
    #Create a new column 'pair' by combining 'source' and 'target' columns
    relationship_filtered_df['pair'] = relationship_filtered_df['source'] + '-' + relationship_filtered_df['target']

    #Group by 'pair' and calculate the sum of values
    pair_sum = relationship_filtered_df.groupby('pair')['value'].sum()

    #Sort the pairs by the sum of values and select the top 10
    top_10_pairs = pair_sum.sort_values(ascending=False).head(10)

    #Define a custom color palette using the Viridis colormap
    colors = cm.viridis(top_10_pairs / top_10_pairs.max())

    #Plot the top 10 pairs with improved aesthetics
    plt.figure(figsize=(12, 8))
    top_10_pairs.plot(kind='bar', color=colors)

    plt.xlabel('Relationship Pair', fontsize=14)
    plt.ylabel('Strength of Relationship', fontsize=14)
    plt.title('Top 10 Character Pairs by Relationship', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    #Add annotations with values on top of bars
    for i, v in enumerate(top_10_pairs):
        plt.text(i, v + 0.5, str(round(v, 2)), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(functions.work_dir + '/assets/img/' + book_title + '_top_10_Relationship_Pairs.png')
    
    print('Saved top 10 relationships visual for ' + book_title)
    print('----------------------------------------------------------------------------------------------')