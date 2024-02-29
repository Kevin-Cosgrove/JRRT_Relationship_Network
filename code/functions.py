import pandas as pd
import numpy as np
import chardet
import unicodedata
import networkx as nx
import os
import re
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns

#Establish working directory
work_dir = 'c:/users/myxna/desktop/datascienceprojects/lotr'


#Wiki page for Tolkien universe
page_url = 'https://lotr.fandom.com/wiki/Category:Characters'

#Wiki pages for book character list
hobbit_character_page = 'Category:The Hobbit characters'
lotr_character_page = 'Category:The Lord of the Rings characters'
silmarillion_character_page = 'Category:The Silmarillion characters'


#Characters that need to be removed from character
remove_character_list = ['Captain of the guard', 
                         'Category:Characters that have appeared in the Hobbit', 
                         'the Lord of the Rings', 
                         'Master of Lake-town', 
                         'Great Goblin', 
                         'Great grey chief wolf', 
                         'Chief of the guards Woodland Realm', 
                         'Councilors Lake-town', 
                         '', 
                         'Category:Major characters The Lord of the Rings', 
                         'Category:Minor characters The Lord of the Rings', 
                         'Witch-king\'s winged steed'
                         ]

nickname_mapping = {
    'Morgoth': 'Melkor',
    'Samwise': 'Sam',
    'Meriadoc': 'Merry',
    'Perigrin': 'Pippin',
    'Gandalf': 'Mithrandir',
    'Gollum': 'Sméagol',
    'Aragorn': 'Strider',
    'Sauron': 'Dark Lord'
}


#Set window size to define what a relationship is
window_size = 5




def get_text_file_names(directory):
    """
    This function stores file names from an inputted directory without .txt extention to list.

    Parameters:
        - directory: The directory where the text files are stored.

    Returns:
        - result: The function returns a list containing the text file names without the .txt extenstion.
    """
    
    
    #Get a list of all files in the directory
    all_files = os.listdir(directory)
    #Filter only the text files
    text_files = [file.split('.')[0] for file in all_files if file.endswith('.txt')]
    
    return text_files



def detect_encoding(file_path):
    """
    This function determines the type of encoding the input text file is written in and assigns a confidence to that classification.

    Parameters:
        - file_path: The path of the file.

    Returns:
        - result: The function returns the encoding and the associated confidence.
    """
    
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        return encoding, confidence
    
    

def read_text_file(file_path, encoding):
    """
    This function receives an input file path and its respective encoding, reads the text file, and returns the text.

    Parameters:
        - file_path: The path of the file.
        - encoding: the encoding of the text file

    Returns:
        - result: The function returns the document text.
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()
    
    
    
def read_text_files_with_encoding(directory):
    """
    This function uses the helper functions detect_encoding and read_text_file to loop through an inputted directory and read and return the text file contents of that directory.

    Parameters:
        - directory: The directory where the text files are located.

    Returns:
        - result: The function returns a dictionary whose keys are the names of the file and values are their respective contents.
    """
    
    text_files = [file.path for file in os.scandir(directory) if file.name.endswith('.txt')]
    file_contents = {}
    for file_path in text_files:
        encoding, confidence = detect_encoding(file_path)
        if confidence > 0.5:  # You can adjust the confidence threshold as needed
            file_contents[file_path] = read_text_file(file_path, encoding)
        else:
            print(f"Warning: Low confidence in encoding detection for file '{file_path}'")
    return file_contents



def split_rows(df):
    """
    This function splits the rows of an inputted DataFrame that contain more than 1 character name.

    Parameters:
        - df: A Pandas DataFrame that contains a column called 'character' that contains character names.

    Returns:
        - result: The function returns a DataFrame whose rows only contain one character name per row.
    """
    
    new_rows = []
    for index, row in df.iterrows():
        names = row['character'].replace(' and ', ',').split(',')
        for name in names:
            new_row = row.copy()
            new_row['character'] = name.strip()
            new_rows.append(new_row)
    return pd.DataFrame(new_rows)



def clean_tidy_book_characters(df):
    """
    This function takes several steps to clean and tidy a Pandas DataFrame that contains character names. 
    It ensures that there is one character name per row using the split_rows() function, removes special characters from names,
    removes characters that are not named or not valid, and returns the clean DataFrame

    Parameters:
        - df: A Pandas DataFrame that contains a column called 'character' that contains character names.

    Returns:
        - result: The function returns a DataFrame whose rows only contain one character name per row.
    """
    
    #Separate rows with multiple names in character_df
    df = split_rows(df)
    #Remove special characters from character names
    df['character'] = df['character'].apply(lambda x: re.sub('[\(}.*?{\)]', '', x))
    #Remove Character entries that do not contain valid characters or characters that are not named in the book,
    #for example: row 9, "Captain of the guard"
    remove_character_list = ['Captain of the guard', 'Category:Characters that have appeared in the Hobbit', 'the Lord of the Rings', 'Master of Lake-town', 'Great Goblin', 'Great grey chief wolf', 'Chief of the guards Woodland Realm', 'Councilors Lake-town', '', 'Category:Major characters The Lord of the Rings', 'Category:Minor characters The Lord of the Rings', 'Witch-king\'s winged steed', ]

    # Filter the DataFrame to keep only rows where 'character' does not contain strings from remove_character_list
    df = df[~df['character'].isin(remove_character_list)]
    
    return df
    
    

def replace_accented_characters(row):
    """
    This function replaces accented characters for characters from 'Category:The Hobbit characters' and 'Category:The Silmarillion characters'.
    
    Parameters:
        - row: A Pandas DataFrame row that contains column called 'book' and a column called 'character'.

    Returns:
        - result: The function returns a DataFrame cell for the column 'character' without accented characters.
    """
    if row['book'] in ['Category:The Hobbit characters', 'Category:The Silmarillion characters']:
        character_list = list(row['character'])
        for i, char in enumerate(character_list):
            if unicodedata.category(char)[0] == 'L':
                # Replace special characters with their non-accented equivalents
                character_list[i] = unicodedata.normalize('NFKD', char).encode('ascii', 'ignore').decode('utf-8')
        return ''.join(character_list)
    else:
        return row['character']
    
    

def modify_add_character_names_for_NER(df):
    """
    This function takes several steps to modify and add names for characters for NER. It uses the replace_accesnted_characters() helper function to remove accented letters for the appropraite books,
    creates a new colunm called character_firstname that contains the characters first name, and lastly, 
    it creates a column called "character_nickname" that replaces uses character first name with the nickname that is specified in the nickname_mapping variable.
    
    Parameters:
        - df: A Pandas DataFrame  that contains a column called ''character' that contains a character name.

    Returns:
        - result: The function does not return anything, but modifies the inputted Pandas DataFrame to add two new columns containing a character's first name and nickname.
    """
    
    #Apply the function to create a new column 'modified_character'
    df['character_modified'] = df.apply(replace_accented_characters, axis=1)
    #Create a column in the character_df containing the character's first name
    df['character_firstname'] = df['character_modified'].apply(lambda x: x.split(' ', 1)[0])
    
    #Create an empty list to store nicknames
    nicknames = []
    #Iterate over the 'character_firstname' column
    for first_name in df['character_firstname']:
        #Check if the first name is in the nickname mapping dictionary
        if first_name in nickname_mapping:
            #If it is, append the corresponding nickname
            nicknames.append(nickname_mapping[first_name])
        else:
            #If not, append an empty string
            nicknames.append('')

    #Add the list of nicknames as a new column to the DataFrame
    df['character_nickname'] = nicknames



def extract_book_entities(book_doc):
    """
    This function extracts all sentences from book text and extracts entities within those sentences and stores data in a Pandas DataFrame and returns that DataFrame. 
    
    Parameters:
        - book_doc: A string of text, book contents.

    Returns:
        - result: The function returns a Pandas DataFrame that contains each sentence from the book and that sentence's recognized entities.
    """
    
    #Create a list to be used to hold sentences and entities contained within respective sentences
    sent_entity_df= []

    #Loop over sentences in books
    for sent in book_doc.sents:
        #Store entities from individual sentence in list
        entity_list = [ent.text for ent in sent.ents]
        #Add sentence and entities to dataframe
        sent_entity_df.append({'sentence': sent, 'entities': entity_list})
    
    #Convert list of dictionaries to dataframe    
    sent_entity_df = pd.DataFrame(sent_entity_df)
    
    return sent_entity_df
    

#Function to filter the sentence-entity dataframe for sentences that only contain 
#character full names, first names, or nicknames
def filter_entity(ent_list, df):
    """
    This function filters the sentence-entity DataFrame for sentences that only contain character full names, first names, or nicknames. 
    The entities come from the sentence-entity DataFrame and the character names come from the character DataFrame
    
    Parameters:
        - ent_list: A list of entities from a single sentence of the book.
        - df: A Pandas DataFrame that contains character names

    Returns:
        - result: The function returns a filtered list of entities that only contain character names specified in the inputted character DataFrame
    """   
    
    return [ent for ent in ent_list 
            if ent in list(df.character_modified)
            or ent in list(df.character_firstname)
            or ent in list(df.character_nickname)
           ]



def clean_modify_entities(sent_entity_df, character_df):
    """
    This function filters the sentence-entity DataFrame for sentences that only contain character full names, first names, or nicknames. 
    The entities come from the sentence-entity DataFrame and the character names come from the character DataFrame. 
    Furthermore, the function also creates a new dataframe where only sentences with character entities are present and tidies the information.
    The function outputs a cleaned and modified Pandas DataFrame of sentences with character entities.
    
    Parameters:
        - sent_entity_df: A Pandas DataFrame that ist of entities from a single sentence of the book.
        - character_df: A Pandas DataFrame that contains character names.

    Returns:
        - result: The function returns a cleaned and modified Pandas DataFrame of sentences with character entities..
    """
    
    #Create a new column for the sentence-entity dataframe that captures only the 
    #specified character names from the entitiy list
    sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(lambda x: filter_entity(x, character_df))
    #Create a new DataFrame that contains only sentences where at least one recognized entity is a
    #specified character
    sent_entity_df_filtered = sent_entity_df[sent_entity_df['character_entities'].map(len) > 0]
    #Split rows of the sentence-entity DataFrame where multiple characters are
    #present in the same sentence
    sent_entity_df_filtered.loc[:, 'character_entities'] = sent_entity_df_filtered['character_entities'].apply(lambda x: [item.split()[0] for
                                                                                                                      item in x])
    return sent_entity_df_filtered



def extract_relationships(entity_df, character_df):
    """
    This function extracts character relationships using entities found within the specified window_size.
    This is done using the sentence-entity DataFrame and the character DataFrame.
    This outputs a relationshp DataFrame containing an source and a target. A relationship can only have two characters. 
    
    Parameters:
        - entity_df: A Pandas DataFrame that contains relevant book sentences containing at least one character.
        - character_df: A Pandas DataFrame that contains character names.

    Returns:
        - result: The function returns a Pandas DataFrame that contains the source and target of a relationship.
    """
    
    #Create a list to house relationships between characters that will become a DataFrame
    relationships = []

    #Loop through the sentence-entity DataFrame
    for i in range(entity_df.index[-1]):
        #Set end of window, ensure the window does not exceed dataframe index
        end_i = min(i + window_size, entity_df.index[-1])
        #Create list of characters that fall in the window size
        char_list = sum((entity_df.loc[i: end_i].character_entities), [])
    
        #Create a new list of only unique characters from char_list
        char_unique = [char_list[i] for i in range(len(char_list))
                  if (i == 0) or char_list[i] != char_list[i-1]]
        #Loop through char_unique to add unique relationships to relationship dataframe.
        #Ensures character does not have a relationship with themselves
        if len(char_unique) > 1:
            for idx, a in enumerate(char_unique[:-1]):
                b = char_unique[idx + 1]
                relationships.append({'source': a, 'target': b})
    
    #Store list of dictionaries as DataFrame
    relationship_df = pd.DataFrame(relationships)

    return relationship_df


def remove_self_relationships(df):
    """
    This function removes relationships where characters have relationships with themselves.
    This is done because it is not meaningful to the overall analysis.
    
    Parameters:
        - df: A Pandas DataFrame that contains character relationship data.

    Returns:
        - result: The function returns a Pandas DataFrame containing character relationships where the source and target are not the same character.
    """
    for index, row in df.iterrows():
        if row['source'] == row['target']:
            df.drop(index, inplace=True)
    return df




def clean_modify_relationships(df):
    """
    This function cleans and modifies the inputted relationship Pandas DataFrame. It alphabetically sorts the relationshps. 
    It also adds a column with an int of 1 for counting the number of times each relationship occurs.
    It removes relationships where characters have relationships with themselves.
    It also adds a column with an int of 1 for counting the number of times each relationship occurs.
    The function returns a cleaned and modified Pandas DataFrame containing relationship data.
    
    Parameters:
        - df: A Pandas DataFrame that contains character relationship data.

    Returns:
        - result: The function returns a Pandas DataFrame containing cleaned and modified character relationship data.
    """
    
    #Alphabetically sort the relationships 
    df = pd.DataFrame(np.sort(df.values, axis = 1), columns = df.columns)
    #Create a new column called 'value' to count each relationship as 1 instance.
    df['value'] = 1
    #Group duplicate relationships and sum the values to quantify the strength of the relationship
    df = df.groupby(['source', 'target'], sort = False, as_index = False).sum()
    #Invert the nickname mapping dictionary
    inverse_mapping = {v: k for k, v in nickname_mapping.items()}

    #Replace values in 'source' and 'target' columns with dictionary keys
    df['source'] = df['source'].map(inverse_mapping).fillna(df['source'])
    df['target'] = df['target'].map(inverse_mapping).fillna(df['target'])
    
    # Apply the function to remove rows where nickname and firstname are self relating
    relationship_filtered_df = remove_self_relationships(df)

    return relationship_filtered_df



def visualize_basic_network(graph, book_title):
    """
    This function visualizes a basic network graph from the inputted Networkx graph object and saves the visualization with the inputted book title.
    
    Parameters:
        - graph: A Networkx graph object.

    Returns:
        - result: The function visualizes and saves the image to the specified directory.
    """
    
    #Set visualization parameters for basic network
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, with_labels = True, node_color = 'skyblue', edge_cmap = plt.cm.Blues, pos = pos)
    #Save figure
    plt.savefig(work_dir + '/assets/img/' + book_title + '_basic_network.png')
    

def update_node_labels(graph, character_dataframe):
    """
    This function updates the nodes of the inputted Networkx graph object with the original character names from the data source.
    
    Parameters:
        - graph: A Networkx graph object.
        - character_dataframe: A Pandas DataFrame containing character name data.

    Returns:
        - result: The function updates the nodes of the inputted Networkx graph object with the original character names.
    """
    
    # Loop through nodes in graph
    for node in graph.nodes:
        first_name = node
        full_name = character_dataframe.loc[character_dataframe['character_firstname'] == 
                                            first_name, 'character'].iloc[0]
        graph.nodes[node]['label'] = full_name
        
        
        
def build_adv_network(graph, df):
    """
    This function builds an advanced network graph from the inputted Networkx graph, updates node labels with character name data from the inputted Pandas DataFrame.
    
    Parameters:
        - graph: A Networkx graph object.
        - df: A Pandas DataFrame containing character name data.

    Returns:
        - result: The function builds and returns an advanced network graph with updated node labels.
    """
    
    #Set paramaters for advanced network visualiztion
    net = Network(notebook=True, width='1000px', height='700px', bgcolor='#222222', font_color='white')

    #Store the degree for each node in a dictionary 
    node_degree = dict(graph.degree)

    #Set size of nodes based on the node_degree
    nx.set_node_attributes(graph, node_degree, 'size')

    #Update node labels
    update_node_labels(graph, df)

    #Rebuild network
    net.from_nx(graph)

    #Set additional options for the network
    net.set_options('''
    var options = {
    "physics": {
    "maxVelocity": 5
    }
    }
    ''')
    return net



def build_adv_comm_net(graph, df, communities):
    """
    This function builds an advanced network graph that also dislays community partitions.
    This is achieved via the inputted Networkx graph, the Pandas DataFrame containing character name data, and the community data from Louvain's algo.
    This function returns the advanced community network graph.
    
    Parameters:
        - graph: A Networkx graph object.
        - df: A Pandas DataFrame containing character name data.
        - communities: A dictionary with the character names and there associated community.

    Returns:
        - result: The function builds and returns an advanced network graph with updated node labels and community partitions.
    """
    
    nx.set_node_attributes(graph, communities, 'group')
    
    comm_net = Network(notebook=True, width='1000px', height='700px', bgcolor='#222222', font_color='white')
    comm_net.set_options('''
    var options = {
    "physics": {
    "maxVelocity": 5
    }
    }
    ''')

    #Update node labels
    update_node_labels(graph, df)
    
    comm_net.from_nx(graph)
    
    return comm_net




def visualize_character_insights(node_df, insight, book_title):
    """
    This function visualizes character centrality insights extracted from graph nodes. 
    The input variables include a Pandas Dataframe containing graph node data, a string that identifies the insight, and the book title.
    This function does not return anything, but saves the visualization to the specified directory.
    
    Parameters:
        - node_df: A Pandas DataFrame containing node data.
        - insight: A String identfying what type of insight is being visualized.
        - book_title: The book title.

    Returns:
        - result: The function does not return anything, but visualizes character centrality insights and saves the images to the specified directory.
    """
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Get top 10 nodes by centrality
    top_10_nodes = node_df.sort_values('centrality', ascending=False).head(10)
    
    # Plot top 10 nodes using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10_nodes.index, y='centrality', data=top_10_nodes, palette='viridis')
    
    # Add labels and title
    plt.title('Top 10 Characters by ' + insight.title() + ' Centrality')
    plt.xlabel('Characters')
    plt.ylabel(insight.title() + " Centrality")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(work_dir + '/assets/img/' + book_title + '_' + insight + '.png')