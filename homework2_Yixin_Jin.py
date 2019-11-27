import pandas as pd
import re
from pandas.io.json import json_normalize
# for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# convert the large JSON full restaurants and reviews files
# into two Pandas dataframes.
def fetch_restaurant(businesses_file):
    restaurant = None
    with open(businesses_file) as in_file:
        raw = pd.read_json(in_file)
        description = raw["businesses"]
        restaurant = json_normalize(description)
        print(restaurant.shape)
        print(restaurant.columns)
    return restaurant

def fetch_reviews(reviews_filename):
    review = None
    with open(reviews_filename) as in_file:
        raw = pd.read_json(in_file)
        description = raw["reviews"]
        review = json_normalize(description)
        review = review.rename(columns={'stars': 'stars_review'})
        print(review.shape)
        print(review.columns)
    return review

# merge two dataframes
def merge_reviews_with_resaurants(business,review):
    merged_df = business.merge(review, on="business_id")
    print(merged_df.shape)
    print(merged_df.columns)
    return merged_df

def q1_yelp(df, business):
    print('Each Star group average percentage:')
    # Get the number of restaurants of each star level
    group_count = business.groupby('stars').size() 
    # Get the number of restaurants of all star levels
    restaurant_total = len(business)
    # Get the percentage of restaurants of each star level
    for stars in business['stars'].unique():
        group_star_pert = group_count[stars] / restaurant_total
        group_star_pert = round(group_star_pert*100,2)
        print(f'   {stars}: {group_star_pert}%')
    print()
    print('Each Star group average words:')
    # a function for counting words in each review
    # input: review extracted from each row of the merged dataframe
    def q1_word_count(review):
        # The pattern of a word at the beginning and the following parts of a review
        pattern = '\s(\w+)|(\w+)\s'
        word_count = re.compile(pattern)
        matches = re.findall(word_count, review)
        return len(matches)
    # Get a new column for the  words' number of each reveiw
    df["word_count"] = df.loc[:, "text"].apply(q1_word_count)
    # Sum up the number of review words for each restaurants level
    group_word_count = df.groupby("stars")["word_count"].sum()
    # Count the number of reviews at each star level
    group_review_count = df.groupby("stars")["text"].size()
    # Compare the mean with the value in each level
    for stars in business["stars"].unique():
        group_word_average = group_word_count[stars] / group_review_count[stars]
        group_word_average = round(group_word_average,2)
        print(f"   {stars}: {group_word_average}")
        
def q2_yelp(business):
    print("All restaurant lables")
    raw_categories = business['categories'].str.split(',',expand=True)
    print(raw_categories.shape) # Get the largest number of categories for each restarant.
    # There're at most 15 sub-categories for each restaurant's category.

    categories_list = []
    row_raw_categories = raw_categories.shape[0]
    column_raw_categories = raw_categories.shape[1]
    # Retrieve the sub-categories from each restaurant
    # and get the category list 
    for restaurant_row in range(row_raw_categories):
        for category_column in range(column_raw_categories):
            category = raw_categories.iat[restaurant_row, category_column]
            if category != None:
                category = category.strip()
                if (category not in categories_list) & (category != None) :   
                    categories_list.append(category)   
    print(f"There are {len(categories_list)} labels.")

    restaurant_count_each_label = 0
    star_count_each_label = 0
    # Generate a dataframe to save the values
    df_category = pd.DataFrame(columns=['restaurant_number', 'mean_score'])
    # Use for-loop to go through each label in the categories dataframe
    for label in categories_list:
        for restaurant_row in range(row_raw_categories):
            for category_column in range(column_raw_categories):
                category = raw_categories.iat[restaurant_row, category_column]
                if category != None:
                    category = category.strip()
                    if label == category:
                        restaurant_count_each_label += 1
                        star_count_each_restaurant = business.loc[restaurant_row]["stars"] 
                        star_count_each_label += star_count_each_restaurant 
        mean_score_each_label = star_count_each_label / restaurant_count_each_label
        mean_score_each_label = round(mean_score_each_label,2)
        df_category.at[label,'restaurant_number'] = restaurant_count_each_label
        df_category.at[label,'mean_score'] = mean_score_each_label

        # Or straightly print out the values in the Terminal
        # print(f"{label}:")
        # print(f"There're {restaurant_count_each_label} restrants.")
        # print(f"The mean score is {mean_score_each_label}.")
        # print()

        restaurant_count_each_label = 0
        star_count_each_label = 0

    # Save the dataframe the into csv file
    # with open("subgroups_label.csv", "w") as out_file:
    #     out_file.write(df_category.to_csv())
    print("A csv file has been generated to record the restaurant number and mean score.")

    # For descriptive statistic: draw a bar plot for most common categories among all restaurants
    df_category_descrptive = df_category.sort_values(by = 'restaurant_number', ascending=False)
    df_category_descrptive_top10 = df_category_descrptive['restaurant_number'].head(10)
    sns.barplot(df_category_descrptive_top10.index, df_category_descrptive_top10.values, alpha=0.8)
    plt.title("Top 10 Categories of All restaurants")
    plt.ylabel('of businesses', fontsize=12)
    plt.xlabel('Categories ', fontsize=12)
    plt.show()

    # For Q3: find out the category features for the high-scoring review
    df_category_star = df_category.sort_values(by = 'mean_score', ascending=False)
    df_category_star_top10 = df_category_star['mean_score'].head(10)
    sns.barplot(df_category_star_top10.index, df_category_star_top10.values, alpha=0.8)
    plt.title("Categories of Top 10 Star Rating")
    plt.ylabel('of businesses', fontsize=12)
    plt.xlabel('Categories ', fontsize=12)
    plt.show()

def q3_yelp(df,business):
    # Descriptive statistic offers us an overview of the varaibles
    # Get the distribution of the stars rating
    df_stars = business['stars'].value_counts()
    df_stars = df_stars.sort_index()
    # Draw histogram of the stars rating
    sns.barplot(df_stars.index, df_stars.values, alpha=0.8)
    plt.title("Star Rating Distribution")
    plt.ylabel('of businesses', fontsize=12)
    plt.xlabel('Star Ratings ', fontsize=12)
    plt.show()
    # for var in df.columns:
    #     print(var)
    #     print(df[var].describe())
    #     print()
    # Find out 6 continuous variables other than "stars"
    continuous_var = [
         "longitude", "latitude", "review_count",  
          "useful", "funny", "cool"]
    # Draw boxplots of the continuous varibles
    for var in continuous_var:
        sns.boxplot(x = "stars", y = var, data = df)
        plt.title(f"{var} by star level")
        plt.show()
    
# For the original instruction, but can be a tool for descritive statistics
def q2_yelp_original(df):
    print("Each region review number:")
    # a function for portal code identification
    # input: the portal code extracted from each row of the merged dataframe
    def q2_yelp_region(code):
        Other1 = "16\d{3}"
        Other2 = "17\d{3}"
        Other3 = "18\d{3}"
        patterns = {
            "Pittsburgh":"15\d{3}",
            "Philadelphia": "19\d{3}",
            "Other":f"{Other1}|{Other2}|{Other3}"}
        region = None
        for regions in patterns:
            pattern = re.compile(patterns[regions])
            regex_region = re.search(pattern, code)
            if regex_region != None:
                region = regions
        return region
    # get a new column for the  region of each reveiw
    df["region"] = df.loc[:, "postal_code"].apply(q2_yelp_region)
    # count the number of reviews for each region
    group_count = df.groupby("region").size()
    print(group_count)
  
# To import this function you will need to install the lxml library using Conda.
from wiki_api import page_text

def get_featured_biographies():
    page_list = page_text('Wikipedia:Featured articles', 'list')
    # print(page_list)
    # After checking the strucuture of outcome,
    # find out that the row end with'[edit]' is a sub-title, which should be the start
    # Therefore, set the first occurence of 'edit'as the the start
    # to skip over the beginning descption of the webpage
    for page_list_row in range(len(page_list)):
        txt = page_list[page_list_row].strip()
        if '[edit]' in txt:
            break
    # Here, page_list_row is the first row of topic
    # txt is the title of the first topic 
    featuredArticle = {}  # Generate a dictionary to save the topic and the titles of the articles 
    Articles_each_topic = None
    subTopic = txt 
    while page_list_row < len(page_list):
        txt = page_list[page_list_row].strip()
        if '[edit]' in txt:
            featuredArticle[subTopic] = Articles_each_topic
            Articles_each_topic = []
            subTopic = txt
            page_list_row += 1
            continue 
        if txt != None:
            Articles_each_topic.append(txt)
        page_list_row += 1
      
    # In the previous check, also find out that
    # featured articles also biographies have 
    # the sub-topic containing "biographies" or "Biographies"
    biographies = []    # Generate a list to save these articles
    for subTopic in featuredArticle.keys():
        if 'biographies' in subTopic.lower() and 'Autobiographies' not in subTopic:
            biographies.extend(featuredArticle[subTopic])
    # clean the featured biologies list
    del biographies[301]
    del biographies[516]
  
    biographyCount = len(biographies)
    articleCount = 0    # Count the number of featured articles    
    for subTopic in featuredArticle:
        articleCount += len(featuredArticle[subTopic])
    print(f"There're {articleCount} featured articles, among which, {biographyCount} articles that are also biographies.")
    article_perct = biographyCount / articleCount
    article_perct = round(article_perct*100,2)
    print(f"The percentage of featured articles among the featured articles is {article_perct}%.")
    return biographies 

def get_first_paragraph(page):
    info = page_text(page, 'text')  # Retrieve information of a page 
    # Split the retrieved information into different paragraphs through Line break in html with `\n`
    info = re.split('\n+', info) 
    # After going through several webpage, find out the common characteristics of the first paragraph:
    # 1. begins at the second items
    # 2. covers more than 100 words
    for txt in info[1:]:
        if len(txt) > 100:
            return txt
    # If there're no paragraph containing more than 100 words,
    # use the second item directly
    return info[1]  

def get_pronouns(text):
    # Construct the pattern for different pronouns
    maleWord = '\W(he|his|him)\W'
    femaleWord = '\W(she|her|hers)\W'
    otherWord = '\W(they|them|their)\W'
    malePattern = re.compile(maleWord)
    femalePattern = re.compile(femaleWord)
    otherPattern = re.compile(otherWord)

    # cleaning data to remove the influences of Capital letters
    text = text.lower()  
    maleMentions = malePattern.findall(text)
    femaleMentions = femalePattern.findall(text)
    otherMentions = otherPattern.findall(text)

    # print("Gender Distribution:")
    # print(f"  man: {len(maleMentions)}")
    # print(f"woman: {len(femaleMentions)}")
    # print(f"other: {len(otherMentions)}")

    # Identify the most possible gender by the occurence times of pronounce
    if len(maleMentions) > len(femaleMentions):
        return 'Male'
    elif len(maleMentions) < len(femaleMentions):
        return 'Female'
    else:
        return 'Unknown'

def additional_analysis(text):
    # Construct the pattern for success and failure
    not_prefix = "[^(not)]\s"
    successWord = "(success|archive|succeed)"
    failureWord = "fail"
    success = f"{not_prefix}{successWord}"
    failure = f"{not_prefix}{failureWord}"
    successPattern = re.compile(success)
    failurePattern = re.compile(failure)

    # cleaning data to remove the influences of Capital letters
    text = text.lower()
    successMentions = successPattern.findall(text)
    failureMentions = failurePattern.findall(text)

    # Identify the success and failure cases by the occurence times of specific word
    if len(successMentions) > len(failureMentions):
        return 'Success'
    elif len(failureMentions) > len(successMentions):
        return 'Failure'
    else:
        return 'Unknown'

def export_dataset(df):
    titles = get_featured_biographies() # Get the titles of featured biographies
    infos = []  # Generate a list to save the page
    print()
    print("The sample featured biography are listed as followed:")
    for page in titles[:20]:    # Get the new sample by change the index of the title list
        print(page)
        firstParagraph = get_first_paragraph(page)  # Retrieve the first paragraph
        text = page_text(page, 'text')  # Retrieve the topic information of a page
        pronouns = get_pronouns(text)  # Get the most possible gender
        success = additional_analysis(text)  # Get the success or failure cases

        info = {
            'title': page,
            'most_pronouns': pronouns,
            'introduction': firstParagraph,
            'stableness_cases': success}

        infos.append(info)

    df_info = pd.DataFrame(infos)
    df_info.to_csv(df + '.csv')    # save the information into the format of csv
    df_info.to_json(df + '.json')    # save the information into the format of JSON

def get_birth_and_death(infobox):
    pass
    # Still Working on it
    born = r'Born</th><td.*?>(.*?)</td>'
    died = r'Died</th><td.*?>(.*?)</td>'
    bornPattern = re.compile(born)
    diedPattern = re.compile(died)
    bornMentions = bornPattern.search(infobox)
    diedMentions = diedPattern.search(infobox)

if __name__ == "__main__":
    businesses_filename = "PA_businesses.json"
    businesses = fetch_restaurant(businesses_filename)
    print()
    reviews_filename = "PA_reviews_full.json"
    reviews = fetch_reviews(reviews_filename)
    print()
    merged_df = merge_reviews_with_resaurants(businesses,reviews)
    print()
    q1_yelp(merged_df,businesses)
    print()
    # q2_yelp_original(businesses)
    q2_yelp(businesses)
    print()
    q3_yelp(merged_df,businesses)

    dataset_filename ="homework2_Yixin_Jin_Sample"
    export_dataset(dataset_filename)
