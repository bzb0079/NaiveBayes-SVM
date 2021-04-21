import pandas as pd
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('train.csv', nrows=10000)
colnames = ['sentiment', 'text', 'actual_sentiment']
testData = pd.read_csv('test.csv', names=colnames)

# trainData.sample(frac=1).head(5)
testData_list = testData.text.tolist()
actualSentiment_list = testData.actual_sentiment.tolist()
testData_list = testData_list[1:] #remove the header from the file

# It is the dictionary that has the data : { label(positive/negative) : { word : count of number of occurences of the word } }
dataset = {}
# It is the dictionary that keeps the count of records that are labeled a label l for each label l
# That is, { label l : No. of records that are labeled l }
no_of_items = {}

# This is the dictionary that contains the count of the occurences of word under each label
# That is, { word : { label l : count of the occurence of word with label l } }
feature_set = {}
df = df[["sentiment", "text"]]
# For each sentence in dataset
for ind in df.index:
    no_of_items.setdefault(df['sentiment'][ind], 0)
    # Increase the count of occurence of label by 1 for every occurence
    no_of_items[df['sentiment'][ind]] += 1
    # print(no_of_items)
    # Initialize the dictionary for a label if not present
    dataset.setdefault(df['sentiment'][ind], {})
    # Split the sentence with respect to non-characters, and donot split if apostophe is present
    split_data = re.split('[^a-zA-Z\']', df['text'][ind])
    # For every word in split data
    for i in split_data:
        # Removing stop words to a small extent by ignoring words with length less than 3
        if len(i) > 2:
            # Initialize the word count in dataset
            dataset[df['sentiment'][ind]].setdefault(i.lower(), 0)
            # Increase the word count on its occurence with label row[1]
            dataset[df['sentiment'][ind]][i.lower()] += 1
            # Initialze a dictionary for a newly found word in feature set
            feature_set.setdefault(i.lower(), {})
            # If the label was found for the word, for the first time, initialize corresponding count value for word as key
            feature_set[i.lower()].setdefault(df['sentiment'][ind], 0)
            # Increment the count for the word in that label
            feature_set[i.lower()][df['sentiment'][ind]] += 1

# To calculate the basic probability of a word for a category
def calc_prob(word, category):
    # print(word)
    # print(category)
    if word not in feature_set or word not in dataset[category]:
        return 0
    # print(dataset)
    # print(dataset[category][word])
    # print(dataset)
    return float(dataset[category][word]) / no_of_items[category]
    # return (float(dataset[category][word]) + 1) / (no_of_items[category] + len(dataset.keys()))

# Weighted probability of a word for a category
def weighted_prob(word, category):
    # basic probability of a word - calculated by calc_prob
    basic_prob = calc_prob(word, category)

    # total_no_of_appearances - in all the categories
    if word in feature_set:
        tot = sum(feature_set[word].values())
    else:
        tot = 0

    # Weighted probability is given by the formula
    # (weight*assumedprobability + total_no_of_appearances*basic_probability)/(total_no_of_appearances+weight)
    # weight by default is taken as 1.0
    # assumed probability is 0.5 here
    weight_prob = ((1.0 * 0.5) + (tot * basic_prob)) / (1.0 + tot)
    return weight_prob


# To get probability of the test data for the given category
def test_prob(test, category):
    # Split the test data
    split_data = re.split('[^a-zA-Z][\'][ ]', test)

    data = []
    for i in split_data:
        if ' ' in i:
            i = i.split(' ')
            for j in i:
                if j not in data:
                    data.append(j.lower())
        elif len(i) > 2 and i not in data:
            data.append(i.lower())

    p = 1
    for i in data:
        p *= weighted_prob(i, category)
    return p


# Naive Bayes implementation
def naive_bayes(test):
    '''
        p(A|B) = p(B|A) * p(A) / p(B)
        Assume A - Category
               B - Test data
               p(A|B) - Category given the Test data
        Here ignoring p(B) in the denominator (Since it remains same for every category)
    '''
    results = {}
    for i in dataset.keys():
        # Category Probability
        # Number of items in category/total number of items
        cat_prob = float(no_of_items[i]) / sum(no_of_items.values())

        # p(test data | category)
        test_prob1 = test_prob(test, i)

        results[i] = test_prob1 * cat_prob

    return results

# result = naive_bayes("thank you we got on a different flight")
# print(result)
dfTest = pd.read_csv('test.csv')
dfTest = dfTest[["text"]]
finalData = []
count = 0
for ind in dfTest.index:
    text = dfTest['text'][ind]
    result = naive_bayes(text)
    if result[1] > result[-1] and result[1] > result[0]:
        finalData.append({'sentiment': 'positive', 'text': text})
        sentiment = '1'
        if sentiment == actualSentiment_list[ind]:
            count += 1
        # writer.writerow({'sentiment': 'positive', 'text': text + '\r\n'})
        # print('positive')
    elif result[0] > result[1] and result[0] > result[-1]:
        finalData.append({'sentiment': 'neutral', 'text': text})
        sentiment = '0'
        if sentiment == actualSentiment_list[ind]:
            count += 1
        # writer.writerow({'sentiment': 'neutral', 'text': text + '\r\n'})
        # print('neutral')
    elif result[-1] > result[1] and result[-1] > result[0]:
        finalData.append({'sentiment': 'negative', 'text': text})
        sentiment = '-1'
        if sentiment == actualSentiment_list[ind]:
            count += 1
        # writer.writerow({'sentiment': 'negative', 'text': text + '\r\n'})
        # print('negative')
# writer.writerow(finalData,)
accuracy = round((count/len(testData_list)) * 100, 2)
print("accuracy", accuracy, "%")
csv_columns = ['sentiment','text']
try:
    with open('result_naive_bayes.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in finalData:
            writer.writerow(data)
except IOError:
    print("I/O error")
labels = ['Naive Bayes']
# print(men_means)
x = np.arange(len(labels))  # the label locations
# width = 1  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x, accuracy)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (in %)')
ax.set_title('Accuracy of Naive Bayes')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# fig.tight_layout()
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)

plt.show()



