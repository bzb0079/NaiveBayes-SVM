import csv
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# train Data
trainData = pd.read_csv('train.csv', nrows=10000)
# test Data
colnames = ['sentiment', 'text', 'actual_sentiment']
testData = pd.read_csv('test.csv', names=colnames)

# trainData.sample(frac=1).head(5)
testData_list = testData.text.tolist()
actualSentiment_list = testData.actual_sentiment.tolist()
testData_list = testData_list[1:] #remove the header from the file
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(trainData['text'])
test_vectors = vectorizer.transform(testData['text'])
# Perform classification with SVM, kernel=linear
accuracy_all = []
for kernel in ('linear', 'poly', 'rbf'):
    for gamma in [0.5, 1, 2]:
        classifier_linear = svm.SVC(kernel=kernel, gamma=gamma)
        # classifier_linear = svm.SVC(kernel=kernel, gamma=gamma)
        # classifier_linear = svm.SVC(kernel='rbf')
        t0 = time.time()
        classifier_linear.fit(train_vectors, trainData['sentiment'].astype(str))
        t1 = time.time()
        prediction_linear = classifier_linear.predict(test_vectors)
        t2 = time.time()
        time_linear_train = t1-t0
        time_linear_predict = t2-t1
        # results
        print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
        # print(len(testData_list))
        # print(prediction_linear)
        finalData = []
        i = 0
        count = 0
        for data in testData_list:
            # print(prediction_linear[i], testData_list[i])
            finalData.append({'sentiment': prediction_linear[i], 'text': testData_list[i]})
            i += 1
            if prediction_linear[i] == actualSentiment_list[i]:
                count +=1
        # print(finalData)
        warnings.filterwarnings('ignore')
        report = classification_report(testData['text'], prediction_linear)
        csv_columns = ['sentiment','text']
        accuracy = round((count/len(testData_list)) * 100, 2)
        accuracy_all.append(accuracy)
        print("accuracy of " + kernel + " kernel" + " with " + "gamma=", gamma, accuracy , "%")
        # try:
        #     with open('result_svm_' + kernel + '_gamma_' + str(gamma) + '.csv', 'w', encoding="utf-8") as csvfile:
        #         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        #         writer.writeheader()
        #         for data in finalData:
        #             writer.writerow(data)
        # except IOError:
        #     print("I/O error")
labels = ['linear','poly','rbf']
print(accuracy_all)
accuracy_value_at_point = accuracy_all[0::3]
accuracy_value_at_one = accuracy_all[1::3]
accuracy_value_at_two = accuracy_all[2::3]
# print(men_means)
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/3, accuracy_value_at_point, width, label='gamma=0.5')
rects2 = ax.bar(x + width/3, accuracy_value_at_one, width, label='gamma=1')
rects3 = ax.bar(x + width, accuracy_value_at_two, width, label='gamma=2')
# rects4 = ax.bar(x - width, women_means, width, label='C=3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (in %)')
ax.set_title('Accuracy of SVM kernels with gamma values')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="lower right")


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
autolabel(rects2)
autolabel(rects3)
# autolabel(rects4)

fig.tight_layout()

plt.show()