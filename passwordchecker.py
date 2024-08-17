import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings 
from warnings import filterwarnings
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from collections import Counter


filterwarnings("ignore")
data = pd.read_csv('C:/Users/K Sarkar/OneDrive/Documents/GitHub/PSC/data.csv', on_bad_lines='skip')
rows_with_null = data[data.isnull().any(axis=1)]
data.dropna(inplace= True)

def find_semantics(row):
    for char in row:
        if char in string.punctuation:
            return 1
        else:
            pass

data[data["password"].apply(find_semantics)==1]   
data.head()
# Length of password

data.password.str.len()
data["length"] = data["password"].str.len()
def freq_lowercase(row):
    return len([char for char in row if char.islower()])/len(row)
# Frequency of uppercase Characters
def freq_uppercase(row):
    return len([char for char in row if char.isupper()])/len(row)
# Frequency of Numeric Characters 
def freq_numerical_case(row):
    return len([char for char in row if char.isdigit()])/len(row)
data["lowercase_freq"] = np.round(data["password"].apply(freq_lowercase) , 3)

data["uppercase_freq"] = np.round(data["password"].apply(freq_uppercase) , 3)

data["digit_freq"] = np.round(data["password"].apply(freq_numerical_case) , 3)
# Frequency of Special-case Characters 
def freq_special_case(row):
    special_chars = [] 
    for char in row:
        if not char.isalpha() and not char.isdigit():
            special_chars.append(char) 
    return len(special_chars) 
data["special_char_freq"] = np.round(data["password"].apply(freq_special_case) , 3)
data["special_char_freq"] = data["special_char_freq"]/data["length"] 
cols = ['length', 'lowercase_freq', 'uppercase_freq',
       'digit_freq', 'special_char_freq']

# data[['length','strength', 'lowercase_freq', 'uppercase_freq','digit_freq', 'special_char_freq']].groupby("strength").agg(["mean"])
# fig , ((ax1 , ax2) , (ax3 , ax4) , (ax5,ax6)) = plt.subplots(3 , 2 , figsize=(15,7))
# sns.boxplot(x="strength" , y='length' , hue="strength" , ax=ax1 , data=data)
# sns.boxplot(x="strength" , y='lowercase_freq' , hue="strength" , ax=ax2, data=data)
# sns.boxplot(x="strength" , y='uppercase_freq' , hue="strength" , ax=ax3, data=data)
# sns.boxplot(x="strength" , y='digit_freq' , hue="strength" , ax=ax4, data=data)
# sns.boxplot(x="strength" , y='special_char_freq' , hue="strength" , ax=ax5, data=data)
# plt.subplots_adjust(hspace=0.6)
# def get_dist(data , feature):
    
#     plt.figure(figsize=(10,8))
#     plt.subplot(1,2,1)
#     # 1 row
#     # 2 column
    
#     # violinplot
#     sns.violinplot(x='strength' , y=feature , data=data )
    
#     plt.subplot(1,2,2)
    
#     sns.distplot(data[data['strength']==0][feature] , color="red" , label="0" , hist=False)
#     sns.distplot(data[data['strength']==1][feature], color="blue", label="1", hist=False)
#     sns.distplot(data[data['strength']==2][feature], color="orange", label="2", hist=False)
    
#     plt.legend()
#     plt.show()
# get_dist(data , "length")
# get_dist(data , 'lowercase_freq')
# get_dist(data , 'uppercase_freq')
# get_dist(data , 'digit_freq')
# get_dist(data , 'special_char_freq')
dataframe = data.sample(frac=1)

vectorizer = TfidfVectorizer(analyzer="char")
x = list(dataframe["password"])
X = vectorizer.fit_transform(x)
len(vectorizer.get_feature_names_out())

df2 = pd.DataFrame(X.toarray() , columns=vectorizer.get_feature_names_out())

df2["length"] = dataframe['length']
df2["lowercase_freq"] = dataframe['lowercase_freq']
y = dataframe["strength"]

X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.20)

clf = LogisticRegression(multi_class="multinomial")
imputer = SimpleImputer(strategy='mean')  # or median, most_frequent
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
Counter(y_pred) 
classification_report(y_test, y_pred)

def predict():
    password = input("Enter a password : ")
    sample_array = np.array([password])
    
    # 151 dimension
    sample_matrix = vectorizer.transform(sample_array) 
    
    # +2 dimension
    length_pass = len(password)
    length_normalised_lowercase = len([char for char in password if char.islower()])/len(password)
    
    # 151 + 2 
    new_matrix2 = np.append(sample_matrix.toarray() , (length_pass , length_normalised_lowercase)).reshape(1,155)
    
    result = clf.predict(new_matrix2)
    
    if result == 0 :
        return "Password is weak"
    elif result == 1 :
        return "Password is normal"
    else:
        return "password is strong"

r= predict()
print(r)     