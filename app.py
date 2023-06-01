from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import json

app = Flask(__name__)



dataset = pd.read_csv("C_LAPTOPS.csv")
dataset_c = pd.read_csv("C_LAPTOPS.csv")
dataset_c.drop(columns=['Product', 'Image'], inplace=True)


dataset['Brand'] = dataset['Brand'].str.lower()
dataset["Processor"] = dataset['Processor'].str.replace(" ", "")
dataset["Processor"] = dataset['Processor'].str.lower()
dataset["RAM"] = dataset['RAM'].str.lower()
dataset["Hard disk"] = dataset['Hard disk'].str.lower()
dataset["SSD"] = dataset['SSD'].str.lower()
dataset['Graphic card'] = dataset['Graphic card'].str.lower()
dataset['Graphic card'] = dataset['Graphic card'].str.replace(" ", '')
dataset['Operating system'] = dataset['Operating system'].str.replace(' ', '')
dataset['Operating system'] = dataset['Operating system'].str.lower()
dataset["Resolution"] = dataset['Resolution'].str.replace(" ", "")
dataset["Resolution"] = dataset['Resolution'].str.lower()
dataset['Dimensions(mm)'] = dataset['Dimensions(mm)'].astype(str)
dataset['Weight(kg)'] = dataset['Weight(kg)'].astype(str)
dataset['Battery life'] = dataset['Battery life'].astype(str)
dataset["Price"] = dataset['Price'].str.replace(" ", "")
dataset['Price'] = dataset['Price'].astype(str)


dataset_product = pd.DataFrame(dataset['Product'])


product = []

for i in dataset['Product']:
    temp = i.split('(')[0]
    product.append(temp[:-1])

dataset_laptop = pd.DataFrame(product)

columns = ['Product']
dataset_laptop.columns = columns
dataset_laptop['Image'] = dataset['Image']
dataset.drop(columns=['Product'], inplace=True)
dataset.drop(columns=['Image'], inplace=True)
lst = []
for k in range(dataset.shape[0]):
    n = [dataset[i][k] for i in dataset.columns]
    n = ' '.join(n)
    lst.append(n)
dataset_spec = pd.DataFrame(lst)
clmn = ['Specification']
dataset_spec.columns = clmn

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset_spec['Specification'])

similarity = cosine_similarity(X, X)

laptop = dataset_product['Product']

indices = pd.Series(dataset_product.index, laptop)


def recommender(title):
    index = indices[title]
    similarity_score = list(enumerate(similarity[index]))
    similarity_score = sorted(
        similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:17]
    lap_indices = [i[0] for i in similarity_score]
    return dataset_product.iloc[lap_indices].values, lap_indices


def get_suggestions():
    return list(dataset_product['Product'])


def lap_im_name(laptops):
    lst = recommender(laptops)
    details = []
    details1 = []
    for values in lst[1]:
        details1.append((dataset_c['Brand'][values], dataset_c['Processor'][values], dataset_c['RAM'][values], dataset_c['Hard disk'][values], dataset_c['SSD'][values], dataset_c['Graphic card'][values],
                        dataset_c['Operating system'][values], dataset_c['Resolution'][values], dataset_c['Dimensions(mm)'][values], dataset_c['Weight(kg)'][values], dataset_c['Battery life'][values], dataset_c['Price'][values], values))
    for i, pro in enumerate(lst[0]):
        index = indices[pro][0]
        details.append(
            (dataset_laptop['Image'][index], dataset_laptop['Product'][index], details1[i]))
    return details


@app.route("/")
def home():
    return render_template('home.html', suggestions=get_suggestions())


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    laptops = request.form['laptop']
    index = indices[laptops]
    laptop = dataset_laptop['Product'][index]
    img = dataset_laptop["Image"][index]
    brand = dataset_c['Brand'][index]
    processor = dataset_c['Processor'][index]
    ram = dataset_c['RAM'][index]
    hard_disk = dataset_c["Hard disk"][index]
    ssd = dataset_c["SSD"][index]
    graphic = dataset_c['Graphic card'][index]
    os = dataset_c['Operating system'][index]
    resolution = dataset_c['Resolution'][index]
    dimensions = dataset_c['Dimensions(mm)'][index]
    weight = dataset_c['Weight(kg)'][index]
    battery = dataset_c['Battery life'][index]
    price = dataset_c['Price'][index]
    final_pair = lap_im_name(laptops)

    return render_template('index.html', pair=final_pair, similar=lst, img=img, laptop=laptop, brand=brand, processor=processor, ram=ram, hard_disk=hard_disk, ssd=ssd, graphic=graphic, os=os, resolution=resolution, dimensions=dimensions, weight=weight, battery=battery, price=price)


if __name__ == "__main__":
    app.run(debug=True)
