from flask import Flask, request, url_for, redirect, render_template, send_file
import pickle
import numpy as np
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.4) #pip install plotly==4.5.4
import plotly.express as px
import json

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def helloworld():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predator():
    int_features = [float(x) for x in request.form.values()]
    # print(request.form.values())
    # print(int_features[2])
    predicted_values = []
    perc_predicted_values = []
    for i in range(1, 6):
        int_features[2] = i
        final = [np.array(int_features)]
        prediction = model.predict(final)
        int_prediction = float(prediction)
        predicted_values.append(int_prediction)
        print(predicted_values)
    for val in predicted_values:
        percentage = val*100
        print(percentage)
        formatted_percentage = round(percentage, 2)
        print(formatted_percentage)
        perc_predicted_values.append(formatted_percentage)

    with open('uni.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['University', 'Chance of Admit'])
        thewriter.writerow([1, perc_predicted_values[0]])
        thewriter.writerow([2, perc_predicted_values[1]])
        thewriter.writerow([3, perc_predicted_values[2]])
        thewriter.writerow([4, perc_predicted_values[3]])
        thewriter.writerow([5, perc_predicted_values[4]])

    # return render_template("index.html", pred=predicted_values)
    return redirect(url_for('visualize'))


@app.route("/visualize")
def visualize():
    df = pd.read_csv("uni.csv")
    barchart = px.bar(
        data_frame=df,
        x="University",
        y="Chance of Admit",
        labels={"University" : "University Rating", "Chance of Admit" : "Chance of Admit (percent)"},
        barmode='relative',
        title='Chance of Admission according to uni ranking',  # figure title
        width=1400,  # figure width in pixels
        height=720,  # figure height in pixels
        template='gridon',
    )
    # plotly.offline.plot(barchart, filename='templates/positives.html', config={'displayModeBar': False})
    barchart.write_html("templates/positives.html")
    return render_template("positives.html")


if __name__ == '__main__':
    app.run(debug=True)
