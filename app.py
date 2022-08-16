

from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
import pandas as pd
import os 

app = Flask(__name__)

df = pd.read_csv(os.path.join('train_colab', 'train.csv'))
MAX_WORDS = 250000 # number of words in the vocab

# load vectorizer
vectorizer = TextVectorization(max_tokens=MAX_WORDS,
                               output_sequence_length=1800,
                               output_mode='int')
# loading the model
model = load_model('epoch20_v4-2-6.h5')


def score_comment(Chat):
    vectorized_comment = vectorizer([Chat])
    results = model.predict(vectorized_comment)
    
    output = []
    for idx, _ in enumerate(df.columns[2:]):
        output.append(results[0][idx]>0.5)
    
    return output

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        comment = request.form.get("comment")
        output = score_comment(comment)
        # print(comment)
        # output = [1, 0, 1, 0]s
        # output[0] = 0 if comment.islower() else 1

        return render_template("proto.html", output=output)
    return render_template("proto.html")

if __name__=='__main__':
    app.run(debug=True, use_reloader=False)