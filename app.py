from flask import Flask, render_template, request
from sklearn.tree import _tree
import pandas as pd 
import os 
import cloudpickle
import numpy as np 

dass_42 = pd.read_csv("dass_42.csv",index_col="qid")

Qit = lambda l: [f"Q{x}" for x in l]
with open("models/Dl_dt.pkl","rb") as f:
    Dl_dt = cloudpickle.load(f)

with open("models/Al_dt.pkl","rb") as f:
    Al_dt = cloudpickle.load(f)

with open("models/Sl_dt.pkl","rb") as f:
    Sl_dt = cloudpickle.load(f)


Depression= Qit([3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42])
Anxiety= Qit([2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41])
Stress= Qit([1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39])

level_mapping = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Extremely Sever"
}

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

ROUNDS = [(Dl_dt,Depression),(Al_dt,Anxiety),(Sl_dt,Stress)]
round_ = None
tree, feature_names = None,None
tree_ = None
node = None
qid = None

questions =None
depth =None 

RESULTS = []
MEMORY = {}



@app.route('/test',methods=['POST','GET'])
def start_test():
    global qid, node,questions, depth 
    global tree, feature_names, tree_, round_
    if request.method == "GET":
        RESULTS.clear()
        MEMORY.clear()

        round_ = 0
        tree, feature_names = ROUNDS[round_]
        tree_ = tree.tree_
        node = 0
        qid = feature_names[tree_.feature[node]]

        questions =0
        depth =0 
        
    
    elif request.method == 'POST':
        MEMORY[qid] = int(request.form.get("option"))
        questions +=1
        while (tree_.feature[node] != _tree.TREE_UNDEFINED) and  (qid in MEMORY):
            depth+=1
            if MEMORY[qid] <= tree_.threshold[node]:
                node = tree_.children_left[node]
            else:
                node = tree_.children_right[node]
            qid = feature_names[tree_.feature[node]]

        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            RESULTS.append(
                    (
                        level_mapping[np.argmax(tree_.value[node])],
                        questions,
                        depth
                    )
                )

            round_+=1
            if round_>=3: 
                return render_template('result.html', result=RESULTS)
                
            
            tree, feature_names = ROUNDS[round_]
            tree_ = tree.tree_
            node = 0
            qid = feature_names[tree_.feature[node]]        

            questions =0
            depth =0 
            
    return render_template('test.html', question=dass_42.loc[qid])


if __name__ == '__main__':
    app.run(debug=True)
