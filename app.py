import pandas as pd
import pickle
import streamlit as st
import time
from sklearn.datasets import load_iris


st.title('🌸Iris Flower Classification ML Model Project🌸')

desc = '''
## Objective
Classify Iris flowers into one of three species — *Setosa*, *Versicolor*, or *Virginica* — using multiple machine learning models.

## Approach

### 1. Data Preparation
- Load the Iris dataset with features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Split dataset into training and testing sets.

### 2. Model Used and Trian
Train and evaluate multiple classifiers:
- Logistic Regression
- Decision Tree 
- Random Forest 
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- AdaBoost '''

st.markdown(desc)

iris_data = load_iris()
iris_df = pd.DataFrame(iris_data['data'],columns = iris_data['feature_names'])
iris_df['target'] = iris_data['target']
iris_df['target_names'] = iris_df['target'].apply(lambda row: iris_data['target_names'][row])

st.sidebar.title('Select Flower Features : ')
st.sidebar.image('https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png')

X_data = []
col = []

for i in iris_df.iloc[:,:-2]:
    input_value = st.sidebar.slider(f'Select {i} Vlaue :',iris_df[i].min(),iris_df[i].max())
    X_data.append(input_value)
    col.append(i)

    
final_X = [X_data]
    
user_df = pd.DataFrame(final_X,columns = col,index = [1])

st.markdown('## Flower Features : ')
st.write(user_df)

with open('chatgpt.pkl','rb') as f:
    model = pickle.load(f)

species =  iris_data['target_names']
ans = model.predict(final_X)[0]    

predicted_flower_species = species[ans]

with st.spinner("Predicting Flower Species ... Please wait."):
    time.sleep(3)
    
st.success(f'Predicted Species : {predicted_flower_species}')

st.image(f"{predicted_flower_species}.jpg")

st.markdown("---")  # horizontal line
import streamlit as st

footer = """
<div style="text-align: center; font-size:14px;">
<p>Designed and Developed by <b>Altaf Husain</b></p>
<p>Connect with me:</p>

<a href="https://www.linkedin.com/in/altaf-husain-05197b346/" target="_blank">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="30px" style="margin:5px;">
</a>
<a href="https://github.com/altaf4228?tab=repositories"_blank">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" width="30px" style="margin:5px;">
</a>
<a href="https://www.instagram.com/altafhusain24/" target="_blank">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/instagram.svg" width="30px" style="margin:5px;">
</a>
<a href="https://www.facebook.com/altaf.husain.909649/" target="_blank">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/facebook.svg" width="30px" style="margin:5px;">
</a>
<a href="https://www.youtube.com/" target="_blank">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/youtube.svg" width="30px" style="margin:5px;">
</a>

</div>
"""

st.markdown(footer, unsafe_allow_html=True)


