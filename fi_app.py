# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 08:30:58 2021

@author: bmussa
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import plotly.express as px
import numpy as np
from itertools import combinations
import statsmodels.formula.api as smf
import base64
   
st.set_page_config(layout='wide')

st.title("Feature Importance and Prediction App")
st.write("**By: [Bilal Mussa](https://www.linkedin.com/in/bilalmussa/)**")
st.write("In this app I am using tree based classifcation methods and Logistic Regression "
         "to firstly determine and plot the important features and then build a model that predicts the outcome."
         "The app comes preloaded with a heart failure dataset.")

st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file]
(https://raw.githubusercontent.com/asad-mahmood/66DaysOfData/main/Heart%20Failure/heart_failure_clinical_records_dataset.csv)
""")

st.sidebar.write('This app will drop date variables. Depenant variables need to be binary')

def get_table_download_link_csv(df, message):
    csv = df.to_csv(index=False).encode('utf-8-sig')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="'+ df.name +'.csv" target="_blank">' + message +'</a>'
    return href

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def tidy_data(file_path):
    df = pd.read_csv(file_path)
    df = df.select_dtypes(exclude=['datetime64'])
    return df


#uploaded_df= tidy_data(uploaded_file)
if not uploaded_file:
    uploaded_df = tidy_data('https://raw.githubusercontent.com/asad-mahmood/66DaysOfData/main/Heart%20Failure/heart_failure_clinical_records_dataset.csv')
else:
    uploaded_df= tidy_data(uploaded_file)
    
columns = uploaded_df.columns.to_list()


option = st.sidebar.selectbox('Select your dependant variable?',
                      (columns))

st.sidebar.write('You selected:', option)

option2 = st.sidebar.selectbox('How would you like to treat string variables?',
                  ('one hot encode','factorise'))

st.sidebar.write('You selected:', option2)

option3 = st.sidebar.selectbox('Method for feature importance?',
                  ('Random Forest','Extra Trees'))

st.sidebar.write('You selected:', option3)

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def dummy_vars(df):
    categorical_cols = df.columns[df.dtypes=='object']
    if option2 =='one hot encode':
        df = pd.get_dummies(df, columns=categorical_cols)
    elif option2 == 'factorise':
        df[categorical_cols] = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])
    return df


uploaded_df= dummy_vars(uploaded_df)

def impPlot(imp, name):
    figure = px.bar(imp,
                    x=imp.values,
                    y=imp.keys(), labels = {'x':'Importance Value', 'index':'Columns'},
                    text=np.round(imp.values, 2),
                    title=name + ' Feature Importance Plot',
                    width=1000, height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure)

def randomForest(x, y):
    model = RandomForestClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=True)
    st.subheader('Random Forest Classifier:')
    impPlot(feat_importances, 'Random Forest Classifier')
    #st.write(feat_importances)
    st.write('\n')
    return feat_importances

def extraTress(x, y):
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=True)
    st.subheader('Extra Trees Classifier:')
    impPlot(feat_importances, 'Extra Trees Classifier')
    st.write('\n')
    return feat_importances
    
@st.cache(suppress_st_warning=True)
def create_model(input_df):
    combi = []
    modelNumber = 1
    output = pd.DataFrame()
    placeholder_text = st.empty()   
    for i in range(1,len(feat_importances)):
        combi = (list(combinations(feat_importances.index, i)))
        for c in combi:
            print('-------------> Model Number:', modelNumber)
            variable_string = str(option + ' ~ 1 ')
            var_iter =1
            for j in list(c):
                #print(len(list(c)))
                #print(' + ' ,j)
                variable_string  +=' + ' + str(j)
                #final_string = ""
                if var_iter == len(list(c)):
                    #print(variable_string)
                    #print(var_iterator )
                    placeholder = 'Model Number: ' + str(modelNumber) + ' Model Variables: ' + str(variable_string)
                    placeholder_text.text(placeholder)
                    try:
                        result = smf.logit(formula= variable_string, data=input_df).fit()
                        coeffs = result.params
                        coeffs = pd.DataFrame({'Variable':coeffs.index, 'Values':coeffs.values})
                        predTable = result.pred_table()
                        #result.summary()
                        prsq = result.prsquared
    
                        tp = predTable[1,1]
                        tn = predTable[0,0]
                        fp = predTable[0,1]
                        fn = predTable[1,0]
                        
                        
                        precision = tp/(tp+fp)
                        recall = tp/(tp+fn)
                        accuracy = (tp + tn)/(tp+tn+fp+fn)
    
                        #new row as dictionary
                        row1 = [{'Variable':'modelNumber', 'Values':modelNumber}
                                , {'Variable':'pRSQ', 'Values':prsq}
                                , {'Variable':'precision', 'Values':precision}
                                , {'Variable':'recall', 'Values':recall}
                                , {'Variable':'accuracy', 'Values':accuracy}
                                , {'Variable':'truepos', 'Values':tp}
                                , {'Variable':'trueneg', 'Values':tn}
                                , {'Variable':'falsepos', 'Values':fp}
                                , {'Variable':'falseneg', 'Values':fn}
                                , {'Variable':'variableString', 'Values':variable_string}
                                , {'Variable':'NumVars', 'Values':len(c)}                           
                                ]
                        coeffs = coeffs.append(row1, ignore_index=True)
                        #append row to dataframe
                        output= output.append(coeffs.set_index('Variable').T)
                        modelNumber += 1
                    except :
                       pass
                var_iter +=1
    placeholder_text.text('Finished')
    placeholder_text.text('')   
    output.name = 'ModelOutput'
    return output

if st.sidebar.button('Run app'):
    x = uploaded_df.loc[:, uploaded_df.columns != option]  # Using all column except for the last column as X
    y = uploaded_df[option]  # Selecting the last column as Y
    
    if option3 == 'Random Forest':
        feat_importances = randomForest(x, y)
    elif option3 == 'Extra Trees':
        feat_importances = extraTress(x, y)
    
    feat_importances= feat_importances.sort_values(ascending=False)
    
    st.subheader('Best Logistic Regression Model using top 10 features only:')
    
    if len(feat_importances)>10:
        feat_importances= feat_importances[:10]

    output_df = create_model(uploaded_df)
    best_model = output_df.sort_values(by='accuracy',ascending=False)['modelNumber'][0]
    variable_string =output_df[output_df['modelNumber']==best_model]['variableString'][0]
    result = smf.logit(formula=variable_string, data=uploaded_df).fit()
    st.write(result.summary())
    st.write('')
    st.markdown(get_table_download_link_csv(output_df,"Download Model Output"), unsafe_allow_html=True)
