import streamlit as st
import pickle

log_model=pickle.load(open('log_model.pkl','rb'))
dec_tree_model=pickle.load(open('dec_tree_model.pkl','rb'))
def classify(num):
    if num<0.5:
        st.text('Iris-setosa')
        setosa=st.image('setosa.jpg')
        return setosa
    elif num <1.5:
        st.text('Iris-versicolor')
        versicolor=st.image('versicolor.jfif')
        return versicolor
    else:
        st.text('Iris-virginica')
        virginica=st.image('virginica.jpg')
        return virginica
def main():
    st.title("Iris Flower Classification")
    activities=['Logistic Regression','Decision Tree Classifier']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    sl=st.sidebar.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.sidebar.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.sidebar.slider('Select Petal Length', 0.0, 10.0)
    pw=st.sidebar.slider('Select Petal Width', 0.0, 10.0)
    inputs=[[sl,sw,pl,pw]]
    if st.button('Classify'):
        if option=='Logistic Regression':
            st.success(classify(log_model.predict(inputs)))

        if option=='Decision Tree Classifier':
            st.success(classify(dec_tree_model.predict(inputs)))

if __name__=='__main__':
    main()



from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.sidebar.text('Upload data to view steps.')
uploaded_file = st.sidebar.file_uploader('Upload your file here')
if uploaded_file:
    #display the data
    df = pd.read_csv(uploaded_file)
    st.markdown("""## Display of data""")
    st.write(df.describe())
    
    st.header('Data header')
    st.write(df.head())

    st.markdown("""### Sum of null values""")
    st.table(df.isnull().sum())
    st.markdown("""### Sum of duplicate data""")
    st.write(df.duplicated().sum())
    st.markdown("""### Deleting duplicate data duplicate data""")
    st.write(df.drop_duplicates(inplace=True))
    st.markdown("""### Sum of duplicate data""")
    st.write(df.duplicated().sum())

    #label encoding 
    from sklearn import preprocessing

    label_encoder = preprocessing.LabelEncoder()
  
    df[' species']= label_encoder.fit_transform(df[' species'])
    df[' species'].unique()

    st.markdown("""### Visualization of correlated data""")
    st.write(df.corr())

    #plotting in heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(),annot=True, cmap="Blues",ax=ax)
    st.write(fig)

    #after dropping species
    st.markdown("""### After dropping species""")
    X = df.drop(' species',axis=1)
    y=df[' species']
    st.write(X)

    #Traning the model
    st.markdown("""### Traning the model""")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test= train_test_split(
    X, y, train_size = 0.20, random_state=33
)
    y_train.shape, X_train.shape, X_test.shape, y_test.shape

    #logical regression model
    st.markdown("""### Logical regression model""")
    classify_model = LogisticRegression(max_iter=1000) 
    classify_model.fit(X_train, y_train)

    #prediction
    st.markdown("""### prediction""")
    preds = classify_model.predict(X_test)
    st.write(preds)

    #comparing predictions
    st.markdown("""### Comparing prediction""")
    comp = pd.DataFrame({'y_test':y_test, 'y_preds': preds})
    st.write(comp)

    #confusion matrix
    st.markdown("""### Confusion matrix""")
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, preds)
    st.write(conf_matrix)

    #seaborn for graphical display
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix,annot=True,ax=ax)
    st.write(fig)

    #accuarcy test
    st.markdown("""### Accuarcy test""")
    from sklearn.metrics import accuracy_score, classification_report
    st.write(accuracy_score(y_test, preds))
    st.write(classification_report(y_test, preds))

    #pickle
    import pickle
    filename= 'trained_model1.sav'
    pickle.dump( uploaded_file,open(filename,'wb'))
    #loading the saved model E:\college&school work\BIMReeyasha\BIM 5th sem\Python\trained_model.sav
    loaded_model = pickle.load(open('trained_model1.sav', 'rb'))

#     input_data=(3,5,5,0.2)
# #changing the input data to numpy array
#     input_data_as_numpy_array=np.asarray(input_data)
# #reshaping the array
#     input_data_reshape= input_data_as_numpy_array.reshape(1,-1)
#     preds = loaded_model.predict(input_data_reshape)
#     preds

#     if (preds[0]==0):
#         print('a')
#     elif (preds[0]==1):
#         print('b')
#     else:
#         print('c')
