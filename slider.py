import streamlit as st
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
from apps import praproses
import os
from dateutil import parser
import joblib
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# @st.experimental_memo
def sidebar():
  # Sidebar - Specify parameter settings
  with st.sidebar.header('2. Set Parameter'):
    split_size = st.sidebar.slider('Rasio Pembagian Data (% Untuk Data Latih)', 10, 90, 80, 5)
    number_of_features = st.sidebar.slider('jumlah pilihan feature (Untuk Data Latih)', 5, 47, 20, 5)
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 10, 100, 50, 10)
    neighbor = st.sidebar.slider('Jumlah K (KNN)', 11, 101, 55, 11)
  return split_size, number_of_features, parameter_n_estimators, neighbor

@st.experimental_memo
def load_data():
  df = pd.read_csv('data/data_praproses.csv')
  return df

@st.experimental_memo
def choose_feature(df):
  number_of_features=sidebar()
  # df=load_data()
  # from sklearn.feature_selection import mutual_info_classif
  #determine the mutual information
  mutual_info = mutual_info_classif(df.drop(columns=['enrolled']), df.enrolled)
  mutual_info = pd.Series(mutual_info)
  mutual_info.index = df.drop(columns=['enrolled']).columns
  mutual_info.sort_values(ascending=False)
  # from sklearn.feature_selection import SelectKBest
  selected_features = SelectKBest(mutual_info_classif, k = number_of_features)
  selected_features.fit(df.drop(columns=['enrolled']), df.enrolled)
  column_selection = df.drop(columns=['enrolled']).columns[selected_features.get_support()]
  pd.Series(column_selection).to_csv('data/feature_pilihan.csv',index=False)
  feature = column_selection.tolist()
  new_feature = df[feature]
  return new_feature

@st.experimental_memo
def standarization(new_feature):
  new_feature = choose_feature()
  sc_X = StandardScaler()
  column_selection = sc_X.fit_transform(new_feature)
  joblib.dump(sc_X, 'data/standar.joblib')
  return column_selection

@st.experimental_memo
def split(column_selection, split_size):
  X_train, X_test, y_train, y_test = train_test_split(column_selection, df['enrolled'],test_size=(100-split_size)/100, random_state=111)
  return X_train, X_test, y_train, y_test

@st.experimental_memo
def naive_bayes(X_train, X_test, y_train, y_test):
  nb = GaussianNB() # Define classifier)
  nb.fit(X_train, y_train)
  # Make predictions
  y_test_pred = nb.predict(X_test)
  matrik_nb = (classification_report(y_test, y_test_pred))
  cm_label_nb = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
  return matrik_nb, cm_label_nb

@st.experimental_memo
def random_forest(X_train, X_test, y_train, y_test, parameter_n_estimators):
  rf = RandomForestClassifier(n_estimators=parameter_n_estimators, max_depth=2, random_state=42) # Define classifier
  rf.fit(X_train, y_train)
  # Make predictions
  y_test_pred = rf.predict(X_test)
  matrik_rf = (classification_report(y_test, y_test_pred))
  cm_label_rf = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
  return matrik_rf, cm_label_rf

@st.experimental_memo
def stack_model(X_train, X_test, y_train, y_test, k):
  # Build stack model
  estimator_list = [
      ('nb',nb),
      ('rf',rf)]
  stack_model = StackingClassifier(
      estimators=estimator_list, final_estimator=KNeighborsClassifier(neighbor),cv=5
  )
  # Train stacked model
  stack_model.fit(X_train, y_train)
  # Make predictions
  y_test_pred = stack_model.predict(X_test)
  # Evaluate model
  matrik_stack = (classification_report(y_test, y_test_pred))
  cm_label_stack = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
  joblib.dump(stack_model, 'data/stack_model.pkl')
  return matrik_stack, cm_label_stack

def app():
  if 'data_praproses.csv' not in os.listdir('data'):
    st.markdown("Please upload data through `Home` page!")
  else:
    sidebar()
    # --- Initialising SessionState ---
    if "load_state" not in st.session_state:
      st.session_state.load_state = False
    if st.sidebar.button('Train & Test')or st.session_state.load_state:
      st.session_state.load_state = True
      st.markdown("""
      #### Train & Test
      """)
      # Load data
      df=load_data()
      # split_size, number_of_features, parameter_n_estimators, neighbor = sidebar()
      # Choose feature
      new_feature = choose_feature(df)
      # Standarization
      column_selection=standarization(new_feature)
      # Split data
      X_train, X_test, y_train, y_test = split(column_selection, split_size)
      # Naive Bayes
      matrik_nb, cm_label_nb = naive_bayes(X_train, X_test, y_train, y_test)
      # Random Forest
      matrik_rf, cm_label_rf = random_forest(X_train, X_test, y_train, y_test, parameter_n_estimators)
      # Stack Model
      matrik_stack, cm_label_stack = stack_model(X_train, X_test, y_train, y_test, k)

      nb_container = st.columns((1.1, 0.9))
      #page layout
      with nb_container[0]:
        st.write("2a. Naive Bayes report using sklearn")
        st.text('Naive Bayes Report:\n ' + matrik_nb)
      st.write(" ")
      st.write(" ")
      st.write(" ")
      with nb_container[1]:
        cm_label_nb.index.name = 'Actual'
        cm_label_nb.columns.name = 'Predicted'
        sns.heatmap(cm_label_nb, annot=True, cmap='Blues', fmt='g')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
      st.write(" ")
      st.write(" ")

      # Evaluate model
      rf_container = st.columns((1.1, 0.9))
      #page layout
      with rf_container[0]:
        st.write("2b. Random Forest report using sklearn")
        st.text('Random Forest Report:\n ' + matrik_rf)
      st.write(" ")
      st.write(" ")
      st.write(" ")
      with rf_container[1]:
        cm_label_rf.index.name = 'Actual'
        cm_label_rf.columns.name = 'Predicted'
        sns.heatmap(cm_label_rf, annot=True, cmap='Blues', fmt='g')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
      st.write(" ")
      st.write(" ")

      stack_container = st.columns((1.1, 0.9))
      #page layout
      with stack_container[0]:
        st.write("2c. Stack report using sklearn")
        st.text('Stack Report:\n ' + matrik_stack)
      st.write(" ")
      st.write(" ")
      st.write(" ")

      with stack_container[1]:
        cm_label_stack.index.name = 'Actual'
        cm_label_stack.columns.name = 'Predicted'
        sns.heatmap(cm_label_stack, annot=True, cmap='Blues', fmt='g')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
      st.write(" ")
      st.write(" ")      

      # var_enrolled = df1['enrolled']
      # #membagi menjadi train dan test untuk mencari user id
      # X_train, X_test, y_train, y_test = train_test_split(df1, df1['enrolled'], test_size=(100-split_size)/100, random_state=111)
      # train_id = X_train['user']
      # test_id = X_test['user']
      # #menggabungkan semua
      # y_pred_series = pd.Series(y_test).rename('Aktual',inplace=True)
      # hasil_akhir = pd.concat([y_pred_series, test_id], axis=1).dropna()
      # hasil_akhir['Prediksi']=y_test_pred
      # hasil_akhir = hasil_akhir[['user','Aktual','Prediksi']].reset_index(drop=True)
      # st.text('Tabel Perbandingan Asli dan Prediksi:\n ')
      # st.write(hasil_akhir)
