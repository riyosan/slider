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
def choose_feature(df, number_of_features):
  ...
  return new_feature

@st.experimental_memo
def standarization(new_feature):
  ...
  return column_selection

@st.experimental_memo
def split(df, column_selection, split_size):
  ...
  return X_train, X_test, y_train, y_test

@st.experimental_memo
def naive_bayes(X_train, X_test, y_train, y_test):
  ...
  return matrik_nb, cm_label_nb, nb


def app():
  if 'data_praproses.csv' not in os.listdir('data'):
    st.markdown("Please upload data through `Home` page!")
  else:
    split_size, number_of_features, parameter_n_estimators, neighbor = sidebar()
    if st.sidebar.button('Train & Test') or st.session_state.load_state:
      st.session_state.load_state = True
      df=load_data()
      # Choose feature
      new_feature = choose_feature(df)
      # Standarization
      column_selection=standarization(new_feature)
      # Split data
      X_train, X_test, y_train, y_test = split(column_selection, split_size)
      # Naive Bayes
      matrik_nb, cm_label_nb = naive_bayes(X_train, X_test, y_train, y_test)

      st.write("2a. Naive Bayes report using sklearn")
      st.text('Naive Bayes Report:\n ' + matrik_nb)
      sns.heatmap(cm_label_nb, annot=True, cmap='Blues', fmt='g')
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.pyplot()
