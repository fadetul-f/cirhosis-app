import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.metrics import *
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from pandas import DataFrame


# st.set_page_config(
#     page_title="Multipage App",
# )


selected = option_menu(
    menu_title = None,
    options=["Tentang Dataset", "Prediksi", "Model"],
    icons=["book","cast", "envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container":{"padding":"0!important", "background-color":"#ffffff",},
        "icons":{"font-size":"25px"},
        "nav-link":{"font-size":"18px",
            "text-align":"center",
            "margin":"0px",
            "--hover-color":"#eee",
        },
    }
)


def load_data():
    pd_crs = pd.read_csv("cirrhosis.csv")
    return pd_crs

# Hanya akan di run sekali
pd_crs = load_data()


if selected == "Tentang Dataset":

    st.write('''# Tentang Dataset''')
    st.write(pd_crs)
    st.write("""
    Data yang akan dianalisis adalah data tentang penyakit sirosis, 
    penyakit sirosis itu sendiri adalah penyakit di liver yang menyerang sel-sel sehat, 
    kemudian seiring waktu berubah menjadi jaringan parut.
    """)

    st.write('''#### Fitur-fitur pada dataset''')
    st.write("Pada dataset ini terdiri sebanyak 418 data dengan 20 fitur. Adapun fitur-fiturnya yaitu:")

    st.info('''
    1. ID: pengidentifikasi unik
    2. N_Days: jumlah hari antara pendaftaran dan kematian yang lebih awal, transplantasi, atau waktu analisis studi pada Juli 1986
    3. Status: status pasien C (disensor), CL (disensor karena tx hati), atau D (meninggal)
    4. Obat : jenis obat D-penicillamine atau placebo
    5. Umur: umur dalam [hari]
    6. Jenis Kelamin: M (laki-laki) atau F (perempuan)
    7. Asites: adanya asites N (Tidak) atau Y (Ya)
    8. Hepatomegali: adanya hepatomegali N (Tidak) atau Y (Ya)
    9. Laba-laba: keberadaan laba-laba N (Tidak) atau Y (Ya)
    10. Edema: adanya edema N (tidak ada edema dan tidak ada terapi diuretik untuk edema), S (ada edema tanpa diuretik, atau edema teratasi dengan diuretik), atau Y (edema meskipun dengan terapi diuretik)
    11. Bilirubin: bilirubin serum dalam [mg/dl]
    12. Kolesterol: kolesterol serum dalam [mg/dl]
    13. Albumin: albumin dalam [gm/dl]
    14. Tembaga: tembaga urin dalam [ug/hari]
    15. Alk_Phos: alkaline phosphatase dalam [U/liter]
    16. SGOT: SGOT dalam [U/ml]
    17. Trigliserida: trigliserida dalam [mg/dl]
    18. Trombosit: trombosit per kubik [ml/1000]
    19. Protrombin: waktu protrombin dalam detik [s]
    20. Stadium: stadium histologis penyakit (1, 2, 3, atau 4)
    ''')

    st.write("""
    Ada empat fitur yang tidak akan digunakan yaitu ID, N-Day(hari mulai saat diputuskan menderita penyakit sirosis), 
    status(masih hidup atau sudah meninggal), Drug(obat yang diminum),  dan Age(umur dalam bentuk hari). 
    Karena tujuan analisis ini untuk mengetahui tingkat stadium seseorang yang terkena sirosis, 
    jadi hanya perlu fitur-fitur yang berhubungan dengan gejalanya.
    """)
    st.write("""
    Karena data dari urutan 314 sampai 418 terdapat 9 fitur dengan isian NA (kosong), maka data tidak akan dipakai. 
    untuk data missing valaue pada data ke 1 sampai 313 akan di isi dengan rata-rata dari fitur yang terdapat missing value.
    """)

    st.write('\n')
    st.write('\n')
    st.write("Source Code : [https://github.com/fadetul-f/cirhosis-app)")


pd_crs.rename(columns = {"Sex": "gender"}, inplace=True)

del(pd_crs['ID'], pd_crs['Status'], pd_crs['Drug'], pd_crs['N_Days'], pd_crs['Age'])
pd_crs.head()

# label_encoder object
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'gender'.
pd_crs['gender']= label_encoder.fit_transform(pd_crs['gender'])
pd_crs['gender'].unique()

# Encode labels in column 'Ascites'.
pd_crs['Ascites']= label_encoder.fit_transform(pd_crs['Ascites'])
pd_crs['Ascites'].unique()

# Encode labels in column 'Hepatomegaly'.
pd_crs['Hepatomegaly']= label_encoder.fit_transform(pd_crs['Hepatomegaly'])
pd_crs['Hepatomegaly'].unique()

# Encode labels in column 'Spiders'.
pd_crs['Spiders']= label_encoder.fit_transform(pd_crs['Spiders'])
pd_crs['Spiders'].unique()

# Encode labels in column 'Edema'.
pd_crs['Edema']= label_encoder.fit_transform(pd_crs['Edema'])
pd_crs['Edema'].unique()

scaler = MinMaxScaler()
crs_new = pd.DataFrame(pd_crs, columns=['Bilirubin',	'Cholesterol',	'Albumin',	'Copper',	
                                    'Alk_Phos',	'SGOT',	'Tryglicerides',	'Platelets',	'Prothrombin'])
scaler.fit(crs_new)
crs_new = scaler.transform(crs_new)

crs_new = DataFrame(crs_new)
pd_crs_stage = pd.DataFrame(pd_crs, columns = ['Stage'])
del(pd_crs['Stage'], pd_crs['Bilirubin'], pd_crs['Cholesterol'], pd_crs['Albumin'], pd_crs['Copper'],
pd_crs['Alk_Phos'], pd_crs['SGOT'], pd_crs['Tryglicerides'], pd_crs['Platelets'], pd_crs['Prothrombin'])

pd_crs_new = pd.concat([pd_crs,crs_new], axis=1)

pd_crs_new.rename(columns = {0: "Bilirubin", 1: "Cholesterol", 2: "Albumin", 3: "Copper", 
                    4: "Alk_Phos", 5: "SGOT", 6: "Tryglicerides", 7: "Platelets", 
                    8: "Prothrombin"}, inplace=True)

data_crs = pd.concat([pd_crs_new,pd_crs_stage], axis=1)

if selected == "Prediksi":
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.header('Input Parameters')

    with st.form(key='my_form'):
        jk = st.number_input('Jenis Kelamin')
        ascites = st.number_input('Ascites')
        hepa = st.number_input('Hepatomegaly')
        sp = st.number_input('Spiders')
        ede = st.number_input('Edema')
        blr = st.number_input('Bilirubin')
        clt = st.number_input('Cholesterol')
        albu = st.number_input('Albumin')
        cho = st.number_input('Chopper')
        alk = st.number_input('Alk_Phos')
        sgot = st.number_input('SGOT')
        tryg = st.number_input('Tryglicerides')
        pla = st.number_input('Platelets')
        pro = st.number_input('Prothrombin')
        if st.form_submit_button('Check'):
            data = {'Prothrombin':pro, 'Platelets':pla,  'Tryglicerides':tryg,
                        'SGOT': sgot, 'Alk_Phos': alk,'Chopper':cho, 'Albumin': albu, 
                        'Cholesterol': clt, 'Bilirubin':blr, 'Edema':ede, 'Spiders':sp, 
                        'Hepatomegaly':hepa, 'Ascites':ascites, 'Sex':jk
            }
            masukan = pd.DataFrame(data, index=[0])

            # memisahkan fitur dan label
            feature=data_crs.iloc[:,0:14].values
            label=data_crs.iloc[:,14].values
            x_train,x_test,y_train,y_test=train_test_split(feature, label, test_size=0.3,random_state=0)
            st.write("#### KNN")
            classifier= KNeighborsClassifier(n_neighbors=5 )  
            classifier.fit(x_train, y_train)
            #lakukan prediksi
            y_pred= classifier.predict(x_test)
            knn=accuracy_score(y_test, y_pred)

            #load_model=pickle.load(open('DecisionTree.pkl', 'rb'))
            #prediction = load_model.predict(masukan)
            prediction=classifier.predict(masukan)

            st.write('''#### Prediksi''')
            st.write(prediction)


if selected == "Model":
    st.write("### Akurasi Model")

    # memisahkan fitur dan label
    feature=data_crs.iloc[:,0:14].values
    label=data_crs.iloc[:,14].values

    st.write("#### Gaussian Naive Bayes")
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size = 0.24, random_state = 1)
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    akurasi=accuracy_score(y_test, y_pred)
    st.info(akurasi)

    st.write("#### Random Forest")
    X_train,X_test,Y_train,Y_test=train_test_split(feature, label, test_size=0.3,random_state=0)
    clf = RandomForestClassifier(n_estimators = 100) 
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    randomF=accuracy_score(Y_test, y_pred)
    st.info(randomF)

    st.write("#### KNN")
    classifier= KNeighborsClassifier(n_neighbors=5 )  
    classifier.fit(x_train, y_train)
    #lakukan prediksi
    y_pred= classifier.predict(x_test)
    knn=accuracy_score(y_test, y_pred)
    st.info(knn)

    st.write("#### Decision Tree")
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=1)
    #klasifikasi menggunakan decision tree
    clf = tree.DecisionTreeClassifier(random_state=3, max_depth=1)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    dr=accuracy_score(y_test, y_pred)
    st.info(dr)




