import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from pandas import DataFrame


# st.set_page_config(
#     page_title="Multipage App",
# )

selected = option_menu(
    menu_title = None, #wajib ada
    options=["Dataset", "Prepocessing", "Model", "Prediksi"],
    icons=["book","cast", "book", "envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container":{"padding":"0!important", "background-color":"#ffffff",},
        "icons":{"font-size":"14px"},
        "nav-link":{"font-size":"15px",
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

pd_crs.rename(columns = {"Sex": "gender"}, inplace=True)

if selected == "Dataset":
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
    4. Obat : jenis obat yang dikonsumsi pasien D-penicillamine atau placebo
    5. Umur: umur pasien dalam [hari]
    6. Jenis Kelamin: Jenis kelamis pasien M adalah laki-laki dan F adalah perempuan
    7. Asites: penumpukan cairan di dalam rongga antara selaput yang melapisi dinding perut (peritoneum) dan organ dalam perut. N (Tidak ada) atau Y (Ada)
    8. Hepatomegali: adalah pembesaran organ hati melebihi ukuran normalnya. Kondisi ini merupakan gejala gangguan pada hati atau organ yang terkait dengan hati, seperti kantong empedu. N (Tidak) atau Y (Ya)
    9. Spiders : adalah suatu kondisi yang menyebabkan kumpulan pembuluh darah kecil yang menyerupai sarang laba-laba terlihat pada permukaan kulit. N (Tidak) atau Y (Ya)
    10. Edema: adalah Bengkak karena akumulasi cairan pada tungkai. edema N (tidak ada edema dan tidak ada terapi diuretik untuk edema), S (ada edema tanpa diuretik, atau edema teratasi dengan diuretik), atau Y (edema meskipun dengan terapi diuretik)
    11. Bilirubin: adalah senyawa pigmen berwarna kuning yang merupakan produk katabolisme enzimatik biliverdin oleh biliverdin reduktase. nilai normal bilirubin direk atau langsung adalah dari 0-0,4 miligram per desiliter (mg/dL). Sementara nilai normal bilirubin total adalah dari 0,3-1,0 mg/dL.
    12. Kolesterol: adalah lemak yang diproduksi oleh tubuh, dan juga berasal dari makanan hewani.kadar kolesterol total kurang dari 200 mg/dL, maka masih di batas normal. Namun, jika sudah mencapai 200-239 mg/dL termasuk batas tinggi. Dan dikategorikan kolesterol tinggi bila lebih dari 240 mg/dL.
    13. Albumin: adalah istilah yang digunakan untuk merujuk ke segala jenis protein monomer yang larut dalam air atau garam dan mengalami koagulasi ketika terpapar panas.Nilai normal albumin serum dalam darah adalah 340-540 gm/dL
    14. Copper: merupakan zat tembaga bagian dari enzim dan berperan aktif untuk menjaga kesehatan sel darah. copper normal adalah 15-60 ug/hari
    15. Alk_Phos: adalah salah satu enzim hidrolase yang terutama ditemukan pada sebagian besar organ tubuh, terutama dalam jumlah besar di hati, tulang, dan plasenta. normalnya adalah 20 hingga 140 IU/L
    16. SGOT: adalah enzim yang biasanya ditemukan pada organ hati (liver), jantung, ginjal, hingga otak. normal dengan batas SGOT 5 - 40 µ/L (mikro per liter) [U/ml]
    17. Trigliserida: adalah salah satu jenis lemak yang mengalir di dalam darah. Zat tersebut berfungsi menyimpan kalori dan menyediakan energi untuk tubuh. trigliserida normal kurang dari 150 mg/dl [mg/dl]
    18. Trombosit: adalah sel terkecil dari darah yang jumlah normalnya berkisar antara 150.000-450.000 keping per mikro liter darah [ml/1000]
    19. Protrombin: adalah sejenis glikoprotein yang dibentuk oleh dan disimpan dalam hati.
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
    st.write("Source Code : https://github.com/fadetul-f/cirhosis-app")

if selected == "Prepocessing":
    nomalisasi = st.selectbox("Pilih tipe dari data yang akan di normalisasi",['Binary', 'Kategori', 'Numerik', 'Normalisasi semua'])

    if nomalisasi == 'Binary':
        del(pd_crs['ID'], pd_crs['Status'], pd_crs['Drug'], pd_crs['N_Days'], pd_crs['Age'])
        pd_crs.head()

        col1, col2 = st.columns([1, 1])

        col1.write("##### Data Binary sebelum normalisasi")
        col1.write(pd_crs.iloc[:,0:4])

        
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

        col2.write("##### Data setelah normalisasi")
        col2.write(pd_crs.iloc[:,0:4])

    elif nomalisasi == 'Kategori':
        col3, col4 = st.columns([1, 1])

        col3.write("##### Data Kategori sebelum normalisasi")
        col3.write(pd_crs['Edema'])

        kategori=pd.get_dummies(pd_crs['Edema'])
        kategori.rename(columns = {'N': "Edema N", 'S': "Edema S", 'Y': "Edema Y"}, inplace=True)

        col4.write("##### Data setelah normalisasi")
        col4.write(kategori)

    elif nomalisasi == 'Numerik':
        scaler = MinMaxScaler()
        crs_new = pd.DataFrame(pd_crs, columns=['Bilirubin',	'Cholesterol',	'Albumin',	'Copper',	
                                            'Alk_Phos',	'SGOT',	'Tryglicerides',	'Platelets',	'Prothrombin'])
        col5, col6 = st.columns([1, 1])

        col5.write("##### Data Numerik sebelum normalisasi")
        col5.write(crs_new)

        scaler.fit(crs_new)
        crs_new = scaler.transform(crs_new)

        crs_new = DataFrame(crs_new)

        crs_new.rename(columns = {0: "Bilirubin", 1: "Cholesterol", 2: "Albumin", 3: "Copper", 
                            4: "Alk_Phos", 5: "SGOT", 6: "Tryglicerides", 7: "Platelets", 
                            8: "Prothrombin"}, inplace=True)

        col6.write("##### Data setelah normalisasi")
        col6.write(crs_new.iloc[:,0:])
    else:
        
        del(pd_crs['ID'], pd_crs['Status'], pd_crs['Drug'], pd_crs['N_Days'], pd_crs['Age'])
        pd_crs.head()

        st.write("#### Data sebelum normalisasi")
        st.write(pd_crs.iloc[:5])

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

        kategori=pd.get_dummies(pd_crs['Edema'])
        kategori.rename(columns = {'N': "Edema N", 'S': "Edema S", 'Y': "Edema Y"}, inplace=True)

        scaler = MinMaxScaler()
        crs_new = pd.DataFrame(pd_crs, columns=['Bilirubin',	'Cholesterol',	'Albumin',	'Copper',	
                                            'Alk_Phos',	'SGOT',	'Tryglicerides',	'Platelets',	'Prothrombin'])
        scaler.fit(crs_new)
        crs_new = scaler.transform(crs_new)

        crs_new = DataFrame(crs_new)
        pd_crs_stage = pd.DataFrame(pd_crs, columns = ['Stage'])
        del(pd_crs['Stage'], pd_crs['Bilirubin'], pd_crs['Cholesterol'], pd_crs['Albumin'], pd_crs['Copper'],
        pd_crs['Alk_Phos'], pd_crs['SGOT'], pd_crs['Tryglicerides'], pd_crs['Platelets'], pd_crs['Prothrombin'], pd_crs['Edema'])

        pd_crs_new = pd.concat([pd_crs, kategori, crs_new], axis=1)

        pd_crs_new.rename(columns = {0: "Bilirubin", 1: "Cholesterol", 2: "Albumin", 3: "Copper", 
                            4: "Alk_Phos", 5: "SGOT", 6: "Tryglicerides", 7: "Platelets", 
                            8: "Prothrombin"}, inplace=True)

        st.write("#### Data setelah normalisasi")
        st.write(pd_crs_new.iloc[:5])


if selected == "Prediksi":
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.header('Input Parameters')

    with st.form(key='my_form'):
        col0, col1 = st.columns([1,1])
        with col0:
            jk = st.radio('Pilih Jenis Kelamin', ('Laki-laki', 'Perempuan'))
            if jk == 'Laki-laki':
                jenis = 1
            else:
                jenis = 0
            ascites = st.radio('Ascites', ('Ada', 'Tidak'))
            if ascites == 'Ada':
                asc = 1
            else:
                asc = 0
            hepa = st.radio('Hepatomegaly', ('Ada', 'Tidak'))
            if hepa == 'Ada':
                hp = 1
            else:
                hp = 0
            spd = st.radio('Spiders', ('Ada', 'Tidak'))
            if spd == 'Ada':
                sp = 1
            else:
                sp = 0
            ede = st.radio('Edema', ('Ada', 'Tidak', 'Teratasi'))
            if ede == 'Ada':
                EdeY = 1
                EdeN = 0
                EdeS = 0
            elif ede == 'Tidak':
                EdeY = 0
                EdeN = 1
                EdeS = 0
            else:
                EdeY = 0
                EdeN = 0
                EdeS = 1
            blr = st.number_input('Bilirubin')
            
        with col1:
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
                        'Cholesterol': clt, 'Bilirubin':blr, 'Edema N':EdeN, 'Edema Y':EdeY, 
                        'Edema S':EdeS,'Spiders':sp, 'Hepatomegaly':hp, 'Ascites':asc, 'Sex':jenis
            }
            masukan = pd.DataFrame(data, index=[0])

            loaded_model = pickle.load(open('RandomForest_model.sav', 'rb'))
            prediction = loaded_model.predict(masukan)

            st.write('''#### Prediksi dengan Metode Random Forest''')
            st.write(prediction)


if selected == "Model":
    del(pd_crs['ID'], pd_crs['Status'], pd_crs['Drug'], pd_crs['N_Days'], pd_crs['Age'])

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

    kategori=pd.get_dummies(pd_crs['Edema'])
    kategori.rename(columns = {'N': "Edema N", 'S': "Edema S", 'Y': "Edema Y"}, inplace=True)

    scaler = MinMaxScaler()
    crs_new = pd.DataFrame(pd_crs, columns=['Bilirubin',	'Cholesterol',	'Albumin',	'Copper',	
                                    'Alk_Phos',	'SGOT',	'Tryglicerides',	'Platelets',	'Prothrombin'])
    scaler.fit(crs_new)
    crs_new = scaler.transform(crs_new)

    crs_new = DataFrame(crs_new)
    pd_crs_stage = pd.DataFrame(pd_crs, columns = ['Stage'])
    del(pd_crs['Stage'], pd_crs['Bilirubin'], pd_crs['Cholesterol'], pd_crs['Albumin'], pd_crs['Copper'],
    pd_crs['Alk_Phos'], pd_crs['SGOT'], pd_crs['Tryglicerides'], pd_crs['Platelets'], pd_crs['Prothrombin'], pd_crs['Edema'])

    pd_crs_new = pd.concat([pd_crs, kategori, crs_new], axis=1)

    pd_crs_new.rename(columns = {0: "Bilirubin", 1: "Cholesterol", 2: "Albumin", 3: "Copper", 
                    4: "Alk_Phos", 5: "SGOT", 6: "Tryglicerides", 7: "Platelets", 
                    8: "Prothrombin"}, inplace=True)

    data_crs = pd.concat([pd_crs_new,pd_crs_stage], axis=1)
    st.write("### Akurasi Model")

    # memisahkan fitur dan label
    feature=data_crs.iloc[:,0:16].values
    label=data_crs.iloc[:,16].values
    x_train,x_test,y_train,y_test=train_test_split(feature, label, test_size=0.3,random_state=0)

    st.write("#### Gaussian Naive Bayes")
    loaded_model = pickle.load(open('NaiveBayes.sav', 'rb'))
    y_pred = loaded_model.predict(x_test)
    akurasi=accuracy_score(y_test, y_pred)
    st.info(akurasi)

    st.write("#### Random Forest")
    loaded_model = pickle.load(open('RandomForest_model.sav', 'rb'))
    pred = loaded_model.predict(x_test)
    randomF=accuracy_score(y_test, pred)
    st.info(randomF)

    st.write("#### KNN")
    loaded_model = pickle.load(open('KNN_model.sav', 'rb'))
    pred1 = loaded_model.predict(x_test)
    knn=accuracy_score(y_test, pred1)
    st.info(knn)

    st.write("#### Decision Tree")
    loaded_model = pickle.load(open('DecisionTree.sav', 'rb'))
    pred2 = loaded_model.predict(x_test)
    dr=accuracy_score(y_test, pred2)
    st.info(dr)

    fig = plt.figure()
    fig.patch.set_facecolor('grey')
    fig.patch.set_alpha(0.5)
    ax = fig.add_axes([0,0,1,1])
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.4)
    ax.plot(['Naive Bayes', 'Random Forest', 'KNN', 'Decision Tree'],[akurasi, randomF, knn, dr], color='green')
    plt.show()
    st.pyplot(fig)




