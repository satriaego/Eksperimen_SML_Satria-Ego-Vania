import pandas as pd

def preprocess_titanic_data(filepath):
    """
    Fungsi otomatisasi khusus untuk dataset Titanic.
    Mengonversi seluruh tahapan eksperimen notebook menjadi fungsi terstruktur.
    """
    # 1. Memuat Data
    df = pd.read_csv(filepath)
    
    # 2. Menangani Data Kosong
    df = df.drop(columns=['Cabin'], errors='ignore') 
    df['Age'] = df['Age'].fillna(df['Age'].median())  
    df['Fare'] = df['Fare'].fillna(df['Fare'].median()) 
    
    # 3. Menghapus Data Duplikat
    df = df.drop_duplicates()
    
    # 4. Menghapus Kolom Administrasi
    kolom_tidak_butuh = ['PassengerId', 'Name', 'Ticket']
    df_selected = df.drop(columns=kolom_tidak_butuh, errors='ignore')
    
    # 5. Encoding Data Kategorikal
    df_ready = pd.get_dummies(df_selected, columns=['Sex', 'Embarked'], drop_first=True)
    
    return df_ready