# Football Player Market Value Prediction: Uncover the Next Million-Dollar Talent

**by Nabil Julian Syah**  
[Original on Medium](https://medium.com/@nabil.julian04/football-player-market-value-prediction-uncover-the-next-million-dollar-talent-7bcf87f353c2)

![Neuron](https://miro.medium.com/v2/resize:fit:640/format:webp/0*vnTb5rgEN3rF1luH.jpg "Judul Gambar")


Tantangan kali ini adalah membuat model Neural Network dengan tujuan untuk memprediksi nilai pasar pemain dalam euro. Dataset disajikan dalam beebrapa file terpisah-pisah meliputi informasi pemain, peforma pertandingan, klub dan kompetisi, serta riwayat transfer. Salah satu tantangan adalah mengintegrasikan data tersebut menjadi satu set fitur yang komprehensif. Langkah-langkah yang akan dilakukan mencangkup:

1. Exploratory Data Analysis (EDA): Memahami struktur data, hubungan antar variabel, dan terutama memilih fitur apa yang di gunakan pada tahap selanjutnya.  
2. Feature Engineering: Membuat fitur baru yang mempresentasikan peforma historis, stabilitas karier, dan faktor kontekstual kompetisi.  
3. Data integration: Menggabungkan berbagai file ke dalam satu tabel analisis.  
4. Preprocessing: Menangani missing value dan melakukan normalisasi.  
5. Modeling: Merancang dan melatih Neural Network dengan arsitektur yang sesuai, termasuk eksperimen layer, neuron, dan fungsi aktivasi.  
6. Evaluasi: Menggunakan metrik Root Mean Squared Error (RMSE) untuk menilai peforma model.

Dengan Kerangka kerja ini, diharapkan model Neural Network yang dikembangkan mampu memberikan prediksi nilai pasar pemain dalam euro dengan tingkatan akurasi yang baik.  
[Our notebook](./Tugas_Modul4.ipynb)

## EDA

…


## Feature Engineering

1. **appearances**  
   ```python
   appearances['goals_per_90'] = appearances['goals'] * 90 / appearances['minutes_played'].replace(0, 90)
   appearances['assists_per_90'] = appearances['assists'] * 90 / appearances['minutes_played'].replace(0, 90)
   appearances['goal_contributions'] = appearances['goals'] + appearances['assists']
   appearances['contribution_per_90'] = appearances['goal_contributions'] * 90 / appearances['minutes_played'].replace(0, 90)
   appearances['cards_total'] = appearances['yellow_cards'] + appearances['red_cards']
    ````
   Fitur ini menstandarisasi jumlah gol, assist, dan kontribusi (gol + assist) menurut durasi bermain per 90 menit, sehingga pemain dengan menit bermain berbeda dapat dibandingkan secara adil. Serta menjumlah   kartu kuning dan merah untuk melihat aspek disiplin.

2. **Agregasi Statistik Pemain**

   ```python
   player_stats = appearances.groupby('player_id').agg({
       'minutes_played': 'sum',
       'goals': 'sum',
       'assists': 'sum',
       'yellow_cards': 'sum',
       'red_cards': 'sum',
       'goals_per_90': 'mean',
       'assists_per_90': 'mean',
       'contribution_per_90': 'mean'
   }).reset_index()
   ```

   Mengombinasikan data per‐pertandingan menjadi ringkasan per pemain, mencakup total menit, jumlah gol, jumlah asist, dan jumlah kartu serta rata-rata metrik per-90 menit.

3. **Rekam Jejak Terbaru**

   ```python
   players_club_latest = players_club.sort_values('date').groupby('player_id').last().reset_index()
   latest_transfers = transfers.sort_values('transfer_date').groupby('player_id').last().reset_index()
   ```

   Mengetahui klub terkini dan riwayat transfer terbaru setiap pemain sangat penting dalam memprediksi harga pasar mereka, karena durasi kontrak, besaran biaya transfer, dan performa awal di klub baru secara langsung mencerminkan permintaan pasar dan kestabilan nilai.


## Encoding Kategorikal

```python
position_mapping = {
    'Attack': 4,
    'Midfield': 3,
    'Defender': 2,
    'Goalkeeper': 1
}
players['position_value'] = players['position'].map(position_mapping)
```

Mengonversi posisi pemain ke nilai numerik (Attack = 4, Midfield = 3, Defender = 2, Goalkeeper = 1) membantu model menimbang perbedaan permintaan pasar dan ekspektasi performa antar posisi secara kuantitatif, sehingga prediksi harga pasar menjadi lebih akurat.

```python
foot_mapping = {'right': 1, 'left': 2, 'both': 3, '': 0}
players['foot_value'] = players['foot'].map(foot_mapping)
```

Mengonversi preferensi kaki dominan pemain ke nilai numerik (right = 1, left = 2, both = 3, tidak diisi = 0) memungkinkan model menangkap kontribusi teknik dan fleksibilitas bermain berdasarkan kaki, sehingga prediksi harga pasar lebih mencerminkan keunikan skill-set tiap pemain.

```python
players['is_attack'] = (players['position'] == 'Attack').astype(int)
players['is_midfield'] = (players['position'] == 'Midfield').astype(int)
players['is_defender'] = (players['position'] == 'Defender').astype(int)
players['is_goalkeeper'] = (players['position'] == 'Goalkeeper').astype(int)
```

Membuat variabel biner (0/1) untuk tiap posisi (is\_attack, is\_midfield, is\_defender, is\_goalkeeper) memastikan model dapat menilai dampak unik setiap peran tanpa menganggap urutan, sehingga prediksi harga pasar mempertimbangkan kontribusi posisi secara terpisah dan lebih akurat.



## Data Integration

Menggabungkan berbagai fitur dari dataset yang terpisah-ke dalam satu `train_df` bertujuan memberikan pandangan menyeluruh tentang setiap pemain, mulai dari atribut personal (posisi, kaki dominan, tinggi, usia, kewarganegaraan), kondisi klub (ukuran skuad, rata-rata usia, persentase orang asing, jumlah pemain timnas), karakter kompetisi (jenis kompetisi, status liga utama), performa individu (statistik pertandingan) hingga nilai transfer terakhir.

Sehingga setiap baris data berisi konteks lengkap yang dibutuhkan model untuk belajar.

Hasil akhirnya adalah:

* Jumlah baris tetap sama seperti dataset asli (19560) karna di gabungkan berdasarkan id yang ada pada `train.`
* Jumlah fitur bertambah dari 2 menjadi 27, mencangkup 25 fitur yang kaya informasi.

```python
train_df_columns = [
  'player_id', 'market_value_in_eur', 'position', 'position_value', 'foot',
  'height_in_cm', 'age', 'country_of_citizenship', 'current_club_id',
  'player_club_domestic_competition_id', 'club_id', 'squad_size',
  'average_age', 'foreigners_percentage', 'national_team_players',
  'competition_id', 'type', 'is_major_national_league', 'minutes_played',
  'goals', 'assists', 'yellow_cards', 'red_cards', 'goals_per_90',
  'assists_per_90', 'contribution_per_90', 'transfer_fee'
]
```

Proses yang sama kemudian diterapkan pada dataset test, agar model yang sudah dilatih dapat langsung mengevaluasi dan memprediksi nilai pasar pemain baru dengan format dan fitur yang identik.



## Preprocessing

Pada tahap ini, kita pertama-tama memisahkan mana yang menjadi label `market_value_in_eur` dan mana yang menjadi fitur (semua informasi lain kecuali `player_id` dan `market_value_in_eur`). Label inilah yang akan kita ajarkan ke model untuk diprediksi, sedangkan fitur adalah data penjelas seperti usia, posisi, statistik pertandingan, dan fitur klub yang digunakan model untuk memahami pola.

Setelah itu, kita membagi data menjadi dua bagian: 80% untuk pelatihan (agar model belajar) dan 20% untuk validasi (untuk menguji seberapa baik model bekerja di data yang belum pernah dilihat), dengan penetapan `random_state=42` supaya split ini konsisten jika diulang.

```python
X = train_df.drop(['player_id', 'market_value_in_eur'], axis=1)
y = train_df['market_value_in_eur']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

Selanjutnya kita perlu menyiapkan dua pengolahan data atau pipeline, satu untuk data numerik dan satunya lagi untuk data kategorikal. Pada pipeline numerik, setiap missing value diisi dengan nilai median agar tidak terpengaruh angka ekstrim, kemudian seluruh kolom dinormalisasi supaya rata-rata menjadi nol dan skala variabel seragam.

Sementara itu, pipeline kategori mengisi missing value dengan kategori yang paling sering muncul, lalu mengubah tiap kategori menjadi beberapa kolom biner menggunakan one-hot encoding agar model dapat memprosesnya sebagai angka. Kedua pipeline ini kemudian digabung menggunakan `ColumnTransformer`, sehingga setiap kolom secara otomatis diproses sesuai jenisnya dalam satu langkah.

```python
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num),
        ('cat', categorical_transformer, kategori)
    ]
)
```

Setelah pipeline disiapkan, kita menjalankannya pada data pelatihan untuk belajar aturan imputasi dan normalisasi, lalu menerapkannya juga pada data validasi. Hasilnya adalah matriks fitur siap pakai dengan 15.648 baris dari 80% data train dan 220 kolom (18 fitur numerik yang sudah dinormalisasi ditambah ratusan kolom hasil one-hot encoding).

Terakhir, karena nilai pasar sangat bervariasi dan cenderung miring, kita menerapkan fungsi `log(x + 1)` pada label ini—sebuah trik matematika untuk meratakan distribusi nilai pasar yang ekstrem sehingga model menjadi lebih stabil dan tidak terlalu terjebak pada kasus-kasus nilai pasar yang sangat tinggi. Dengan semua ini, data kita benar-benar siap dimasukkan ke dalam algoritma regresi untuk melatih dan menghasilkan prediksi nilai pasar pemain.

```python
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
```



## Modeling

Pada tahap ini, kita membangun sebuah model Neural Network yang berfungsi layaknya otak buatan: menerima sekumpulan fitur pemain yang sudah diproses sebelumnya, lalu belajar menghubungkannya dengan nilai pasar yang menjadi target.

```python
nn_model = Sequential([
    # Input layer
    layers.Dense(256, activation='relu', input_dim=X_train_processed.shape[1]),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Hidden layers
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    # Output layer - regresi
    layers.Dense(1)
])
```

Jaringan ini terdiri dari beberapa lapisan: dimulai dari lapisan input yang menerima seluruh kolom (220 fitur), diikuti oleh empat lapisan tersembunyi dengan jumlah neuron yang bertingkat 256, 128, 64, dan 32 yang saling terhubung. Setiap lapisan menggunakan fungsi aktivasi ReLU untuk menambahkan non-linearitas, sehingga jaringan mampu menangkap pola kompleks dalam data.

Agar jaringan tidak terlalu hafal data pelatihan (overfitting), kita menambahkan Batch Normalization di setiap lapisan untuk menstabilkan dan mempercepat proses pembelajaran, serta Dropout dimana sebagian neuron dipadamkan secara acak pada tiap epoch dengan tingkat penonaktifan berkurang dari 40% di lapisan pertama hingga 10% di lapisan paling dalam. Strategi ini membantu jaringan tetap general dan tidak tergantung berlebihan pada subset fitur apa pun.

```python
nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=[
        'mean_absolute_error',
        tf.keras.metrics.RootMeanSquaredError(name='rmse')
    ]
)

callbacks_list = [
    # Early stopping untuk mencegah overfitting
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),
    # Learning rate reduction
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001
    )
]
```

Selama pelatihan, dua callback krusial dijalankan otomatis:

* **EarlyStopping** menghentikan training jika loss pada validasi tidak membaik setelah 20 epoch, lalu memulihkan bobot terbaik agar model tidak terlalu lama belajar pada noise data.
* **ReduceLROnPlateau** menurunkan learning rate setengahnya jika validasi loss stagnan setelah 10 epoch, memungkinkan proses pembelajaran halus ketika mendekati solusi optimal.

```python
history = nn_model.fit(
    X_train_processed, y_train_log,
    epochs=200,
    batch_size=32,
    validation_data=(X_val_processed, y_val_log),
    callbacks=callbacks_list,
    verbose=1
)
```

Training berlangsung hingga maksimal 200 epoch atau hingga EarlyStopping bekerja. Setelah selesai, kita memprediksi kembali nilai pasar pada data validasi dalam skala log, lalu mengembalikannya ke skala asli. Hasil evaluasi pada data validasi menunjukkan Log RMSE ≈ 0,9265, yang merepresentasikan rata-rata kesalahan prediksi dalam skala logaritmik dan menunjukkan performa model dalam menangkap pola nilai pasar pemain.

Dengan begitu, seluruh rangkaian modeling telah lengkap dan siap untuk diaplikasikan pada data baru.



## Visualization of Predictions
![Perbandingan nilai sebenarnya dan prediksi nilai](https://miro.medium.com/v2/resize:fit:786/format:webp/1*e4kcuwOaEIO2PIDdIAPxKg.png)


Dari grafik memperlihatkan perbandingan antara nilai pasar pemain yang sebenarnya (sumbu horizontal) dengan nilai pasar yang diprediksi oleh model (sumbu vertikal). Kedua sumbu menggunakan skala logaritmik, artinya jarak antara 10.000 €, 100.000 €, 1.000.000 €, dan seterusnya terdistribusi secara merata.

* Penyebaran di sekitar garis menunjukkan seberapa dekat (atau jauh) prediksi model terhadap kenyataan. Titik di atas garis berarti model melakukan over-estimasi (prediksi lebih tinggi dari nilai sebenarnya), sedangkan titik di bawah garis berarti model melakukan under-estimasi (prediksi lebih rendah).
* Kerapatan titik yang tebal di sekitar garis, terutama di kisaran 10⁵ — 10⁷, menandakan model umumnya cukup akurat untuk sebagian besar pemain dengan nilai pasar menengah.
* Namun, kita juga dapat melihat beberapa titik yang jauh menyimpang, terutama di ujung kanan atas ini adalah pemain dengan nilai pasar sangat tinggi yang sulit diprediksi, sehingga model kadang meremehkan atau melebihkan nilainya secara signifikan.

Secara keseluruhan, pola menyebar yang relatif rapat di sekitar garis merah menandakan bahwa model kita memiliki kecocokan yang baik dan mampu menangkap tren nilai pasar pemain. Sementara titik-titik yang menyimpang memberi petunjuk area di mana model masih perlu ditingkatkan misalnya dengan menambah data fitur khusus untuk pemain top atau menangani outlier dengan metode yang lebih canggih.

Dengan visualisasi ini, kita dapat langsung melihat kekuatan dan batasan prediksi sebelum akhirnya menerapkannya secara nyata.



## Kesimpulan

Dalam tugas kali ini, kita dituntut untuk membangun model prediksi menggunakan pendekatan Neural Network, dan saya memilih menggunakan arsitektur Artificial Neural Network (ANN) atau Jaringan Syaraf Tiruan. ANN merupakan salah satu jenis model dalam machine learning yang terinspirasi dari cara kerja otak manusia dalam mengolah dan memahami informasi.

ANN terdiri dari beberapa lapisan (layer) yang terdiri atas neuron-neuron. Lapisan pertama disebut input layer, tempat data awal dimasukkan. Lapisan di tengah disebut hidden layers, yang menangani proses pembelajaran dan ekstraksi pola-pola kompleks dari data. Terakhir ada output layer yang menghasilkan prediksi akhir.

Dalam implementasi tugas ini, saya membangun ANN dengan beberapa hidden layer yang dilengkapi dengan teknik tambahan seperti ReLU activation untuk memperkenalkan non-linearitas, batch normalization untuk mempercepat dan menstabilkan pelatihan, serta dropout sebagai mekanisme regularisasi agar model tidak mudah overfitting.

Model ini dilatih menggunakan algoritma backpropagation dengan optimizer Adam yang menyesuaikan bobot jaringan berdasarkan error yang dihasilkan. Untuk menilai kinerja model, digunakan metrik regresi seperti Mean Squared Error (MSE), Mean Absolute Error (MAE), dan Root Mean Squared Error (RMSE). Selain itu, digunakan teknik seperti early stopping untuk menghentikan pelatihan saat model tidak lagi menunjukkan peningkatan, dan learning rate reduction untuk mengatur laju pembelajaran secara dinamis.

Dengan pendekatan ANN ini, model berhasil belajar dari kombinasi fitur-fitur kompleks hasil integrasi dan rekayasa data, serta menunjukkan performa prediksi yang cukup akurat dalam memperkirakan nilai pasar pemain sepak bola. Hasil ini menunjukkan bahwa ANN efektif dalam menangkap pola-pola non-linear yang tersembunyi dalam data multidimensi seperti ini.



*Written by Nabil Julian Syah*
*No responses yet*



*Source: ([Medium][1], [Medium][1])*

[1]: https://medium.com/%40nabil.julian04/football-player-market-value-prediction-uncover-the-next-million-dollar-talent-7bcf87f353c2 "Football Player Market Value Prediction: Uncover the Next Million-Dollar Talent | by Nabil Julian Syah | May, 2025 | Medium"

