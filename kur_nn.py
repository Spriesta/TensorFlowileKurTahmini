# datamızı 5 dakika da  bir kur değerlerini yerleştiriyoruz ve gelecek teki 5 ve 10 nuncu kur değerini bulması için train, test ettirip çıkış verisini bulmasını sağlıyoruz


import pandas as pd             # csv dosyasını okumak için kullanıcaz
import numpy as np                  

from keras.models import Sequential  # kerastan model modülünü import  ediyoruz
from keras.layers import Dense  # kerastan tabakayı modülünü import  ediyoruz. bu modül ek olarak bir tabaka daha eklememize olanak sağlıyor


windows_size = 70       # Onceki 350 dk içinde ki dataları alıp önümüzde ki değerleri tahmin etmeye çalışıyoruz
output_size = 2         # sonraki 2 degeri tahmin etmeye calis
batch_size = 8          # hangi aralıklarla ağırlık katsayasını update ediceğimizi söylüyoruz
epochs = 700            # islemi tekrar sayisi(data üzerinde ki tur sayısı)


def get_data(y):

    train_size = int(len(y)*0.7)                # Verinin %70 ini egitim icin kalaninin da test icin ayiracagiz, 180 tane

    ############## TRAIN DATA ####################
    train_x = []
    train_y = []
    for i in range(0, train_size):       # 0 dan % 70 e kadar döngü oluşturuyoruz
        train_x.append(y[i:i + windows_size])  # % 70 e kadar train ediyoruz
        train_y.append((y[i + windows_size:i + windows_size + output_size]))  # dataları sonraki 2 değeri ve çıkış değerini train listemize append ediyoruz

    train_x = np.array(train_x)   # train arraylerimizi numpy'a dönüştürüyoruz
    train_y = np.array(train_y)

    ########### TEST DATA ########################
    test_x = []  # test array lerimizi oluşturduk
    test_y = []
    last = len(y) - output_size - windows_size
    for i in range(train_size, last):     # tranin size dan başlayıp data nın bitimine kadar döngüye alıyoruz
        test_x.append(y[i:i + windows_size])   
        test_y.append(y[i + windows_size:i + windows_size + output_size])  # dataları sonraki 2 değeri ve çıkış değerini test listemize append ediyoruz

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    ######## Tahmin edilecek data #######################
    data_x = [y[-windows_size:len(y)]]   # en son tahmin edilen sondan geriye 70 tanesi veriyi sisteme input olarak vericez
    data_x = np.array(data_x)  # data x'i numpy'a çeviriyoruz

    return train_x, train_y, test_x, test_y, data_x


raw_data = pd.read_csv('./datasets/ue128.csv', header=None, names=["i", "t", "y"]) # datamızı pandas yardımıyla okutuyoruz
t = np.array(raw_data.t.values)
y = np.array(raw_data.y.values)

min = y.min()
max = y.max()

y = np.interp(y, (min, max), (-1, +1))  # datayı belli bir aralıkta tutmamız gerekli aksi taktirde bazı değerler çok büyük olursa diğer değerleri domine eder bu yüzden -1 +1 değerleri arasına kısıtlıyoruz

x_train, y_train, x_test, y_test, data_x = get_data(y) #fonsiyonları çağırıp datamızı   alıyoruz

model = Sequential()  # modelimizi oluşturduk
model.add(Dense(32, input_dim=windows_size, activation='relu'))  # tabaka(dense) ekliyoruz. birinci layer için 32 tane eleman kullanıyoruz 
model.add(Dense(64))  #ikinci tabaka için 64 tane eleman ekliyoruz
model.add(Dense(output_size))  # son tabakımız  sistemimiz çıkışı

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy']) # modeli compile(derliyoruz) ediyoruz
model.fit(x_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size) # modelimize datayı train ile öğretiyoruz

score = model.evaluate(x_test, y_test, batch_size=batch_size)  # başarı oranımızı test ediyoruz
print("%2s: %.2f%%" % (model.metrics_names[1], score[1]*100)) #başarı oranımızı yazdırıyoruz
model.summary()

data_y = model.predict(data_x) #bugünden geçmişe doğru 70 tane veriyi veriyoruz ve gelecekteki 2 değeri bulduruyoruz

result = np.interp(data_y, (-1, +1), (min, max))    # çıkan 2 değeri bulup, diğer değerleri domine etmesini engelleyen -1 +1 aralığını tersine çeviriyoruz ve asıl değerlere ulaşıyoruz

print("Gelecekteki Degerler (output_size) :", result) # geleecekte ki 2 değeri yazdırıyoruz
