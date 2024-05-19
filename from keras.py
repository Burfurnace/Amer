from keras.models import load_model
import numpy as np
from PIL import Image

# Eğitilmiş modelin yolu
model_path = r'C:\Users\burkan\Desktop\dosyalar\eğitilmiş dosyalar1\model_inception.hdf5'

# Eğitilmiş modeli yükleme
model = load_model(model_path)

# Giriş resminin yolu
img_path = r'C:\Users\burkan\Desktop\dosyalar\1ae4880cc953aa21.jpg'

# Giriş resmini yükleme ve işleme
img_width, img_height = 224, 224  # Modelin beklediği boyuta uygun olarak ayarlayın
img = Image.open(img_path)
img = img.resize((img_width, img_height))  # Modelin beklediği boyuta yeniden boyutlandırma
img_array = np.array(img) / 255.0  # Normalizasyon
img_array = np.expand_dims(img_array, axis=0)  # Boyutu genişletme, çünkü tek bir resim kullanıyoruz

# Tahmin yapma
predictions = model.predict(img_array)

# Sınıflar
classes = ['boat', 'stop sign']  # Modelin öğrenmiş olabileceği sınıf etiketleri

# Tahmin sonuçları
predicted_class_index = np.argmax(predictions)
predicted_class = classes[predicted_class_index]
confidence = predictions[0][predicted_class_index]

print("Predicted Class:", predicted_class)
print("Confidence:", confidence)
