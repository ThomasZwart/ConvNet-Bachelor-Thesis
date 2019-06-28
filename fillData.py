from os import listdir
import numpy as np

x = 0 
n_classes = 21
data_dir = './data/'
data_limit_per_label = 20000
data = None
# De 20 klassen (gebruikt voor het maken van one-hot labels)
labels = ['ant', 'bread', 'carrot', 'cloud', 'cup', 'dolphin', 'feather', 'fish', 'flower', 'giraffe', 'horse', 'lion', 'mountain', 'onion', 'panda', 'pizza', 'rainbow', 'saw', 'smiley face', 'washing machine']

for file_name in listdir(data_dir):
  # Lees de file
  data_t = (np.load(data_dir + file_name, mmap_mode='r')).copy()
  # Alles boven 90000 is gereserveerd voor test data
  data_t = data_t[:90000]
  np.random.shuffle(data_t)
  data_t = data_t[:data_limit_per_label]
  
  # Make a one-hot label
  label = file_name.split('_')[3].split('.')[0]
  index = labels.index(str(label))
  labelarray = np.zeros(n_classes)
  labelarray[index] = 1
  labeleddata = None

  # Voeg label toe aan foto data
  for drawing in data_t:
      if labeleddata is None:
          labeleddata = np.array([[drawing / 255, labelarray]])
      else:
          labeleddata = np.concatenate((labeleddata, np.array([[drawing / 255, labelarray]])))

  # Maak de dataset
  if data is None:
    data = labeleddata
  else:
    data = np.concatenate((data, labeleddata))
  print(file_name)

null_data = None
# Maakt de null klasse, met een combinatie van schetsen uit 30 andere klassen
for file_name in listdir('./nulldata/'):
  data_t = (np.load('./nulldata/' + file_name, mmap_mode='r')).copy()
  np.random.shuffle(data_t)
  # Aantal data per klasse delen door 30
  data_t = data_t[:(int)(data_limit_per_label/30)]
  labelarray = np.zeros(n_classes)
  labelarray[n_classes - 1] = 1
  labeleddata = None;

  for drawing in data_t:
      if labeleddata is None:
          labeleddata = np.array([[drawing / 255, labelarray]])
      else:
          labeleddata = np.concatenate((labeleddata, np.array([[drawing / 255, labelarray]])))
  
  if null_data is None:
    null_data = labeleddata
  else:
    null_data = np.concatenate((null_data, labeleddata))
  print(file_name)

data = np.concatenate((data, null_data)) 

print("saving...")
np.save("testdata", data)
print(data.shape)
