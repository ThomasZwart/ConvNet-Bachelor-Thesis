fillData.py:
De data wordt hier gelabeled en gemaakt zodat het neurale netwerk er gebruik van kan maken.
De labels die in de code staan zijn corresponderend met de titel die Google de numpy data heeft
gegeven, bijvoorbeeld: "full_numpy_bitmap_ant.npy". Alleen het 'ant' gedeelde blijft over van
de titel na .split('_')[3].split('.')[0] en dat correspondeert met de labels in de array 'labels'
Vervolgens worden de labels toegevoegd aan de data en de null klasse wordt op dezelfde manier 
gemaakt. Als laatst wordt de data opgeslagen in een nieuwe .npy (numpy) file.

Als je deze code zelf wilt gebruiken, kan dat door in dezelfde folder als de code een nieuwe
folder 'data' aan te maken en daarin de .npy bestanden van 
'https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap/?pli=1'
te downloaden. En de labels in de array 'labels' aan te passen corresponderend aan de 
gedownloade bestanden. Daarna kan je ervoor kiezen of je een null klasse wilt, zo ja, doe 
hetzelfde nog een keer maar dan met een folder 'nulldata'. Hierbij hoeven geen labels 
veranderd te worden.



cnn.py:
Er wordt data opgehaald in een numpy array. De data die wordt opgehaald is gemaakt in fillData.py.
Vervolgens kan de data opgesplitst worden in trainingsdata en validatiedata, daarnaast kunnen ook
de parameters voor het convolutionele neurale netwerk gezet worden. De parameter 'n_classes' 
moet gelijk zijn aan het aantal klassen dat is aangemaakt in fillData.py. Als een oud model 
ingeladen moet worden zet dan 'use_saved_model' op 'True'. In de functie 'train_neural_network'
wordt het model opgehaald en/of opgeslagen, gebruik hierbij de juiste path in 'saver.save(...)'.


model:
Hierin staan al wat getrainde modellen opgeslagen. De parameters gebruikt staan in de titel.
Lees de titel als volgt: eerste nummer is de batch size, tweede nummer is het procentueel 
aantal dropout en daarachter staat op welke laag de dropout is toegevoegd.