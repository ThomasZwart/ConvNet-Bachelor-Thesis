## Een model trainen
#### Data voorbereiden fillData.py
Maak eerst in dezelfde map als deze repository twee folders genaamd 'data' en 'nulldata'. 
Download vervolgens de klassen waarop je wilt trainen van de volgende link https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap/?pli=1
en zet ze in de 'data' folder. Mocht je een 'null-klasse' ter controle willen, download
dan ook nog andere klassen en zet ze in de 'nulldata' folder. Geef vervolgens in
fillData.py aan hoeveel data je per klasse wilt gebruiken met de variabele 'data_limit_per_label'
en vul de array 'labels' met strings die corresponderen met de klassen die je hebt 
gedownload in de 'data' folder. De variabele 'n_classes' moet gelijk zijn aan het aantal
klassen dat je hebt gedownload in de 'data' folder en als je een 'null-klasse' gebruikt
telt dit ook als extra klasse. Run deze code om een .npy bestand te krijgen waarop het model
getraind kan worden.
###### Waarschuwing: Deze code kost veel werkgeheugen.


#### Model Trainen cnn.py
Als je de data hebt voorbereid kan je met cnn.py het model trainen. Laad het .npy data bestand
in. Doe dit door de naam van het bestand in te voeren bij de variabele 'data' bovenin cnn.py. 
Splits vervolgens de data op naar wens met de variabelen 'train_data' en 'test_data'.
Zet de parameters ook naar eigen wens, 'n_classes' moet gelijk zijn aan het aantal klassen
waarop je traint en 'use_saved_model' moet 'False' zijn. Het model wordt op de 10 epochs
opgeslagen in een .ckpt bestand, de naam hiervan kan zelf ingevuld worden, dit is te vinden
in de laatste functie 'train_neural_network'. 
###### Waarschuwing: Het trainen van een neuraal netwerk op de CPU is niet efficiënt en duurt erg lang. Om te trainen op de GPU is tensorflow-gpu nodig, raadpleeg deze link voor meer informatie: https://www.tensorflow.org/install/gpu



