# hida_covid_challenge_repeat
 repeating the HIDA covid datathon challenge, let's see how far I can get

Geeigneten Imputer finden


# Übersicht über die Daten 

(Gelb = Fehlt )
![Matrixübersicht_OhneBild](https://user-images.githubusercontent.com/57065083/128595496-d45512fa-8a1e-4951-8b9d-32d6b3270e90.png)

unterschedliche verteilung 

Fibogen
DDimer
SaO2
Fehlen deutlich mehr Daten als bei dem Rest 

(underfitting potential)

PatientID,  Fehlende Daten: 0, Prozent 0.00

ImageFile,  Fehlende Daten: 0, Prozent 0.00

Hospital,  Fehlende Daten: 0, Prozent 0.00

Age,Fehlende Daten: 1, Prozent 0.12

Sex,                    Fehlende Daten: 0, Prozent 0.00

Temp_C,                 Fehlende Daten: 154, Prozent 17.84

Cough,                  Fehlende Daten: 0, Prozent 0.00

DifficultyInBreathing,  Fehlende Daten: 4, Prozent 0.46

WBC,  Fehlende Daten: 9, Prozent 1.04

CRP,  Fehlende Daten: 33, Prozent 3.82

>Fibrinogen,  Fehlende Daten: 591, Prozent 68.48      <----------

LDH,  Fehlende Daten: 136, Prozent 15.76

>Ddimer,  Fehlende Daten: 621, Prozent 71.96      <----------

Ox_percentage,  Fehlende Daten: 243, Prozent 28.16

PaO2,  Fehlende Daten: 170, Prozent 19.70

>SaO2,  Fehlende Daten: 583, Prozent 67.56        <---------

pH,  Fehlende Daten: 207, Prozent 23.99       

CardiovascularDisease,  Fehlende Daten: 19, Prozent 2.20

RespiratoryFailure,  Fehlende Daten: 159, Prozent 18.42

Prognosis,  Fehlende Daten: 0, Prozent 0.00


Eventuell eine Spalte einführen, die die Prognosis hervorhebt...
