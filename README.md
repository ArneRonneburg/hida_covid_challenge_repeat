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

| Column   |  Fehlende Daten| Prozent |  Markant |
| ---------| ---------------|---------|----------|
|PatientID |  0 |  0.00 | |

|ImageFile |  0|  0.00| |

|Hospital|  0|  0.00|  |

|Age|1 | 0.12| |

Sex,                    Fehlende Daten: 0,  0.00

Temp_C,                 Fehlende Daten: 154,  17.84

Cough,                  Fehlende Daten: 0,  0.00

DifficultyInBreathing,  Fehlende Daten: 4,  0.46

WBC,  Fehlende Daten: 9,  1.04

CRP,  Fehlende Daten: 33,  3.82

>Fibrinogen,  Fehlende Daten: 591,  68.48      <----------
>
LDH,  Fehlende Daten: 136,  15.76

>Ddimer,  Fehlende Daten: 621,  71.96      <----------

Ox_percentage,  Fehlende Daten: 243,  28.16

PaO2,  Fehlende Daten: 170,  19.70

>SaO2,  Fehlende Daten: 583,  67.56        <---------

pH,  Fehlende Daten: 207,  23.99       

CardiovascularDisease |19|  2.20 ||

|RespiratoryFailure|  | 159 | 18.42 ||

|Prognosis| 0| 0.00 ||


Eventuell eine Spalte einführen, die die Prognosis hervorhebt...
