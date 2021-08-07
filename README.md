# hida_covid_challenge_repeat
 repeating the HIDA covid datathon challenge, let's see how far I can get

Geeigneten Imputer finden


# Übersicht über die Daten 

(Gelb = Fehlt )
![Matrixübersicht_OhneBild](https://user-images.githubusercontent.com/57065083/128595496-d45512fa-8a1e-4951-8b9d-32d6b3270e90.png)



| Column   |  Fehlende Daten| Prozent |  Markant |
| ---------| ---------------|---------|----------|
|PatientID|0|0.00| |
|ImageFile|0|0.00| |
|Hospital|0|0.00| |
|Age|1|0.12| |
|Sex|0|0.00| |
|Temp_C|154|17.84| |
|Cough|0|0.00| |
|DifficultyInBreathing|4|0.46| |
|WBC|9|1.04| |
|CRP|33|3.82| |
|Fibrinogen|591|68.48| X|
|LDH|136|15.76| |
|Ddimer|621|71.96| X|
|Ox_percentage|243|28.16| |
|PaO2|170|19.70| |
|SaO2|583|67.56|X |
|pH|207|23.99| |
|CardiovascularDisease|19|2.20| |
|RespiratoryFailure|159|18.42| |
|Prognosis|0|0.00| |

Unterschiedliche verteilung 

Fibogen
DDimer
SaO2
Fehlen deutlich mehr Daten als bei dem Rest 

(underfitting potential)

Krankenhausdaten prüfen

|Hospital| Anzahl|
|--------|-------|
|0      |104 |
|1     |31 |
|2    |139 | 
|3    |101 |
|4    |488 |

Heatmap sortiert nach Krankenhäusern

[Matrixübersicht_OhneBild_sort_Hosp](https://user-images.githubusercontent.com/57065083/128598568-4c275ed1-f2aa-449d-ab19-7cb5bff85e24.png)

| Vollständige Datensäzte | 6 |
|------------|-----|
