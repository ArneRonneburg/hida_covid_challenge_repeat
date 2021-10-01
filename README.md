# hida_covid_challenge_repeat
 repeating the HIDA covid datathon challenge, let's see how far we can get....the paper gets an accuracy of 78 %(?), can we get better?
During the datathon, an incomplete set of clinical data of covid-19 patients was provided, together with chest x-ray images. Based on these data, the severity of the course of the disease (mild vs severe) shall be predicted. Therefore, two tasks have to be accomplished

1) complete the missing values of the clinical data. 
2) Use the completed data and the X-ray images to predict the severity of the disease



Paper:
 https://statisticalhorizons.com/wp-content/uploads/MissingDataByML.pdf

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

![Overview different](https://user-images.githubusercontent.com/57065083/128631716-1c5c8bbc-c76a-412e-a65e-8c44b2773eea.png)

Unterschiedliche verteilung 

Fibogen
DDimer
SaO2

Fehlen deutlich mehr Daten als bei dem Rest 
(underfitting potential)!!!!

Krankenhausdaten prüfen

|Hospital| Anzahl|
|--------|-------|
|0      |104 |
|1     |31 |
|2    |139 | 
|3    |101 |
|4    |488 |

Heatmap sortiert nach Krankenhäusern

![Matrixübersicht_OhneBild_sort_Hosp](https://user-images.githubusercontent.com/57065083/128598568-4c275ed1-f2aa-449d-ab19-7cb5bff85e24.png)

| Vollständige Datensäzte | 6 |
|------------|-----|

# Ausreißer ?

![Ausreißer](https://user-images.githubusercontent.com/57065083/128627025-4d0df93d-76e5-4e08-81db-7ed5a42dc5ae.png)

interessant ist hier, dass sich die unterschiedlichen Werte in "Gruppen" anordnen. -> Daher funktioniert KNN so gut. 
Allerdings fällt auch auf, dass es einige Ausreißer gibt, die wahrscheinlch nicht von den KNN einbezogen werden. (z.B. WBC streut sehr)
Auch sieht man, dass bei den Daten, wo viele Fehlstellen sind, der Fit wahrshcienlich scheiße ist (DDemer) 

# Daten Korrelation

![Correlation](https://user-images.githubusercontent.com/57065083/128599579-4ec158d6-8f3b-4fe1-9033-a6335474597a.png)

# Idee
1. Daten splitten in viele fehlende Daten und wenig fehlende Daten
2. imputer auf wenig fehlende Daten anwenden
3. imputer mit den eingesetzten Daten auf die vielen fehlenden Daten einsetzen
4. lossfunktionen abgleichen

Ich vermute, das unterschiedliche Imputer bei unterschiedlichen Anzahl fehlender Daten besser Funktionieren. 
 Sprich Imputer a ist eher für wenig fehlende Daten
        imputer b ist eher für viele fehlende Daten
        
Aber rein logisch, würde ich sagen, dass je mehr Daten ein Imputer zur Verfügung hat, desto besser und genauer werden die Predictions

