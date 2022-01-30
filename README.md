# Regresja vs. Klasyfikacja
Przewidywanie wartości powinowactwa związków do danego receptora metodami uczenia maszynowego - przewidywanie konkretnych wartości Ki vs. podział na klasy aktywny/nieaktywny. Porównanie skuteczności różnych podejść.

## Podejście do problemu
W projekcie postanowiłyśmy porównać skuteczność różnych metod Regresji i Klasyfikacji w przewidywaniu wartości Ki. W tym celu zostały stworzone modele Regresji Liniowej, Lasso i Ridge'a oraz Regresji Logistycznej, Random Forest, SVC i Naive Bayes. Modele były trenowane na 9 zbiorach danych: dla każdego z 3 receptorów opioidowych, każdy z 3 fingerprintów.<br /> <br />
**Opioidy**:  <br />
-**µ**: Pobudzenie powoduje zniesienie bólu (łącznie 4939 związków) <br />
-**κ**: Pobudzenie powoduje zwężenie źrenic i sedację (łącznie 4628 związków) <br />
-**δ**: Pobudzenie powoduje zniesienie bólu (łąćznie 4906 związków) <br />
<br />
**Fingerprinty:** <br />
-**Fingerprint Klekota**: sposób reprezentacji związku, w którym kodowane są ugrupowania, które często występują w substancjach aktywnych (4860 bitów) <br />
-**Fingerprint MACCS**: koduje informacje dotyczące struktury cząsteczki (166 bitów)<br />
-**Fingerprint Morgana**: fingerprint haszowany (1024 bity)<br />
<br /> 

## Uruchamianie
W celu uruchomienia programu wystarczy wywołać funkcję **datasets_handle()** w głównym pliku projektu **main**.

## Wyniki
Wyniki zostały przedstawione na 3 sposoby:<br />
1. Porównanie wyników pomiędzy różnymi modelami regresji<br />
2. Porównanie wyników pomiędzy różnymi modelami klasyfikacji<br />
3. Wprowadzenie zaokrąglenia wyników modeli regresji (punkt odcięcia wartości Ki = 100) w celu miarodajnego porównania ich z wynikami modeli klasyfikacji<br />
Dla modeli regjresji w celu porównania zostały wyliczone **R^2** oraz **RMSE**. Natomiast dla modeli klasyfikacji **acuuracy** oraz **F1**. Każdy wynik zostaje wypisany oraz na jego podstawie stworzona została heatmap'a w celu wizualizacji wyników.

**Opioid Delta, metryka accuracy:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/delta_accuracy.png?raw=true)<br /><br />
**Opioid Delta, metryka F1:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/delta_f1.png?raw=true)<br /><br />
**Opioid Kappa, metryka accuracy:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/kappa_accuracy.png?raw=true)<br /><br />
**Opioid Kappa, metryka F1:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/kappa_f1.png?raw=true)<br /><br />
**Opioid Mu, metryka accuracy:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/mu_accuracy.png?raw=true)<br /><br />
**Opioid Mu, metryka F1:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/mu_f1.png?raw=true)<br /><br />
