# Regresja vs. Klasyfikacja
Przewidywanie wartości powinowactwa związków do danego receptora metodami uczenia maszynowego - przewidywanie konkretnych wartości Ki vs. podział na klasy aktywny/nieaktywny. Porównanie skuteczności różnych podejść.

## Podejście do problemu
W projekcie postanowiłyśmy porównać skuteczność różnych metod Regresji i Klasyfikacji w przewidywaniu wartości Ki oraz w klasyfikacji związków jako aktywny/nieaktywny. W tym celu zostały stworzone modele Regresji Liniowej, LASSO i Ridge oraz Regresji Logistycznej, Random Forest, SVC i Naive Bayes. Modele były trenowane na 9 zbiorach danych: dla każdego z 3 receptorów opioidowych, każdy z 3 fingerprintów.<br /> <br />
**Receptory opioidowe**:  <br />
-**µ**: Pobudzenie powoduje zniesienie bólu (łącznie 4939 związków) <br />
-**κ**: Pobudzenie powoduje zwężenie źrenic i sedację (łącznie 4628 związków) <br />
-**δ**: Pobudzenie powoduje zniesienie bólu (łącznie 4906 związków) <br />
<br />
**Fingerprinty:** <br />
-**Fingerprint Klekota**: sposób reprezentacji związku, w którym kodowane są ugrupowania, które często występują w substancjach aktywnych (4860 bitów) <br />
-**Fingerprint MACCS**: koduje informacje dotyczące struktury cząsteczki (166 bitów)<br />
-**Fingerprint Morgana**: fingerprint haszowany (1024 bity)<br />
<br /> 

## Uruchamianie
W celu uruchomienia programu wystarczy wywołać funkcję **datasets_handle()** w głównym pliku projektu **main**. Wyniki pisemne wypisywane są na ekran, natomiast heatmapy porównań między modelami zaopisywane są w folderze **results**.

## Wyniki
Wyniki zostały przedstawione na 3 sposoby:<br />
1. Porównanie wyników pomiędzy różnymi modelami regresji. W celu porównania zostały wyliczone **R^2** oraz **RMSE**.<br />
2. Porównanie wyników pomiędzy różnymi modelami klasyfikacji. W celu porównania zostały wyliczone **accuracy** oraz **F1**.<br />
3. Porównanie wyników pomiędzy regresją i klasyfikacją - po wytrenowaniu regresji zamiana wartości Ki ze zbioru testowego i zamiana przewidzianych wartości Ki na wartości binarne (punkt odcięcia wartości Ki = 100 nM), a następnie obliczenie **accuracy** oraz **F1** w celu miarodajnego porównania ich z wynikami modeli klasyfikacji.<br />
Każdy wynik zostaje wypisany oraz na jego podstawie stworzona została heatmap'a w celu wizualizacji wyników.

### Klasyfikacja
**Receptor Delta, metryka accuracy:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/delta_accuracy.png?raw=true)<br /><br />
**Receptor Delta, metryka F1:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/delta_f1.png?raw=true)<br /><br />
**Receptor Kappa, metryka accuracy:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/kappa_accuracy.png?raw=true)<br /><br />
**Receptor Kappa, metryka F1:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/kappa_f1.png?raw=true)<br /><br />
**Receptor Mu, metryka accuracy:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/mu_accuracy.png?raw=true)<br /><br />
**Receptor Mu, metryka F1:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/mu_f1.png?raw=true)<br /><br />

### Regresja
**Receptor Delta, metryka r^2:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/delta_r2.png?raw=true)<br /><br />
**Receptor Delta, metryka rmse:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/delta_rmse.png?raw=true)<br /><br />
**Receptor Kappa, metryka r^2:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/kappa_r2.png?raw=true)<br /><br />
**Receptor Kappa, metryka rmse:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/kappa_rmse.png?raw=true)<br /><br />
**Receptor Mu, metryka r^2:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/mu_r2.png?raw=true)<br /><br />
**Receptor Mu, metryka rmse:**<br />
![alt text](https://github.com/Seygaa/umwpl2021/blob/main/results/mu_rmse.png?raw=true)<br /><br />
