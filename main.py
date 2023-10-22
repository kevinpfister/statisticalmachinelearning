# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm

def linearregression (X,Y):
    X_test = injectionMolding_test[
        ["Inj1PosVolAct_Var", "Inj1PrsAct_meanOfInjPhase", "Inj1HtgEd3Act_1stPCscore", "ClpFceAct_1stPCscore",
         "ClpPosAct_1stPCscore"]]
    y_test = injectionMolding_test["mass"]
    print("hello")
    # Initialisieren des linearen Regressionsmodells
    model = LinearRegression()

    # Trainieren des Modells
    model.fit(X, Y)

    # Ausgabe der Koeffizienten und des Schnitts (Intercepts)
    print("Koeffizienten:", model.coef_)
    print("Schnitt (Intercept):", model.intercept_)

    # Vorhersage auf dem gesamten Datensatz
    y_pred = model.predict(X)

    # Ausgabe des R-quadrat-Wert
    r_squared = model.score(X, Y)
    print("R-quadrat-Wert:", r_squared)
    plt.scatter(Y, y_pred)
    plt.xlabel("Tatsächliche Werte")
    plt.ylabel("Vorhergesagte Werte")
    plt.title("Lineare Regression: Tatsächliche vs. Vorhergesagte Werte")
    plt.show()

    y_pred_test = model.predict(X_test)

    # Berechne den Mean Squared Error (MSE) für Trainings- und Testdaten
    mse_train = np.mean((Y - y_pred) ** 2)
    mse_test = np.mean((y_test - y_pred_test) ** 2)

    print("Trainings-MSE:", mse_train)
    print("Test-MSE:", mse_test)


def calculate_p_r_value(features, Y):

    results = []

    for feature in features:
        X = injectionMolding[feature]
        X = sm.add_constant(X)
        model = sm.OLS(injectionMolding[Y], X).fit()  # Y ist die abhängige Variable
        r_squared = model.rsquared
        p_value = model.pvalues[1]  # Der p-Wert für die unabhängige Variable

        results.append({'Variable': feature, 'R_squared': r_squared, 'p_value': p_value})

    # Erstellen einer DataFrame
    results_df = pd.DataFrame(results)

    # Ausgabe der Tabelle
    print(results_df)



if __name__ == '__main__':
    matplotlib.use('TkAgg')
    injectionMolding = pd.read_csv("InjectionMolding_Train.csv", usecols=["PowTotAct_Min","Inj1PosVolAct_Var","Inj1PrsAct_meanOfInjPhase","Inj1HopTmpAct_1stPCscore",
                                                  "Inj1HtgEd3Act_1stPCscore","ClpFceAct_1stPCscore","ClpPosAct_1stPCscore","OilTmp1Act_1stPCscore","mass"])
    injectionMolding_test = pd.read_csv("InjectionMolding_Test.csv",
                                   usecols=["PowTotAct_Min", "Inj1PosVolAct_Var", "Inj1PrsAct_meanOfInjPhase",
                                            "Inj1HopTmpAct_1stPCscore",
                                            "Inj1HtgEd3Act_1stPCscore", "ClpFceAct_1stPCscore", "ClpPosAct_1stPCscore",
                                            "OilTmp1Act_1stPCscore", "mass"])
    correlation_matrix = injectionMolding.corr()
    column_names = injectionMolding.columns.tolist()

    calculate_p_r_value(column_names,"mass")



df = pd.DataFrame(correlation_matrix)

# Erstellen des Heatmap-Plots
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korrelationsmatrix")

plt.show()



# Auswählen der unabhängigen Variablen (Features) und der abhängigen Variable (Ziel)
X = injectionMolding[
    ["Inj1PosVolAct_Var", "Inj1PrsAct_meanOfInjPhase", "Inj1HtgEd3Act_1stPCscore", "ClpFceAct_1stPCscore",
     "ClpPosAct_1stPCscore"]]
Y = injectionMolding["mass"]

X_test = injectionMolding_test[
    ["Inj1PosVolAct_Var", "Inj1PrsAct_meanOfInjPhase", "Inj1HtgEd3Act_1stPCscore", "ClpFceAct_1stPCscore",
     "ClpPosAct_1stPCscore"]]
Y_test = injectionMolding_test["mass"]

linearregression(X,Y)






