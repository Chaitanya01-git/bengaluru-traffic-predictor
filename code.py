import random 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib


areas = ["Whitefield","Electronic City","Marathahalli","Silk Board","Koramangala","MG Road","Indiranagar","Hebbal","Yelahanka","Jayanagar"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weather = ["Sunny", "Rainy", "Cloudy"]
hours = list(range(24))
busy_area = ["Whitefield","Electronic City","Marathahalli","Silk Board","MG Road"]
data = []
for i in range(1000):
    area = random.choice(areas)
    hour =  random.choice(hours)
    day = random.choice(days)
    weather_type = random.choice(weather)
    if area in busy_area and (7 <= hour <= 10 or 17 <= hour <=22):
        traffic ="High"
    elif((day == "Sunday" and 12 <= hour <=16) or (0 <= hour <= 5)):
        traffic ="Low"
    else:
        traffic ="Medium"

    data.append([area, hour, day, weather_type,traffic])

        
    
df = pd.DataFrame(data, columns =["Area", "Hour", "Day", "Weather", "Traffic"])

df.to_csv("bengaluru_traffic.csv", index = False)



df_encoded = pd.get_dummies(df, columns = ["Area", "Day", "Weather"])
print(df_encoded.head())
print(df_encoded.shape)
print(df_encoded.columns)

X = df_encoded.drop("Traffic", axis = 1)
y = df_encoded["Traffic"]




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)




model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model Trained Sucessfully")
y_pred = model.predict(X_test)


print(y_pred[:10])
print(y_test.head(10))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)

joblib.dump(model, "traffic_model.pkl")