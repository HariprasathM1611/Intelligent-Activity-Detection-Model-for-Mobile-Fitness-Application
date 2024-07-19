import joblib
import warnings
warnings.filterwarnings('ignore')


def predict_activity_using_pickled(test_data):
  model = joblib.load("rfm_model.pkl")
  prediction = model.predict(test_data)
  print("Activity Predicted by Random Forest Classifier Model :",prediction[0])
  model = joblib.load("dt_model.pkl")
  prediction = model.predict(test_data)
  print("Activity Predicted by Decision Tree Classifier Model :",prediction[0])
  model = joblib.load("knn_model.pkl")
  prediction = model.predict(test_data)
  print("Activity Predicted by K-Nearest Neighbors Model :",prediction[0])

choice="Y"
while choice=="Y" or choice=="y" :
  test_data=[]
  data=[]
  data.append(float(input("Enter the X-value of orientation sensor: ")))
  data.append(float(input("Enter the Y-value of orientation sensor: ")))
  data.append(float(input("Enter the Z-value of orientation sensor: ")))
  data.append(float(input("Enter the X-value of rotation sensor: ")))
  data.append(float(input("Enter the Y-value of rotation sensor: ")))
  data.append(float(input("Enter the Z-value of rotation sensor: ")))
  data.append(float(input("Enter the X-value of accelerometer sensor: ")))
  data.append(float(input("Enter the Y-value of accelerometer sensor: ")))
  data.append(float(input("Enter the Z-value of accelerometer sensor: ")))
  data.append(float(input("Enter the X-value of gyro sensor: ")))
  data.append(float(input("Enter the Y-value of gyro sensor: ")))
  data.append(float(input("Enter the Z-value of gyro sensor: ")))
  data.append(float(input("Enter the X-value of magnetic sensor: ")))
  data.append(float(input("Enter the Y-value of magnetic sensor: ")))
  data.append(float(input("Enter the Z-value of magnetic sensor: ")))
  data.append(float(input("Enter the value of Light sensor: ")))
  data.append(float(input("Enter the value of Sound Level: ")))
  test_data.append(data)
  predict_activity_using_pickled(test_data)
  choice=input("Do you want try again?(Y/N):")
print("Thank You")