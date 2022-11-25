from sklearn import preprocessing
import pandas as pd

def preprocess(trainDF, testDF, previous_hours=None):
  """ Take train and test datasets at the same time
  """

  # Remove row if consumption is NaN
  # (Only for training data)
  trainDF = trainDF.dropna(subset=['consumption'])

  
  # scaler for consumption
  normScaler = preprocessing.StandardScaler()
  normScaler.fit(trainDF[['consumption']])
  trainDF['consumption'] = normScaler.transform(trainDF[['consumption']])

  # Add n previous hour features
  if previous_hours:
    for i in range(1, previous_hours+1):
      trainDF['{}cons'.format(i)] = trainDF.consumption.shift(i)
    
    # Cut out first n rows with NaNs
    trainDF = trainDF[previous_hours:]

  # All remaning NaNs replaced with 0
  trainDF = trainDF.fillna(0)
  testDF = testDF.fillna(0)

  # Get weekday, month and hour of day
  trainDF['time'] = pd.to_datetime(trainDF.time, utc=True)
  trainDF['time'] = trainDF['time'].dt.tz_convert('Europe/Tallinn')
  trainDF['day'] = trainDF['time'].dt.weekday
  trainDF['month'] = trainDF['time'].dt.month
  trainDF['hour'] = trainDF['time'].dt.hour

  testDF['time'] = pd.to_datetime(testDF.time, utc=True)
  testDF['time'] = testDF['time'].dt.tz_convert('Europe/Tallinn')
  testDF['day'] = testDF['time'].dt.weekday
  testDF['month'] = testDF['time'].dt.month
  testDF['hour'] = testDF['time'].dt.hour

  # Normalization
  colList = ['temp', 'dwpt','rhum', 'wdir', 'wspd', 'wpgt', 'pres', 'el_price']

  for col in colList:
    scaler = preprocessing.MinMaxScaler()
    normCol = scaler.fit_transform(trainDF[[col]])
    trainDF[col] = normCol
    testDF[col] = scaler.transform(testDF[[col]])

  # One hot encoding days
  onehot_days = pd.get_dummies(trainDF['day'])
  onehot_days = onehot_days.set_axis(['Monday', 'Tuesday', 'Wendesday', 
                                      'Thursday', 'Friday', 'Saturday', 'Sunday'], axis=1)
  trainDF = trainDF.join(onehot_days)

  onehot_days = pd.get_dummies(testDF['day'])
  onehot_days = onehot_days.set_axis(['Monday', 'Tuesday', 'Wendesday', 
                                      'Thursday', 'Friday', 'Saturday', 'Sunday'], axis=1)
  testDF = testDF.join(onehot_days)
  
  # One hot encoding days
  onehot_hours = pd.get_dummies(trainDF['hour'])
  hour_cols = ["{}hour".format(i) for i in range(24)]
  onehot_hours = onehot_hours.set_axis(hour_cols, axis=1)
  trainDF = trainDF.join(onehot_hours)

  onehot_hours = pd.get_dummies(testDF['hour'])
  onehot_hours = onehot_hours.set_axis(hour_cols, axis=1)
  testDF = testDF.join(onehot_hours)


  #TODO: Onehot months
  #TODO: Onehot coco

  # If one-hot vecencodingtor is done remove:
  scaler = preprocessing.MinMaxScaler()
  normCol = scaler.fit_transform(trainDF[['month']])
  trainDF['month'] = normCol
  testDF['month'] = scaler.transform(testDF[['month']])

  # Sin and cos of hours and days
  trainDF['sin_hour'] = np.sin(2*np.pi*(trainDF['hour'])/24)
  trainDF['cos_hour'] = np.cos(2*np.pi*(trainDF['hour'])/24)   
  trainDF['sin_day'] = np.sin(2*np.pi*(trainDF['day'])/7)
  trainDF['cos_day'] = np.cos(2*np.pi*(trainDF['day'])/7) 

  testDF['sin_hour'] = np.sin(2*np.pi*(testDF['hour'])/24)
  testDF['cos_hour'] = np.cos(2*np.pi*(testDF['hour'])/24)   
  testDF['sin_day'] = np.sin(2*np.pi*(testDF['day'])/7)
  testDF['cos_day'] = np.cos(2*np.pi*(testDF['day'])/7) 


  trainDF = trainDF.drop(columns=['day', 'hour', 'coco'])
  testDF = testDF.drop(columns=['day', 'hour', 'coco'])
  # And add:
  #processedDataFrame = dataFrame.drop(columns=['day', 'month', 'hour']

  return trainDF, testDF, normScaler
