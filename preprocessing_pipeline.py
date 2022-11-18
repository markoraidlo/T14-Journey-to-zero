from sklearn import preprocessing
import pandas as pd

def preprocess(dataFrame, previous_hours=None):
  """
  """

  # Remove row if consumption is NaN
  # (Only for training data)
  if 'consumption' in dataFrame:
    dataFrame = dataFrame.dropna(subset=['consumption'])

  # Add n previous hour features
  if previous_hours:
    for i in range(1, previous_hours+1):
      dataFrame['{}cons'.format(i)] = dataFrame.consumption.shift(i)
    
    # Cut out first n rows with NaNs
    dataFrame = dataFrame[previous_hours:]

  # All remaning NaNs replaced with 0
  dataFrame = dataFrame.fillna(0)

  # Get weekday, month and hour of day
  dataFrame['time'] = pd.to_datetime(dataFrame.time, utc=True)
  dataFrame['time'] = dataFrame['time'].dt.tz_convert('Europe/Tallinn')
  dataFrame['day'] = dataFrame['time'].dt.weekday
  dataFrame['month'] = dataFrame['time'].dt.month
  dataFrame['hour'] = dataFrame['time'].dt.hour

  # Normalization
  colList = ['temp', 'dwpt','rhum', 'wdir', 'wspd', 'wpgt', 'pres', 'el_price']

  for col in colList:
    scaler = preprocessing.MinMaxScaler()
    normCol = scaler.fit_transform(dataFrame[[col]])
    dataFrame[col] = normCol

  # One hot encoding days
  onehot_days = pd.get_dummies(dataFrame['day'])
  onehot_days = onehot_days.set_axis(['Monday', 'Tuesday', 'Wendesday', 
                                      'Thursday', 'Friday', 'Saturday', 'Sunday'], axis=1)
  dataFrame = dataFrame.join(onehot_days)
  
  # One hot encoding days
  onehot_hours = pd.get_dummies(dataFrame['hour'])
  hour_cols = ["{}hour".format(i) for i in range(24)]
  onehot_hours = onehot_hours.set_axis(hour_cols, axis=1)
  dataFrame = dataFrame.join(onehot_hours)


  #TODO: Onehot months
  #TODO: Onehot coco

  # If one-hot vecencodingtor is done remove:
  scaler = preprocessing.MinMaxScaler()
  normCol = scaler.fit_transform(dataFrame[['month']])
  dataFrame[col] = normCol

  processedDataFrame = dataFrame.drop(columns=['day', 'hour', 'coco'])
  # And add:
  #processedDataFrame = dataFrame.drop(columns=['day', 'month', 'hour'])
  
  return processedDataFrame