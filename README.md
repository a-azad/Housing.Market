
# INTRODUCTION

## About
This work is a personal study on Seattle's housing market and it is not for commercial use. Unfortunately, the data is limited to 2014-2015 and extension of the work to today's [crazy] market is not practical. Data exploration and ML modeling is captured in multiple "jupyter notebooks" in this repository. The problem is initially a regression modeling exercise. Simple linear model and tree-based algorithms was used. Although Random Forest (like always) offered a significant improvement to the model, for the purpose of statistical inference, improved (Ridge) linear model was chosen for this study.

**Acknowledgement:** The original dataset belongs to Kaggle: [House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction) provided to redict house price using regression.

## Data


```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
data = pd.read_csv("clean_data.csv")
```


```python
data.drop(['Unnamed: 0'], axis=1, inplace=True)
```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>...</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.309982</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>8.639411</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>8.639411</td>
      <td>2014</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.195614</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>8.887653</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>400</td>
      <td>1951</td>
      <td>1</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>8.941022</td>
      <td>2014</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.100712</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>9.210340</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8.994917</td>
      <td>2015</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.311329</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>8.517193</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>...</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>8.517193</td>
      <td>2014</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.142166</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8.997147</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>8.923058</td>
      <td>2015</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
import plotly
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='----', api_key='----')
mapbox_access_token = '-----'

data_ = [
    go.Scattermapbox(
        lat=list(data.lat),
        lon=list(data.long),
        mode='markers',
        marker=dict(size=2),
        text=[''],
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=47.5,
            lon=-122
        ),
        pitch=0,
        zoom=5
    ),
)

fig = dict(data=data_, layout=layout);
plotly.plotly.iplot(fig)
```

<img src="pix\map.JPG">


###  Fin!
