import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import requests
import pandas as pd
import json
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score



from datetime import datetime, timedelta
import pytz
import math
from sqlalchemy import create_engine


# Daten Einlesen und Anpassen 

# Zeitzone bestimmen
GERMAN_TZ = pytz.timezone('Europe/Berlin')

# Verbindungsparameter zur PostgreSQL-Datenbank
db_params = {
    'dbname': 'Wetterstation',
    'user': 'postgres',
    'password': 'Montag1618',
    'host': 'localhost',
    'port': '5432'
}

# Verbindung zur PostgreSQL-Datenbank herstellen
engine = create_engine(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["dbname"]}')

# Funktion zum Abrufen der Daten aus der Datenbank und Umwandeln in einen DataFrame
def fetch_data_from_db():
    query = """
    SELECT created_at AS timestamp, location_lat, location_lon, value, sensor_id
    FROM Wetterstation;
    """
    sense_df = pd.read_sql(query, engine)

    sensor_mapping = {
    '6645e1c8eb5aad0007b226b9': 'Luftdruck',
    '6645e1c8eb5aad0007b226b7': 'rel. Luftfeuchte',
    '6645e1c8eb5aad0007b226b6': 'Temperatur',
    '6645e1c8eb5aad0007b226b8': 'UV-Intensität'}

    grouped = sense_df[['timestamp', 'value', 'sensor_id']].groupby(['timestamp', 'sensor_id'], as_index=False).mean()
    pivoted = grouped.pivot(index='timestamp', columns='sensor_id', values='value')
    pivoted.rename(columns=sensor_mapping, inplace=True)
    data = pivoted.reset_index()
    return data.sort_values('timestamp')

# Daten abrufen und in sense_df speichern
sense_df = fetch_data_from_db()

# Einheiten der Messwerte
units = {
    'Luftdruck' : 'hPa',
    'rel. Luftfeuchte' : '%',
    'Temperatur' : '°C',
    'UV-Intensität' : 'µW/cm²'}


# Prediction mit SARIMA

# Funktion zum Vorbereiten der Daten auf Stunden aggregieren
def prepare_data(data):
    data.set_index('timestamp', inplace=True)
    data_hourly = data.resample('h').mean()
    return data_hourly

sensor1_data = prepare_data(sense_df[['timestamp','Luftdruck']].copy())
sensor2_data = prepare_data(sense_df[['timestamp','rel. Luftfeuchte']].copy())
sensor3_data = prepare_data(sense_df[['timestamp', 'Temperatur']].copy())
sensor4_data = prepare_data(sense_df[['timestamp', 'UV-Intensität']].copy())


# Funktion zum Trainieren des Modells und Vorhersagen machen
def make_predictions(data, forecast_steps=3*24):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    p, d, q = 2, 1, 2  # ARIMA-Komponenten
    P, D, Q, m = 1, 1, 1, 24  # Saisonale Komponenten 
    model = sm.tsa.SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
    fit = model.fit()
    # Vorhersage
    forecast = fit.get_forecast(steps= forecast_steps)
    forecast_index = pd.date_range(start=train.index[-1] + pd.DateOffset(hours=1), periods=forecast_steps, freq='h')
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

    # MSE
    mse_scores = mean_squared_error(test, forecast_series[:len(test)])
    
    # Bestimmtheitsmaß
    r2_scores = r2_score(test, forecast_series[:len(test)])

    return forecast_series.sort_index(), mse_scores, r2_scores

# Vorhersagen für die nächsten 3 Tage machen
forecasts = {}
forecasts['Luftdruck'] = make_predictions(sensor1_data)[0]
forecasts['rel. Luftfeuchte'] = make_predictions(sensor2_data)[0]
forecasts['Temperatur'] = make_predictions(sensor3_data)[0]
forecasts['UV-Intensität'] = make_predictions(sensor4_data)[0]

mse = {}
mse['Luftdruck'] = make_predictions(sensor1_data)[1]
mse['rel. Luftfeuchte'] = make_predictions(sensor2_data)[1]
mse['Temperatur'] = make_predictions(sensor3_data)[1]
mse['UV-Intensität'] = make_predictions(sensor4_data)[1]

r2 = {}
r2['Luftdruck'] = make_predictions(sensor1_data)[2]
r2['rel. Luftfeuchte'] = make_predictions(sensor2_data)[2]
r2['Temperatur'] = make_predictions(sensor3_data)[2]
r2['UV-Intensität'] = make_predictions(sensor4_data)[2]


# Funktionen 

# Funktion zur Bestimmung der Farbe des Messwerts    
def get_color_based_on_measurement(measurement):
    if measurement == 'Temperatur':
        return 'rgb(219, 15, 53)'
    elif measurement == 'Luftdruck':
        return 'rgb(148, 6, 191)'
    elif measurement == 'rel. Luftfeuchte':
        return 'rgb(17, 119, 214)'
    else:
        return 'yellow'

# Funktion zur Bestimmung der Einheit des Messwertes
def get_unit(measurement):
    return data_df.loc[data_df.measurement == measurement, 'unit'].unique().item()

# Funktion zur Berechnung des UV-Index
def calculate_uv_index(uv_power):
    ''' 
    Berechnet den Uv-Index
    '''
    uv_index = float(uv_power) / 25

    # UV-Index-Klasse
    if uv_index <= 2:
        uv_class = "Niedrig"
    elif uv_index <= 5:
        uv_class = "Mäßig"
    elif uv_index <= 7:
        uv_class = "Hoch"
    elif uv_index <= 10:
        uv_class = "Sehr hoch"
    else:
        uv_class = "Extrem"
    
    return uv_index, uv_class

# Funktion zur Bestimmung des Heat Index
def calculate_heat_index(temperature, humidity):
    """
    Berechnet den Heat Index (gefühlte Temperatur) basierend auf der Lufttemperatur und der relativen Luftfeuchtigkeit.
    (Formel: https://rechneronline.de/barometer/hitzeindex.php#google_vignette)

    """
    T = temperature
    F = humidity
    HI = -8.784695 + 1.61139411*T + 2.338549*F - 0.14611605*T*F - 0.012308094*(T**2) - 0.016424828*(F**2) + 0.002211732*(T**2)*F + 0.00072546*T*(F**2) - 0.000003582*(T**2)*(F**2)
    return HI

# Funktion zur Bestimmung der Differenz von aktuellem und durchschnitts Wert des Luftdrucks
def calculate_pressure_difference(pressure):
    '''
    Berechnet den Unterschied zwischen dem durchschnittlichen Luftdruck in Karlsruhe 1000 hPa und dem aktuellen Wert.
    (Durchschnittswert: https://web1.karlsruhe.de/codeIgniter/wetter/jahreswerte)
    '''
    pressure_difference = pressure - 1000
    pressure_difference_percentage = (pressure_difference / 1000) * 100

    if pressure_difference >= 0:
        direction = 'über'
    else:
        direction = 'unter'

    return pressure_difference, pressure_difference_percentage, direction

# Funktion zur Bestimmung des Taupunktes
def calculate_taupunkt(humidity, temperatur):
    '''
    Berechnet den Taupunkt aus Temperatur und Luftfeuchtigkeit.
    (Formel: https://loxwiki.atlassian.net/wiki/spaces/LOX/pages/1518403585/Taupunkt+berechnen)
    '''
    t = temperatur
    h = humidity
    tp = 243.12*((17.62*t)/(243.12+t)+math.log(h/100))/((17.62*243.12)/(243.12+t)-math.log(h/100))
    if tp >= 17:
        schwuel = (f" ,es ist schwül.")
    else:
        schwuel = ""
    return tp, schwuel


# Dash-App erstellen  

# API zu unsere Sensen-Box
API_URL = "https://api.opensensemap.org/boxes/6645e1c8eb5aad0007b226b5?format=json"

# DataFrame für dei Live Daten
data_df = pd.DataFrame(columns=['timestamp', 'measurement', 'value', 'unit'])

# Dark-mode
dark_mode_styles = {
    'backgroundColor': 'rgb(17, 17, 17)',
    'color': 'white'
}

app = dash.Dash(__name__)

# Layout vom Dash-Board 
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Live Daten', value='tab-1', style={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'}, selected_style=dark_mode_styles),
        dcc.Tab(label='Langzeit Daten', value='tab-2', style={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'}, selected_style=dark_mode_styles),
        dcc.Tab(label='Vorhersage', value='tab-3', style={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'}, selected_style=dark_mode_styles),
    ]),
    html.Div(id='tabs-content'),  
    html.Button('Stop Fetching Data', id='stop-button', n_clicks=0),
    dcc.Interval(id='interval-component', interval=60000, n_intervals=0)
],style=dark_mode_styles)

# Live Daten 

fetching = True

# Abfragen der Live-Daten
def fetch_live_data():
    global data_df
    response = requests.get(API_URL)
    if response.status_code == 200:
        data = response.json()
        measurements = data['sensors']
        live_data = {'timestamp': [], 'measurement': [], 'value': [], 'unit': []}
        for measurement in measurements:
            live_data['timestamp'].append(datetime.strptime(measurement['lastMeasurement']['createdAt'], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(GERMAN_TZ))
            live_data['measurement'].append(measurement['title'])
            live_data['value'].append(measurement['lastMeasurement']['value'])
            live_data['unit'].append(measurement['unit'])
        data_df = pd.concat([data_df, pd.DataFrame(live_data)], ignore_index=True)
        data_df = pd.concat([data_df, pd.DataFrame(live_data)], ignore_index=True)
        return data_df
    else:
        return pd.DataFrame()
    
# Updaten der Live-Daten
@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_live_data(n):
    live_data = fetch_live_data()
    if not live_data.empty:
        graphs = []
        for measurement in live_data['measurement'].unique():

            # Daten extrahierne
            data = live_data[live_data['measurement'] == measurement]
            data['value'] = pd.to_numeric(data['value'])
            data.groupby(['timestamp','measurement', 'unit'], as_index=False).mean()

            # Farbe bestimmen
            color = get_color_based_on_measurement(measurement)

            # Kennzahl berechnen
            text = ""
            if measurement == 'UV-Intensität':
                uv_index, uv_class = calculate_uv_index(data['value'].iloc[-1])
                text = (f"aktueller UV-Index: {uv_index:.2f}, Klasse: {uv_class}")
            elif measurement == 'Temperatur':
                heat_index = calculate_heat_index(float(data['value'].iloc[-1]), float(live_data.loc[live_data.measurement == 'rel. Luftfeuchte', 'value'].iloc[-1]))
                text = (f"aktuelle gefühlte Temperatur: {round(heat_index, 1)} {get_unit(measurement)}")
            elif measurement == 'Luftdruck':
                pressure_difference, pressure_difference_percentage, direction = calculate_pressure_difference(float(data['value'].iloc[-1]))
                text = (f"aktuell ist der Luftdruck {round(pressure_difference,1)} {get_unit(measurement)} ({round(pressure_difference_percentage,2)}%) {direction} dem Durchschnitt")
            elif measurement == 'rel. Luftfeuchte':
                tp, schwuel = calculate_taupunkt(float(data['value'].iloc[-1]), float(live_data.loc[live_data.measurement == 'Temperatur', 'value'].iloc[-1]))
                text = (f"Taupunkt bei {round(tp,2)} °C{schwuel}")

            # Graph für die Life-Daten erstellen
            graph = dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=data['timestamp'],
                            y=data['value'].sort_values(),
                            mode='lines+markers',
                            marker=dict(color=color),
                            line=dict(color=color)
                        )
                    ],
                    layout=go.Layout(
                        title=f'Live Daten: {measurement}',
                        xaxis=dict(title='Zeit'),
                        yaxis=dict(title=f'{measurement} ({data["unit"].iloc[0]})'),
                        template='plotly_dark'
                    )))
            
            # Kennzahlen anzeigen lassen 
            kpi = html.Div([
                html.P(text)],
                 style={'color': 'black', 'backgroundColor': color, 'padding': '0.1px', 'borderRadius': '100px', 'textAlign': 'center', 'font-family': 'Roboto, sans-serif', 'width': '550px', 'margin': 'auto'})

            # Graph und Kennzahl zusammenfügen
            graphs.append(html.Div([graph, kpi]))
        live_update_text = html.Div(children=graphs)

    # Ausgabe bei Fehler
    else:
        live_update_text = html.Div("Fehler beim Abrufen der Live-Daten")
    
    return live_update_text
        
    
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(id='live-update-text')
    
    # Langzeit-Daten
    
    elif tab == 'tab-2':
        graphs = []
        for measurement in ['Luftdruck', 'rel. Luftfeuchte', 'Temperatur', 'UV-Intensität']:

            # Farbe bestimmen
            color = get_color_based_on_measurement(measurement)

            # Daten extrahieren / anpassen
            data = sense_df[[measurement, 'timestamp']]
            data['timestamp'] = data['timestamp'].dt.tz_localize(GERMAN_TZ, ambiguous='NaT', nonexistent='shift_forward')

            # Kennzahlen berechnen
            mean_value = data.loc[data.timestamp > (data.timestamp.max() - timedelta(days=1)), measurement].mean()
            max_value = data.loc[data.timestamp > (data.timestamp.max() - timedelta(days=1)), measurement].max()
            min_value = data.loc[data.timestamp > (data.timestamp.max() - timedelta(days=1)), measurement].min()

            # Graph erstellen 
            graph = dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=sense_df['timestamp'],
                            y=sense_df[measurement],
                            mode='lines+markers',
                            marker= dict(color=color),
                            line= dict(color=color)
                        )
                    ],
                    layout=go.Layout(
                        title=f'Langzeit-Daten: {measurement}',
                        xaxis=dict(title='Datum'),   
                        yaxis=dict(title=f"{measurement} ({get_unit(measurement)})"),
                        template='plotly_dark'       
                    )
                ), style={'flex': '3'}
            )

            # Kennzahlen anzeigen lassen 
            kpi = html.Div([
                    html.H4('Kennzahlen'),
                    html.P(f"Durchschnittlich der letzten 24h {measurement}: {mean_value:.2f} {get_unit(measurement)}"),
                    html.P(f"Maximum der letzten 24h {measurement}: {max_value:.2f} {get_unit(measurement)}"),
                    html.P(f"Minimum der letzten 24h {measurement}: {min_value:.2f} {get_unit(measurement)}")],
                 style={'color': color, 'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px', 'borderRadius': '1px', 'textAlign': 'center', 'font-family': 'Roboto, sans-serif'})
            
            # Graph und Kennzahl zusammenfügen
            graphs.append(html.Div([graph, kpi], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'margin-bottom': '20px'}))
        
        return html.Div(children=graphs)

            # Prediction

    elif tab == 'tab-3':

        # Graph erstellen 
        graphs = []
        for measurement  in ['Luftdruck', 'rel. Luftfeuchte', 'Temperatur', 'UV-Intensität']:
            color = get_color_based_on_measurement(measurement)
            forecast = forecasts[measurement]
            prepare = prepare_data(sense_df[['timestamp',measurement]].copy())
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=prepare.index, y=prepare.values.flatten(), mode='lines+markers', name=f'{measurement} {get_unit(measurement)} (Daten)', line=dict(color=color)))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name=f'{measurement} {get_unit(measurement)} (Vorhersage)', line=dict(color='white')))
            fig.update_layout(title=f'{measurement} (Daten und Vorhersage für die nächsten 2 Tage)', xaxis=dict(title='Datum'), yaxis=dict(title=f"{measurement} ({get_unit(measurement)})"), template='plotly_dark')
            graphs.append(dcc.Graph(figure= fig))

            fehlermaß = html.Div([
                    html.H4('Fehlermaß'),
                    html.P(f"MSE der {measurement}: {mse[measurement].round(2)}"),
                    html.P(f"R^2 der {measurement}: {r2[measurement].round(2)}")],
                 style={'color': color, 'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px', 'borderRadius': '1px', 'textAlign': 'center', 'font-family': 'Roboto, sans-serif'})
            graphs.append(fehlermaß)
        return html.Div(children=graphs)
    
# Funktion zum stoppen des Fetchings    
@app.callback(
    Output('stop-button', 'children'),
    [Input('stop-button', 'n_clicks')]
)
def stop_fetching(n_clicks):
    global fetching
    if n_clicks > 0:
        fetching = False
        return 'Data Fetching Stopped'
    return 'Stop Fetching Data'

if __name__ == '__main__':
    app.run_server(debug=True, port=8056)

# CTRL + C zum stoppen