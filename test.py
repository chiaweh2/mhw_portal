import dash
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

# Sample data with more points
data = {
    'lon': [-74.006, -118.2437, -0.1276, 139.6917, 151.2093, 2.3522, -3.7038, 37.6173, 144.9631, -46.6333],
    'lat': [40.7128, 34.0522, 51.5074, 35.6895, -33.8688, 48.8566, 40.4168, 55.7558, -37.8136, -23.5505],
    'SST anomaly(C)': [1.2, 2.3, 0.5, 1.8, 2.1, 0.9, 1.5, 2.0, 1.3, 2.4]
}
df = pd.DataFrame(data)

# Create plotly figure
fig = px.scatter_mapbox(df,
    lon='lon',
    lat='lat',
    color='SST anomaly(C)',
    size='SST anomaly(C)',  # Use the same column for size
    size_max=15,  # Increase the maximum size of the markers
    color_continuous_scale='orrd',
    range_color=[0, 3],
    zoom=1,
    mapbox_style='open-street-map'
)

# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    dcc.Graph(id='mhw-graph', figure=fig)
])

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)