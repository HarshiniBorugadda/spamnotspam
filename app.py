import dash_bootstrap_components as dbc
from dash import Dash, Output, Input, State
from models import model, tfidf

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

@app.callback(
    Output("output-range", "value"),
    Output("output-range", "color"),
    Input("submit-button", "n_clicks"),
    State("input-text", "value"),
    prevent_initial_call=True,
)
def calculate(n_clicks, value):
    tfidf_value = tfidf.transform([value])
    _, prediction = model.predict_proba(tfidf_value)[0]
    return prediction * 100, "danger" if prediction > 0.66 else "warning" if prediction > 0.33 else "success" 

app.layout = [
    dbc.Navbar([
        dbc.NavbarBrand("SpamNotSpam"),
    ], dark=True, color="dark", class_name="py-2 px-4"),
    dbc.Stack([
        dbc.Input(placeholder="Enter your text", id="input-text"),
        dbc.Button("Calculate", id="submit-button"),
        dbc.Progress(id="output-range", color="danger"),
    ], class_name="p-4 gap-2")
]

if __name__ == "__main__":
    app.run(debug=False)