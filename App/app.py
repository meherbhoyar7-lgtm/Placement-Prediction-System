from flask import Flask, render_template, request
from src.data.load_data import load_data, get_summary

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dataset")
def dataset():
    df = load_data()
    summary = get_summary(df)
    return render_template(
        "load_dataset.html",
        summary=summary,
        first_rows=df.head().to_html(index=False)
    )

@app.route("/eda")
def eda():
    return render_template("eda.html")

if __name__ == "__main__":
    app.run(debug=True)