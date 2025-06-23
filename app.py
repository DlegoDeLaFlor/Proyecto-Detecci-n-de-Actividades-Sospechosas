from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from datetime import datetime
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("best.pt")

TRADUCCION = {
    "suspicious": "sospechoso",
    "helmet": "casco",
    "hooded": "encapuchado",
    "gun": "arma"
}

CLASES_ALERTA = {"sospechoso", "casco", "encapuchado", "arma"}

@app.route("/", methods=["GET", "POST"])
def index():
    alerta = None
    if request.method == "POST":
        image = request.files.get("image")
        if not image or not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template("index.html", output=None, clases=None, alerta=None)

        # Evitar nombres duplicados
        filename = datetime.now().strftime("%Y%m%d%H%M%S_") + image.filename
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(path)

        try:
            Image.open(path).verify()  # Verifica que la imagen sea válida
        except Exception:
            return render_template("index.html", output=None, clases=None, alerta=None)

        results = model(path)[0]
        results.save(filename=path)

        names = model.names
        clases_detectadas = [
            TRADUCCION.get(names[int(c)].lower(), names[int(c)])
            for i, c in enumerate(results.boxes.cls)
            if float(results.boxes.conf[i]) >= 0.4  # Ajusta el umbral si deseas
        ]

        for c in clases_detectadas:
            if c.lower() in CLASES_ALERTA:
                alerta = c
                break

        filename = os.path.relpath(path, os.path.join(os.getcwd(), "static")).replace("\\", "/")
        return render_template("index.html", output=filename, clases=clases_detectadas, alerta=alerta)

    return render_template("index.html", output=None, clases=None, alerta=None)

if __name__ == "__main__":
    app.run(debug=True)
