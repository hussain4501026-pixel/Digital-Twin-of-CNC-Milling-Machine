from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# =========================================================
# FLASK APP — CNC Milling Machine Digital Twin
# Inputs  (6): Feed Rate, Spindle Speed, Depth of Cut,
#              Width of Cut, Cutting Direction, Coolant
# Outputs (3): Surface Roughness, Average Power, Tool Wear
# =========================================================

app = Flask(__name__)

# ── Load all 3 models once at startup ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_roughness = joblib.load(os.path.join(BASE_DIR, "Surface Roughness.pkl"))
model_power     = joblib.load(os.path.join(BASE_DIR, "Average Power Required.pkl"))
model_wear      = joblib.load(os.path.join(BASE_DIR, "Tool Wear.pkl"))

print("✅ All 3 models loaded successfully.")

# ── Column names — must match training exactly ────────────
INPUT_COLS = [
    "Feed_Rate_(mm/min)",
    "Spindle_Speed_(rpm)",
    "Depth_of_Cut_(mm)",
    "Width_of_Cut_(mm)",
    "Cut_Dir_Encoded",
    "Coolant_Encoded",
]


# ── Home route ────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():

    roughness_pred = None
    power_pred     = None
    wear_pred      = None
    error_message  = None

    if request.method == "POST":
        try:
            # Get form values
            feed_rate         = float(request.form["feed_rate"])
            spindle_speed     = float(request.form["spindle_speed"])
            depth_of_cut      = float(request.form["depth_of_cut"])
            width_of_cut      = float(request.form["width_of_cut"])
            cutting_direction = request.form["cutting_direction"]
            coolant           = request.form["coolant"]

            # Encode categoricals — same as training
            cut_dir_encoded = 1 if cutting_direction == "Down" else 0
            coolant_encoded = 1 if coolant == "On" else 0

            # Build input DataFrame — 6 columns
            input_data = pd.DataFrame(
                [[feed_rate, spindle_speed, depth_of_cut,
                  width_of_cut, cut_dir_encoded, coolant_encoded]],
                columns=INPUT_COLS
            )

            # Predict all 3 outputs
            roughness_pred = round(float(model_roughness.predict(input_data)[0]), 3)
            power_pred     = round(float(model_power.predict(input_data)[0]),     2)
            wear_pred      = round(float(model_wear.predict(input_data)[0]),       4)

        except Exception as e:
            error_message = f"Prediction failed: {str(e)}"

    return render_template(
        "index.html",
        roughness_pred = roughness_pred,
        power_pred     = power_pred,
        wear_pred      = wear_pred,
        error_message  = error_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
