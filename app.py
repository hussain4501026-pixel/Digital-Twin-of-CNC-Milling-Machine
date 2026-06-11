from flask import Flask, render_template, request
import pandas as pd
import joblib
import os


app = Flask(__name__)


# =====================================================
# LOAD MODELS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model_roughness = joblib.load(
    os.path.join(BASE_DIR, "Surface Roughness.pkl")
)

model_power = joblib.load(
    os.path.join(BASE_DIR, "Average Power Required.pkl")
)

model_wear = joblib.load(
    os.path.join(BASE_DIR, "Tool Wear.pkl")
)


print("Models loaded successfully")


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template("index.html")



# =====================================================
# PREDICTION
# =====================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        print("Prediction started")


        feed_rate = float(request.form["feed_rate"])

        spindle_speed = float(
            request.form["spindle_speed"]
        )

        depth = float(
            request.form["depth_of_cut"]
        )

        width = float(
            request.form["width_of_cut"]
        )


        cutting_direction = request.form["cut_direction"]

        coolant = request.form["coolant"]



        # Encoding

        if cutting_direction == "Down":
            cut_dir_encoded = 1
        else:
            cut_dir_encoded = 0



        if coolant == "On":
            coolant_encoded = 1
        else:
            coolant_encoded = 0



        # IMPORTANT
        # same columns used during training

        input_data = pd.DataFrame({

            "Feed_Rate_(mm/min)":[feed_rate],

            "Spindle_Speed_(rpm)":[spindle_speed],

            "Depth_of_Cut_(mm)":[depth],

            "Width_of_Cut_(mm)":[width],

            "Cut_Dir_Encoded":[cut_dir_encoded],

            "Coolant_Encoded":[coolant_encoded],

            "Tool_Wear_(mm)":[0.06]

        })


        print(input_data)



        # Predictions

        roughness_prediction = (
            model_roughness.predict(input_data)[0]
        )


        power_prediction = (
            model_power.predict(input_data)[0]
        )


        wear_prediction = (
            model_wear.predict(input_data)[0]
        )


        print("Prediction completed")



        return render_template(

            "index.html",

            surface=round(
                roughness_prediction,3
            ),

            power=round(
                power_prediction,3
            ),

            tool=round(
                wear_prediction,3
            )

        )



    except Exception as e:


        print("ERROR:")
        print(e)

        return str(e)



if __name__=="__main__":

    app.run(debug=True)