from flask import Flask, render_template, request
import pandas as pd
import joblib
import os


# =====================================================
# FLASK APP — CNC MILLING DIGITAL TWIN
# INPUTS  : 6
# OUTPUTS : 3
# =====================================================

app = Flask(__name__)


# =====================================================
# LOAD TRAINED MODELS
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


print("Models Loaded Successfully")


# =====================================================
# INPUT FEATURES
# MUST MATCH TRAINING DATA EXACTLY
# =====================================================

INPUT_COLS = [

    "Feed_Rate_(mm/min)",

    "Spindle_Speed_(rpm)",

    "Depth_of_Cut_(mm)",

    "Width_of_Cut_(mm)",

    "Cut_Dir_Encoded",

    "Coolant_Encoded"

]



# =====================================================
# HOME PAGE + PREDICTION
# =====================================================


@app.route("/", methods=["GET", "POST"])
def home():

    roughness_pred = None
    power_pred = None
    wear_pred = None
    error_message = None


    if request.method == "POST":


        try:

            # -----------------------------
            # GET INPUT VALUES FROM WEBSITE
            # -----------------------------

            feed_rate = float(
                request.form["feed_rate"]
            )


            spindle_speed = float(
                request.form["spindle_speed"]
            )


            depth_of_cut = float(
                request.form["depth_of_cut"]
            )


            width_of_cut = float(
                request.form["width_of_cut"]
            )


            cutting_direction = request.form[
                "cutting_direction"
            ]


            coolant = request.form[
                "coolant"
            ]



            # -----------------------------
            # ENCODING
            # -----------------------------

            cut_dir_encoded = (
                1 if cutting_direction == "Down"
                else 0
            )


            coolant_encoded = (
                1 if coolant == "On"
                else 0
            )



            # -----------------------------
            # CREATE INPUT DATAFRAME
            # -----------------------------


            input_data = pd.DataFrame(

                [

                    [

                    feed_rate,

                    spindle_speed,

                    depth_of_cut,

                    width_of_cut,

                    cut_dir_encoded,

                    coolant_encoded

                    ]

                ],

                columns=INPUT_COLS

            )



            print("\nINPUT DATA:")
            print(input_data)



            # -----------------------------
            # MODEL PREDICTIONS
            # -----------------------------


            roughness_pred = round(

                float(
                    model_roughness.predict(input_data)[0]
                ),

                3

            )


            power_pred = round(

                float(
                    model_power.predict(input_data)[0]
                ),

                3

            )


            wear_pred = round(

                float(
                    model_wear.predict(input_data)[0]
                ),

                4

            )



            print("\nPrediction Completed")

            print(
                "Surface Roughness:",
                roughness_pred
            )

            print(
                "Power:",
                power_pred
            )

            print(
                "Tool Wear:",
                wear_pred
            )



        except Exception as e:


            error_message = (

                "Prediction failed: "

                + str(e)

            )


            print(error_message)




    return render_template(

        "index.html",

        roughness_pred=roughness_pred,

        power_pred=power_pred,

        wear_pred=wear_pred,

        error_message=error_message

    )





# =====================================================
# RUN SERVER
# =====================================================


if __name__ == "__main__":

    app.run(debug=True)