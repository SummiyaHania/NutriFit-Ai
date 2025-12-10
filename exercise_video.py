from flask import request, jsonify


def handle_exercise_video():

    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    # Print the received data
    print("Received Exercise Video Data:")
    print(data)

    # You can later add code to save this data to a database
    return jsonify({"status": "success", "message": "Data received"}), 200
