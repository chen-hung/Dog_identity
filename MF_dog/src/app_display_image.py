import os
from uuid import uuid4
from test_sphereface import detect_dog
from flask import Flask, request, render_template, send_from_directory
import time

__author__ = 'ibininja'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    #save_name = "nana_1"
    #return render_template("complete_display_image.html", image_name4="{}_result_dog0.jpg".format(save_name),image_name2="{}_breed0.jpg".format(save_name),image_name3="{}_sphereface.jpg".format(save_name))

    return render_template("upload.html")#render_template("complete_display_image.html")
#render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print("hahaha")
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        save_name=os.path.splitext(filename)[0]
        print("Save_name",save_name)
        detect_dog(destination,save_name)
        #time.sleep(2)
        #upload.save("/".join([target, "breed.jpg"]))
    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete_display_image.html", image_name=filename,image_name4="{}_result_dog0.jpg".format(save_name),image_name2="{}_breed0.jpg".format(save_name),image_name3="{}_sphereface.jpg".format(save_name))

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=4555, debug=True)
    #app.run(host='0.0.0.0', debug=True)
