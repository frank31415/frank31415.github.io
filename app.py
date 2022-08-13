from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, send_from_directory, request, session
from skimage.transform import resize
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from skimage.color import rgb2gray
from skimage import io
import imageio
import numpy as np
import imghdr
from common import cache

import cv2

 
#Initialize Flask and set the template folder to "template"
#app = Flask(__name__, template_folder = 'template')
app = Flask(__name__, template_folder='template', static_folder='static')

#Open our model
path = 'C:/Users/frank/OneDrive/Počítač/Programming/Flask/AgeGuessing/'


model = Sequential()
model.add(Conv2D(32, kernel_size=(7,7),input_shape=(100, 100, 1),activation='relu'))
#model.add(Conv2D(48, kernel_size=(3,3),activation='relu'))#,padding='same'
#model.add(Conv2D(64, kernel_size=(5,5),activation='relu'))
model.add(Conv2D(32, kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
#model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
#model.add(Dense(128, activation="relu", input_shape=(48, 48, 1)))
#model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="relu"))
#model.add(Dense(48, activation="relu"))
#model.add(Dense(64, activation="relu"))
#model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))


model.add(Dense(10, activation="softmax"))

#opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer= "adam")

model.load_weights(path+"models/age_predictor.h5")
#model = pickle.load(open(path + 'models/age_predictor.h5','rb'))

modelgen = Sequential()
modelgen.add(Conv2D(32, kernel_size=(7,7),input_shape=(100, 100, 1),activation='relu'))
modelgen.add(Conv2D(32, kernel_size=(5,5),activation='relu'))#,padding='same'
modelgen.add(MaxPooling2D(pool_size=(2,2)))
modelgen.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
#model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
#model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
modelgen.add(Flatten())
modelgen.add(Dropout(0.5))
#model.add(Dense(128, activation="relu", input_shape=(48, 48, 1)))
#model.add(Dense(128, activation="relu"))
modelgen.add(Dense(32, activation="relu"))
#model.add(Dense(128, activation="relu"))
modelgen.add(Dropout(0.25))


modelgen.add(Dense(1, activation="sigmoid"))
#model.add(Dense(117, activation="softmax"))


#opt = tf.keras.optimizers.Adam(learning_rate=0.001)
modelgen.compile(loss="binary_crossentropy", optimizer= "adam")

modelgen.load_weights(path+"models/gender_predictor.h5")

#img = 'static/backgroundpic.jpg'
cache.init_app(app=app, config={"CACHE_TYPE": "filesystem",'CACHE_DIR': '__pycache__'})

# store a value
cache.set("img", 'static/backgroundpic.jpg')


# Get a value
#my_value = cache.get("my_value")


#create our "home" route using the "index.html" page
@app.route('/')
def home():
    f = open("static/Counter.txt", "r")
    num = f.read(64)
    f.close()
    f = open("static/Counter.txt", "w")
    f.write(str(int(num)+1))
    f.close()
    cache.set("counter", int(num)+1)
    return render_template('index.html', user_image = 'static/backgroundpic.jpg', counter__1 = cache.get("counter"))

UPLOAD_FOLDER = os.path.join('static', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
 
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
#app = Flask(__name__, template_folder='template', static_folder='staticFiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
 

@app.route('/upload',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        
        img_file_path = session.get('uploaded_img_file_path', None)

        cache.set("img", img_file_path)

        # Display image in Flask application web page
        return render_template('index.html', user_image = img_file_path, counter__1 = cache.get("counter"))
        #return render_template('index.html')

from flask import send_file

def det_face():
    img = cache.get("img")
    #pic1 = io.imread(img)
    pic = cv2.imread(img)
    #gray1 = rgb2gray(pic1)
    #gray = np.array(gray1, dtype='uint8')
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=12,
                minSize=(120, 120))

    coun = 0
    
    for (x, y, w, h) in faces:
        
        roi_color = pic[y:y + int(1.15*h), x-int(0.05*w):x + w+int(0.05*w)] 
        #print("[INFO] Object found. Saving locally.") 
        roi_color = re_size(roi_color)
        cv2.imwrite(path+"static/" + str(coun) + '_faces.jpg', roi_color)
        cv2.rectangle(pic, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
        #io.imsave("static/" + str(coun) + '_faces.jpg', roi_color)
        coun = coun+1
        cv2.putText(pic,str(coun),(x,y-30), cv2.FONT_HERSHEY_PLAIN, 8, (255, 255, 255), 3, cv2.LINE_AA)
    
    status = cv2.imwrite(path+"static/faces_detected.jpg", pic)

    return coun


@app.route('/guess1', methods=['GET', 'POST'])
def downloadFile(): #In your case fname is your filename
    img = cache.get("img")

    coun = det_face()

    predictions = []
    output = ""

    for j in range(coun):
            #pic1 = cv2.imread(path+"static/0_faces.jpg", cv2.IMREAD_GRAYSCALE)
            #pg = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
            pic1 = io.imread("static/" + str(j) + "_faces.jpg") #pic1[:,:,:,::-1]
            pg = rgb2gray(pic1).reshape(1, -100, 100, 1)
            predictions.append(modelgen.predict(pg))

            if predictions[j] < 0.5:
                output += str(j+1) + ". Male "
            else:
                output += str(j+1) + ". Female "

            #pic = cv2.imread(img)
            

    if len(predictions) == 0:
        return render_template('index.html', user_image = img, counter__1 = cache.get("counter"))
    else:        
        #prediction = predictions[0]
        #Round the output to 2 decimal places
        #output1 = prediction[0]
        
        #if output1 < 0.5:
        #    output += "Male"
        #else:
            #output += "Female"
        #img_file_path = cache.get("img_file_path")
        #If the output is negative, the values entered are unreasonable to the context of the application
        #If the output is greater than 0, return prediction
        return render_template('index.html', user_image = "static/faces_detected.jpg", counter__1 = cache.get("counter"), prediction_text = 'Predicted Sex: {}'.format(output))


@app.route('/detect', methods = ['POST'])
def detect():
    
    img = cache.get("img")

    coun = det_face()

    if coun > 0:
        return render_template('index.html', user_image = "static/faces_detected.jpg", counter__1 = cache.get("counter"))
    else:
        return render_template('index.html', user_image = img, counter__1 = cache.get("counter"))






@app.route('/guess', methods = ['POST'])
def predict1():
    
    img = cache.get("img")
    #pic = io.imread(img)
    #pic = cv2.imread(img)
    #gray1 = rgb2gray(pic)
    #gray = np.array(gray1, dtype='uint8')
    #gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    #pic = re_size(pic)
    #obtain all form values and place them in an array, convert into integers
    #int_features = [float(x) for x in request.form.values()]
    #Combine them all into a final numpy array
    #final_features = [np.array(int_features)]
    #predict the price given the values inputted by user

    coun = det_face()
    #pic = re_size(pic)
    #obtain all form values and place them in an array, convert into integers
    #int_features = [float(x) for x in request.form.values()]
    #Combine them all into a final numpy array
    #final_features = [np.array(int_features)]
    #predict the price given the values inputted by user
    predictions = []
    labels = ['0-5','6-12','13-18','19-25','26-35','36-45','46-55','56-65','66-75','76+']
    output = ""

    from skimage.util import img_as_float
    if coun > 0:
        for j in range(coun):
            #pic1 = cv2.imread(path+"static/0_faces.jpg", cv2.IMREAD_GRAYSCALE)
            #pg = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
            pic1 = io.imread("static/" + str(j) + "_faces.jpg") #pic1[:,:,:,::-1]
            #pg = rgb2gray(pic1)#.reshape(1, 100, 100, 1)
            #io.imsave("static/" + str(coun) + '_pg.jpg', pg)
            pg = rgb2gray(pic1).reshape(1, -100, 100, 1)
            prediction = model.predict(pg)
            predictions.append(prediction)
            prediction = labels[prediction.argmax()]
            output += str(j+1) + ". " + str(prediction) + "; "
        
        #img_file_path = cache.get("img_file_path")
        #If the output is negative, the values entered are unreasonable to the context of the application
        #If the output is greater than 0, return prediction
        return render_template('index.html', user_image = "static/faces_detected.jpg", counter__1 = cache.get("counter"), prediction_text = 'Predicted age: {}'.format(output))
    else:
        return render_template('index.html', user_image = img, counter__1 = cache.get("counter"))

def re_size(picc):
    #pik = rgb2gray(picc)
    a = (picc.shape[0] / 100)
    b = (picc.shape[1] / 100)
    cache.set("a", a)
    cache.set("b", b)
    return cv2.resize(picc,(100,100),interpolation=cv2.INTER_LINEAR) #.reshape(100,100,1)
    #return resize(picc, (picc.shape[0] // a , picc.shape[1] //  b), anti_aliasing=True).reshape(1, -48, 48, 1)

#Run app
if __name__ == "__main__":
    app.run(debug=True)