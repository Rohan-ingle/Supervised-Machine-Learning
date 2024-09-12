import streamlit as st
import cv2
import pickle
import numpy as np
from skimage.feature import hog
from skimage import exposure
from ultralytics import YOLO
from collections import Counter
import io

def load_models():
    with open(r'models\best_svm_classifier.pkl', 'rb') as f:
        svm_classifier = pickle.load(f)
    with open(r'models\rf_classifier_best.pkl', 'rb') as f:
        rf_classifier = pickle.load(f)
    with open(r'models\best_knn.pkl', 'rb') as f:
        knn_classifier = pickle.load(f)
    with open(r'models\best_naive_bayes.pkl', 'rb') as f:
        nb_classifier = pickle.load(f)
    return svm_classifier, rf_classifier, knn_classifier, nb_classifier

svm_classifier, rf_classifier, knn_classifier, nb_classifier = load_models()

# yolo_anomaly = YOLO(r"P:\SML\runs\detect\train3\weights\best.pt")
# yolo_anomaly = YOLO(r"train2\weights\best.pt")
yolo_anomaly = YOLO(r"best.pt")
yolo_accident = YOLO(r'models\accident.pt')

def predict_and_display(frame, model):
    if frame is None:
        st.write("Provided frame is not valid")
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=frame_rgb, imgsz=640)
    annotated_image = results[0].plot()
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return annotated_image_bgr

def anomaly_detection(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=frame_rgb, imgsz=640)
    annotated_image = results[0].plot()
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return annotated_image_bgr

def extract_hog_features(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if len(gray.shape) > 2:
        h, w, _ = gray.shape
        if h != w:
            dim = max(h, w)
            resized = cv2.resize(gray, (dim, dim))
        else:
            resized = gray
    else:
        resized = gray

    fd, hog_image = hog(resized, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    fixed_length = 13312
    if len(fd) < fixed_length:
        fd = np.pad(fd, (0, fixed_length - len(fd)))
    elif len(fd) > fixed_length:
        fd = fd[:fixed_length]

    return fd

models = [svm_classifier, rf_classifier, knn_classifier, nb_classifier]
model_names = ['SVM', 'Random Forest', 'K-NN', 'Naive Bayes']


# def extract_color_histogram(image, bins=(8, 8, 8)):
#     hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten()

# def load_and_process_image(image):
#     if image is not None:
#         hog_features = extract_hog_features(image)
#         color_hist = extract_color_histogram(image)
#         combined_features = np.hstack((hog_features, color_hist))
#         return combined_features
#     else:
#         return None
    
# def ml_predict(image, model):
#     image_features = load_and_process_image(image)
#     return model.predict([image_features])[0]
def resize_with_padding(image, target_size):
    height, width = image.shape[:2]

    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height

    aspect_ratio = width / height

    if aspect_ratio > target_aspect_ratio:
        resized_height = target_height
        resized_width = int(target_height * aspect_ratio)
    else:
        resized_width = target_width
        resized_height = int(target_width / aspect_ratio)

    resized_image = cv2.resize(image, (resized_width, resized_height))

    return resized_image


def ml_predict(image, model):
    image_features = extract_hog_features(image)
    return model.predict([image_features])[0]

def ensemble_predictions(image, models, techniques):
    if image.size!= (640,640): resize_with_padding(image, (640, 640))
    predictions = []

    for model in models:
        predictions.append(1 if ml_predict(image, model)== "Accident" else 0)
    
    if 'Averaging' in techniques:
        confidence = np.mean(predictions)
        return 'Accident' if confidence > 0.5 else 'Not Accident'
        
    elif 'Voting' in techniques:
        accident_count = sum(predictions)
        return 'Accident' if accident_count > len(models) / 2 else 'Not Accident'
    else:
        return "Didnt run"

st.title('Road Anomaly Detection System')
option = st.sidebar.selectbox('Choose the detection mode:', ['Road Anomaly Detection', 'Accident Detection', 'Combined'])


def combined_detection(frame, anomaly_model, accident_model, ml_models, ensemble_method):
    anomaly_frame = anomaly_detection(frame, anomaly_model)
    
    accident_frame = predict_and_display(frame, accident_model)
    
    ml_prediction = ensemble_predictions(frame, ml_models, [ensemble_method])
    
    return anomaly_frame, accident_frame, ml_prediction



####################################################################################################################################################################################################################################################################
if option == 'Road Anomaly Detection':
    source = st.sidebar.radio('Input source:', ['Upload Image', 'Live Camera'])
    
    if source == 'Live Camera':
        fps_anomaly = st.sidebar.number_input('Set FPS (frames per second)', min_value=1, max_value=60, value=30)
        cap = cv2.VideoCapture(2)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = predict_and_display(frame, yolo_anomaly)
            stframe.image(annotated_frame, channels='BGR', use_column_width=True)
            cv2.waitKey(int(1000/fps_anomaly))
        cap.release()


    elif source == 'Upload Image':
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'mp4'])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            if uploaded_file.type.startswith('image'):
                frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
                annotated_frame = predict_and_display(frame, yolo_anomaly)
                st.image(annotated_frame, channels='BGR', use_column_width=True)
            elif uploaded_file.type.startswith('video'):
                pass
                # vid_stream = io.BytesIO(file_bytes)
                # vid_decoded = cv2.VideoCapture()
                # while True:
                #     success, frame = vid_decoded.read()
                #     if not success:
                #         break
                #     frame = cv2.imdecode(np.frombuffer(vid_stream.read(), np.uint8), cv2.IMREAD_COLOR)
                #     annotated_frame = predict_and_display(frame, yolo_anomaly)
                #     st.image(annotated_frame, channels='BGR', use_column_width=True)
                # vid_decoded.release()

                
####################################################################################################################################################################################################################################################################



elif option == 'Accident Detection':
    source = st.sidebar.radio('Input source:', ['Upload Image', 'Live Camera'])
    detection_method = st.sidebar.radio('Choose detection method:', ['YOLO Model', 'Machine Learning Models'])
    
    if detection_method == 'YOLO Model':
        st.write("Using YOLO model for accident detection.")

        if source == 'Live Camera':
            fps = st.sidebar.number_input('Set FPS (frames per second)', min_value=1, max_value=60, value=30)
            cap = cv2.VideoCapture(2)
            stframe = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated_frame = predict_and_display(frame, yolo_accident)
                stframe.image(annotated_frame, channels='BGR', use_column_width=True)
                cv2.waitKey(int(1000/fps))
            cap.release()
        elif source == 'Upload Image':
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                if uploaded_file.type.startswith('image'):
                    frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
                    annotated_frame = predict_and_display(frame, yolo_accident)
                    st.image(annotated_frame, channels='BGR', use_column_width=True)
                else:
                    pass

    elif detection_method == 'Machine Learning Models':
        
        model_usage = st.sidebar.multiselect('Choose models to use:', model_names, default=['SVM', 'Random Forest'])
        ensemble_method = st.sidebar.radio('Select ensemble technique:', ['Averaging', 'Voting'])
        selected_models = [models[i] for i, name in enumerate(model_names) if name in model_usage]

        st.write("Using selected ML models for accident detection with {} technique.".format(ensemble_method))

        if source == 'Upload Image':
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                st.image(file_bytes, channels='BGR', use_column_width=True)
                if uploaded_file.type.startswith('image'):
                    image = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
                    prediction = ensemble_predictions(image, selected_models, [ensemble_method])
                    st.write("Ensemble Prediction:", prediction)
                else:
                    pass
        elif source == 'Live Camera':
            flag_frame = st.sidebar.number_input('Trigger accident after (frames)', min_value=1, value=2)
            fps = st.sidebar.number_input('Set FPS (frames per second)', min_value=0.1, max_value=60.0, value=1/5)
            cap = cv2.VideoCapture(2)
            stframe = st.empty()
            flag = 0
            prev_acc=0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                prediction = ensemble_predictions(frame, selected_models, [ensemble_method])
                if prediction == 'Accident':
                    flag += 1 
                stframe.image(frame, channels='BGR', use_column_width=True)
                if flag > flag_frame:
                    st.write("Accident was detected")
                    # prev_acc+=1

                # cv2.waitKey(int(1000 / fps))
            cap.release()

########################################################################################################################################

if option == 'Combined':
    source = st.sidebar.radio('Input source:', ['Upload Image', 'Live Camera'])
    model_usage = st.sidebar.multiselect('Choose models to use:', model_names, default=['SVM', 'Random Forest'])
    ensemble_method = st.sidebar.radio('Select ensemble technique:', ['Averaging', 'Voting'])
    selected_models = [models[i] for i, name in enumerate(model_names) if name in model_usage]

    st.write("Using combined model for anomaly and accident detection with {} technique.".format(ensemble_method))
    
    if source == 'Upload Image':
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            st.image(file_bytes, channels='BGR', use_column_width=True)
            if uploaded_file.type.startswith('image'):
                image = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
                anomaly_frame, accident_frame, ml_prediction = combined_detection(image, yolo_anomaly, yolo_accident, selected_models, ensemble_method)
                st.image(anomaly_frame, channels='BGR', use_column_width=True, caption="Anomaly Detection")
                st.image(accident_frame, channels='BGR', use_column_width=True, caption="Accident Detection")
                st.write("Ensemble ML Prediction:", ml_prediction)
            else:
                pass
    elif source == 'Live Camera':
        fps = st.sidebar.number_input('Set FPS (frames per second)', min_value=0.1, max_value=60.0, value=1/5)
        cap = cv2.VideoCapture(2)
        stframe_anomaly = st.empty()
        stframe_accident = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            anomaly_frame, accident_frame, ml_prediction = combined_detection(frame, yolo_anomaly, yolo_accident, selected_models, ensemble_method)
            stframe_anomaly.image(anomaly_frame, channels='BGR', use_column_width=True, caption="Anomaly Detection")
            stframe_accident.image(accident_frame, channels='BGR', use_column_width=True, caption="Accident Detection")
            st.write("Ensemble ML Prediction:", ml_prediction)
            cv2.waitKey(int(1000 / fps))
        cap.release()
