import sys,os #for work related to directories and saving
#==================================================================================================== PATHS ====================================================================================================
current=os.path.dirname(__file__)
parent=os.path.abspath(os.path.join(current,".."))
sys.path.append(parent)
resources=os.path.join(current,"resources")
gif=os.path.join(resources,"gif")
logo=os.path.join(resources,"logo")
splash=os.path.join(resources,"splash","splash.png")
log_path=os.path.join(parent,"..","Dashboard","live_predictions_log.xlsx")
#==================================================================================================== IMPORTS ====================================================================================================
import numpy as np #image reading,resizing,heatmap overlay
import cv2 #used to read & resize image, make colormap & overlay
from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget,QHBoxLayout,QVBoxLayout,QLabel,QPushButton,QFileDialog,QCheckBox,QComboBox,QFrame,QSplashScreen,QToolTip #ui widgets
from PyQt5.QtGui import QPixmap,QIcon,QMovie,QCursor #images,icon,loader,tooltip
from PyQt5.QtCore import Qt,QTimer #alignment,loading delay
from inference.predictor import predict_image #import of predictor.py
from explainability.Grad_Cam import GradCAM #import of gradcam.py
import pandas as pd
from datetime import datetime
#==================================================================================================== UI CLASS ====================================================================================================
class App(QMainWindow):
    def __init__(self):
        super().__init__() #initialise main window(inherit qmainwindow features)
        self.prediction_active=False #prediction activation flag
        #==================================================================================================== WINDOW ====================================================================================================
        self.setWindowTitle("Crop Disease Prediction") #window title
        self.setWindowIcon(QIcon(os.path.join(logo,"app_icon.png"))) #window icon
        self.resize(1,1) #window auto-size according to content
        central=QWidget() #central container
        self.setCentralWidget(central) #add central container
        outer=QVBoxLayout(central) #alignment of central container is set to vertical
        outer.setAlignment(Qt.AlignCenter) #align items in container to center
        #==================================================================================================== App's Styling ====================================================================================================
        self.setStyleSheet("""
        QMainWindow {background-color:#ecfdf5;}
        QFrame {background-color:white; border-radius:18px;}
        QLabel {color:#064e3b;}
        QPushButton {padding:10px; border-radius:10px; background-color:#16a34a; color:white; font-weight:600;}
        QPushButton:disabled { background-color: #a7f3d0;}
        QToolTip {background-color: #064e3b; color: #ecfdf5; border: 1px solid #22c55e; border-radius: 6px; padding: 6px 10px; font-size: 12px;}
        """)
        #==================================================================================================== main_box ====================================================================================================
        main_container=QFrame() #main container
        main_container.setFixedWidth(480) #main container width 
        outer.addWidget(main_container) #add main container to window
        layout=QVBoxLayout(main_container) #layout of main container
        layout.setSpacing(16) #spacing of main container
        layout.setContentsMargins(48,48,48,48) #padding of main container
        #==================================================================================================== TITLE ====================================================================================================
        title=QLabel("🌿🌿 Crop Disease Prediction 🌿🌿") #title
        title.setAlignment(Qt.AlignCenter) #title allignment
        title.setStyleSheet("font-family:'Inter'; font-size:24px; font-weight:600; letter-spacing:0.6px; color:#1f2937;") #title font styling
        layout.addWidget(title) #add title
        #==================================================================================================== CROP-DROPDOWN-BOX ====================================================================================================
        self.crop_box=QComboBox() #dropdown widget(self variable to make it accessible by all class's functions)
        self.crop_box.addItems(["-Select-Crop-Name-","Corn","Cotton","Potato","Rice","Sugarcane","Tea","Wheat"]) #dropdown items
        layout.addWidget(self.crop_box) #add dropdown
        self.crop_box.currentIndexChanged.connect(self.update_predict_state) #checks for change in dropdown box and updates predict button state
        self.crop_box.currentIndexChanged.connect(self.lock_placeholder) #checks for change in dropdown box to lock default choice
        model=self.crop_box.model() #access dropdown item model
        model.item(0).setEnabled(False) #disable dropdown placeholder
        #==================================================================================================== UPLOAD ====================================================================================================
        upload=QPushButton("Upload Leaf Image") #upload button initialised
        upload.clicked.connect(self.load_image) #call load_image function on click
        layout.addWidget(upload) #add upload button
        self.image_path=None #set uploaded image path to null
        #==================================================================================================== PREVIEW ====================================================================================================
        self.preview=QLabel("Image Preview") #preview space for uploaded image
        self.preview.setFixedSize(380,260) #dimensions of preview space
        self.preview.setAlignment(Qt.AlignCenter) #align preview space center
        self.preview.setStyleSheet("border:2px dashed #bbf7d0; border-radius:14px; font-size:14px;") #preview space styling
        layout.addWidget(self.preview,alignment=Qt.AlignCenter) #add preview space to main container
        #==================================================================================================== GRAD-CAM ====================================================================================================
        self.gradcam_path=None #set grad-cam image path to null
        self.last_gradcam=False #flag to check if grad-cam image is generated
        self.checkbox=QCheckBox("Enable GRAD-CAM") #checkbox for grad-cam
        self.checkbox.setChecked(False) #set checkbox to unchecked (default)
        layout.addWidget(self.checkbox) #add checkbox to main container
        self.checkbox.stateChanged.connect(self.gradcam_toggle) #checks toggle of checkbox and calls function 
        #==================================================================================================== PREDICT BUTTON====================================================================================================
        self.predict_button=QPushButton("Predict Disease") #button to predict
        self.predict_button.setEnabled(False) #button is disabled (default)
        self.predict_button.clicked.connect(self.loading_icon) #on-click call loading function
        layout.addWidget(self.predict_button) #add predict button
        #==================================================================================================== TOOLTIP ====================================================================================================
        self.predict_tooltip_text=("Select crop and upload image to predict") #initialise tooltip prompt
        self.predict_button.enterEvent=self.show_tooltip #show prompt on hower
        self.predict_button.leaveEvent=self.hide_tooltip #hide prompt
        #==================================================================================================== RESET BUTTON ====================================================================================================
        self.reset_btn=QPushButton("Reset") #reset button initialised
        self.reset_btn.setStyleSheet("background:#e5e7eb; color:#064e3b;") #style reset button
        self.reset_btn.clicked.connect(self.reset) #call reset function
        layout.addWidget(self.reset_btn) #add reset button
        #==================================================================================================== RESULT BOX LAYOUT ====================================================================================================
        self.result_box_layout=QFrame() #result box
        self.result_box_layout.setFixedHeight(200) #initialise height of result box
        self.result_box_layout.setStyleSheet("background:white; border-radius:14px;") #result box design
        #==================================================================================================== LOADING ====================================================================================================
        self.loader=QLabel() #label for loading
        self.loader.setAlignment(Qt.AlignCenter) #align loading label center
        self.loader_gif=QMovie(os.path.join(gif,"loading.gif")) #initialise loading animation
        self.loader.setMovie(self.loader_gif) #attach loading animation inside label
        self.loader.hide() #loader is hidden till prediction 
        #==================================================================================================== RESULT TEXT ====================================================================================================
        self.result=QLabel("") #initialise result text
        self.result.setAlignment(Qt.AlignCenter) #align result center
        self.result.setStyleSheet("font-size:16px; font-weight:600;") #text styling
        #==================================================================================================== RESULT BOX ====================================================================================================
        result_layout=QVBoxLayout(self.result_box_layout) #add result box
        result_layout.setAlignment(Qt.AlignCenter) #align result box center
        result_layout.setContentsMargins(0,0,0,0) #set 0 padding to content inside result box(remove extra space inside)
        result_layout.addWidget(self.loader) #add loader in result box
        result_layout.addWidget(self.result) #add results to result box
        # ---- thank you image inside result box ----
        self.thankyou_img = QLabel()
        self.thankyou_img.setAlignment(Qt.AlignCenter)
        self.thankyou_img.hide()

        self.thankyou_pix = QPixmap(
            os.path.join(resources, "thankyou", "thankyou.png")
        )

        self.thankyou_img.setPixmap(
            self.thankyou_pix.scaled(
                180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        result_layout.addWidget(self.thankyou_img)

        layout.addWidget(self.result_box_layout) #add result box to main box
                # ==================== FEEDBACK UI ====================

        self.feedback_label = QLabel("Can you help us verify this result?")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.hide()
        layout.addWidget(self.feedback_label)
        self.feedback_label.setWordWrap(True)


        self.feedback_btn_box = QWidget()
        fb_layout = QHBoxLayout(self.feedback_btn_box)

        self.btn_yes = QPushButton("Yes, it looks correct")
        self.btn_no  = QPushButton("No, it's incorrect")

        fb_layout.addWidget(self.btn_yes)
        fb_layout.addWidget(self.btn_no)

        self.feedback_btn_box.hide()
        layout.addWidget(self.feedback_btn_box)

        # ---- dropdown (only for No) ----
        self.correct_label = QLabel("What is the correct disease?")
        self.correct_label.setAlignment(Qt.AlignCenter)
        self.correct_label.hide()
        layout.addWidget(self.correct_label)
        self.correct_label.setWordWrap(True)


        self.correct_box = QComboBox()
        self.correct_box.hide()
        layout.addWidget(self.correct_box)

        self.submit_review_btn = QPushButton("Submit review")
        self.submit_review_btn.hide()
        layout.addWidget(self.submit_review_btn)

       

        # connections
        self.btn_yes.clicked.connect(self.feedback_yes_clicked)
        self.btn_no.clicked.connect(self.feedback_no_clicked)
        self.submit_review_btn.clicked.connect(self.feedback_submit_clicked)

        #==================================================================================================== DISCLAIMER ====================================================================================================
        self.disclaimer=QLabel("Prediction results are generated on the basis of image quality and similarity to training data.") #disclaimer text
        self.disclaimer.setAlignment(Qt.AlignCenter) #center align disclaimer
        self.disclaimer.setWordWrap(True) #textwrap
        self.disclaimer.setStyleSheet("color:#dc2626; font-size:11px;") #disclaimer style
        self.disclaimer.hide() #hidden till prediction is generated
        layout.addWidget(self.disclaimer) #add to main box
    #==================================================================================================== RESET FUNCTION====================================================================================================
    def reset(self): #function for reset button
        self.prediction_active=False #when reset is clicked no prediction takes place
        self.result_box_layout.setStyleSheet("background:white; border-radius:14px;") #clears result box
        self.crop_box.setCurrentIndex(0) #set dropdown to default
        self.preview.clear() #removes uploaded image
        self.preview.setText("Image Preview") #reset preview
        self.result.clear() #remove result
        self.disclaimer.hide() #remove disclaimer
        self.predict_button.setEnabled(False) #predict button disabled
        self.checkbox.setChecked(False) #Grad-cam unchecked (default)
        self.gradcam_path=None #gradcam path initialised null
        self.last_gradcam=False #flag for last grad-cam
        self.loader_gif.stop() #stop loading
        self.loader.hide() #remove loading
        self.image_path=None #clears uploaded image path
        model = self.crop_box.model() #access dropdown
        model.item(0).setEnabled(True) #enables default choice in dropdown
        self.crop_box.setCurrentIndex(0) #set dropdown to default
        self.hide_feedback_controls()
        self.thankyou_img.hide()
        self.result.show()
        self.checkbox.show()

    #==================================================================================================== TOOLTIP FUNCTION====================================================================================================
    def show_tooltip(self,event): #function to show tooltop
        if not self.predict_button.isEnabled(): #checks if predict button is disabled
            QToolTip.showText(QCursor.pos(),self.predict_tooltip_text,self.predict_button) #show tooltip
    def hide_tooltip(self, event): #function to hide tooltip
        QToolTip.hideText() #hide tooltip
    #==================================================================================================== GRADCAM TOGGLE FUNCTION ====================================================================================================
    def gradcam_toggle(self,state): #function to toggle gradcam
        if self.image_path is None: #check if image has been uploaded
            return
        if state==Qt.Unchecked: #check gradcam checkbox for unchecked
            self.preview.setPixmap(QPixmap(self.image_path).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        # ON → last generated Grad-CAM (agar available hai)
        elif state==Qt.Checked and self.last_gradcam: #check gradcam checkbox for checked and last gradcam is available
            self.preview.setPixmap(QPixmap(self.gradcam_path).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    #==================================================================================================== LOCK DROPDOWN ====================================================================================================
    def lock_placeholder(self): #function to lock dropdown
        if self.crop_box.currentIndex()!=0: #checks if dropdown is set to default
            model=self.crop_box.model() #access dropdown
            model.item(0).setEnabled(False) #disable dropdown placeholder
    #==================================================================================================== PREDICT BUTTON BEHAVIOUR ====================================================================================================
    def update_predict_state(self): #function for behaviour of predict button
        valid_crop=self.crop_box.currentIndex()!=0 #flag variable for current dropdown index
        has_image=self.image_path is not None #flag variable to check image upload status
        if valid_crop and has_image: #check image upload and crop selection
            self.predict_button.setEnabled(True) #prediction enabled
            self.predict_tooltip_text = "" #tooltip text removed
        else:
            self.predict_button.setEnabled(False) #predict button disabled
            if not valid_crop and not has_image: #if both are not present
                self.predict_tooltip_text = "Select crop and upload image to predict"
            elif not valid_crop: #if only crop is not selected
                self.predict_tooltip_text = "Select a crop"
            elif not has_image: #if image is not uploaded
                self.predict_tooltip_text = "Upload an image"
    #==================================================================================================== LOAD IMAGE ====================================================================================================
    def load_image(self): #load image for prediction
        path,_=QFileDialog.getOpenFileName(self,"Select Image","","Images (*.png *.jpg *.jpeg)") #file picker
        if path: #if image path exists(image is uploaded)
            self.image_path=path #save image path
            self.preview.setPixmap(QPixmap(path).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)) #load,resize image to preview box and display without distortion smoothly
            self.update_predict_state() #updates predict button state
    #==================================================================================================== LOADING ====================================================================================================
    def loading_icon(self): #function for loading
        self.thankyou_img.hide()
        self.prediction_active=True #flag for prediction
        self.result_box_layout.setStyleSheet("background: #f0fdf4; border-radius: 14px;") #updates result box for prediction
        self.result.clear() #remove last result
        self.disclaimer.hide() #hides disclaimer
        self.loader.show() #shows loading
        self.loader_gif.start() #loading animation starts
        self.predict_button.setEnabled(False) #disable predict button while loading
        QTimer.singleShot(2500,self.stop_loading) #stops loading after 2.5 sec and predicts
    #==================================================================================================== STOP LOADING ====================================================================================================
    def stop_loading(self): #function for stoping loading
        if not self.prediction_active or self.image_path is None: #if image not uploaded and prediction flag is inactive(reset) does nothing
            return 
        self.loader_gif.stop() #else pause loading
        self.loader.hide() #remove loading
        self.predict() #start prediction
    #==================================================================================================== PREDICTION ====================================================================================================
    def predict(self): #function for prediction
        crop=self.crop_box.currentText() #selects current crop for prediction
        result=predict_image(self.image_path,crop) #calls predict image file for current image and crop
        self.result.setText(f""" 
            <b>Crop:</b> {crop}<br><br>
            <b>Top:</b> {result['top1_class']} ({result['top1_con']:.2f}%)<br>
            <b>Second:</b> {result['top2_class']} ({result['top2_con']:.2f}%)
        """) #fetch top2 probabilities for selected crop and prints
        self.last_prediction = {
        "crop": crop,
        "predicted_label": result["top1_class"],
        "confidence": result["top1_con"],
        "image_path": self.image_path
        }

        self.disclaimer.show() #show disclaimer
        if self.checkbox.isChecked(): #checks if gradcam checkbox is checked
            gradcam=GradCAM(model=result["model"],target_layer=result["model"].features[-1]) #gradcam is called and last cnn layer(-1) is called as it has most features
            heatmap=gradcam.generate(input_tensor=result["tensor"],class_idx=result["class_idx"])  #applies gradcam algorithm on basis of the tensor(uploaded image) and crop class
            img=cv2.imread(self.image_path) #load image
            img=cv2.resize(img,(224,224)) #change image's dimension (for readable by gradcam we use 224 as used by models)
            heatmap_color=cv2.applyColorMap(np.uint8(255*heatmap),cv2.COLORMAP_TURBO) #converts heatmap value from (0-1)(heatmap) to (0-255)(required by opencv)(greyscale) and maps its value to rgb
            overlay=cv2.addWeighted(img,0.6,heatmap_color,0.4,0) #overlays heatmap on top of leaf image(60-40 ratio)
            out_path=os.path.join(current,"gradcam_output.png") #path to save gradcam
            cv2.imwrite(out_path,overlay) #saves gradcam
            self.gradcam_path=out_path #variable for saved gradcam path
            self.last_gradcam=True #last gradcam is available
            self.preview.setPixmap(QPixmap(out_path).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)) #show gradcam in preview
        self.thankyou_img.hide()
        self.result.show()

        
        self.predict_button.setEnabled(True) #predict button is enabled again
        # hide gradcam checkbox if user did not use it
        if not self.checkbox.isChecked():
            self.checkbox.hide() #if check box is not checked and predict button is pressed, hide checkbox

        # ----- show feedback UI -----#######################################################################################################################
        self.feedback_label.show()
        self.feedback_btn_box.show()

        # fill dropdown dynamically from classes
        self.correct_box.clear()
        self.class_names = result["class_names"]
        self.correct_box.addItems(self.class_names)


#================================================================================================================================================================
    def feedback_yes_clicked(self):

        self.feedback_btn_box.hide()
        self.feedback_label.hide()
         # hide result text
        self.result.hide()

        # show thank you image in result box
        self.thankyou_img.show()
        self.append_review_to_excel(
        crop=self.last_prediction["crop"],
        predicted_label=self.last_prediction["predicted_label"],
        confidence=self.last_prediction["confidence"],
        user_verified="yes",
        user_correct_label=self.last_prediction["predicted_label"]
        )   

    def feedback_no_clicked(self):

        self.feedback_btn_box.hide()
        self.feedback_label.hide()

        self.correct_label.show()
        self.correct_box.show()
        self.submit_review_btn.show()


    def feedback_submit_clicked(self):

        self.correct_label.hide()
        self.correct_box.hide()
        self.submit_review_btn.hide()
        self.feedback_label.hide()

        # hide result text
        self.result.hide()

        # show thank you image in result box
        self.thankyou_img.show()

        correct_label = self.correct_box.currentText()

        self.append_review_to_excel(
            crop=self.last_prediction["crop"],
            predicted_label=self.last_prediction["predicted_label"],
            confidence=self.last_prediction["confidence"],
            user_verified="no",
            user_correct_label=correct_label
        )



    def hide_feedback_controls(self):
        self.feedback_label.hide()
        self.feedback_btn_box.hide()
        self.correct_label.hide()
        self.correct_box.hide()
        self.submit_review_btn.hide()
    def append_review_to_excel(
        self,
        crop,
        predicted_label,
        confidence,
        user_verified,
        user_correct_label
    ):

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "crop": crop,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "user_verified": user_verified,          # Yes / No
            "user_correct_label": user_correct_label
        }

        df_new = pd.DataFrame([row])

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        if os.path.exists(log_path):
            df_old = pd.read_excel(log_path)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final.to_excel(log_path, index=False)


#==================================================================================================== LIKE MAIN FUNCTION ====================================================================================================
if __name__=="__main__": #checks if current file is run directly
    app=QApplication(sys.argv) #engine for qt(gui)
    window=App() #create class object window
    window.adjustSize() #calculate final window size from layout
    window_size=window.size() # get calculated window size 
    #==================================================================================================== SPLASH ====================================================================================================
    splash=os.path.join(resources,"splash","splash.png") #path for splash screen
    splash_pix=QPixmap(splash).scaled(window_size,Qt.IgnoreAspectRatio) #scale splash to window size
    splash_img=QSplashScreen(splash_pix,Qt.WindowStaysOnTopHint)  #create splash screen window
    splash_img.show() #show splash screen
    QTimer.singleShot(3000,splash_img.close) #close splash after 3 seconds
    QTimer.singleShot(3000,window.show) #show main window after splash
    sys.exit(app.exec_()) #start Qt event loop and exit cleanly