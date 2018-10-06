# DigitRecognition
This package come with the trained and saved model. If you just want to try, use following command:<br/>
python inference.py <br/> <br/>
This will load the default training model saved in mnist-model.h5, and predict the label of img_1.jpg file. After predicting the label, the inference plot the image. The image is displayed for 2 seconds and automatically closes.


The training file is mnistcnn.py. To train the model, use following command
python mnistcnn.py

After training is complete the model is saved in minst-mode.h5. 

Your can use python inference.py <Raw Image File Name> to run the inference. The folder testSample contains the raw image file. You can issue following commands for inference:
python inference.py testSample/img_67.jpg 
