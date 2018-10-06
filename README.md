# DigitRecognition
<b> Quickly Run Inference with Defualt Model and Image File </b><br/>
This package come with the trained and saved model. If you just want to try, use following command:<br/>
<I>python inference.py </I><br/> 
This will load the default training model saved in mnist-model.h5, and predict the label of img_1.jpg file. After predicting the label, the inference plot the image. The image is displayed for 2 seconds and automatically closes.

<b> Training and Testing the Model </b><br/>
The training file is mnistcnn.py. To train the model, use following command<br>
<I>python mnistcnn.py</I><br/>

After training is complete the model is saved in minst-mode.h5. </br>

<b> Inference </b><br/>
Your can use python inference.py <Raw Image File Name> to run the inference. The folder testSample contains the raw image file. You can issue following commands for inference:<br/>
  <I> python inference.py testSample/img_67.jpg </I>
