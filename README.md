# CNN
Experiment of CNN for Scratch build layers VS network VGG16

# Task 1

• Train the VGG16 with your training data (from provided dataset) 

![image](https://github.com/globlo/CNN/assets/49570577/198ef9a1-888f-42f0-8712-7070f2246862)


• Test the accuracy
![image](https://github.com/globlo/CNN/assets/49570577/3c319feb-7cea-4b08-822a-34b6bdc17b87)



# Task 2
• Train your network with the dataset provided 
This time, not like the HW3 I did, I set epoch 3 with and I used the same design as HW3. To apply the data seg_train and seg_test, I added the new line for resize the data.
 
 ![image](https://github.com/globlo/CNN/assets/49570577/8625a132-671a-42b8-8db1-e254ca377c93)
![image](https://github.com/globlo/CNN/assets/49570577/fb81f157-e45b-431b-89da-955b6bb54341)


• Test the accuracy

 ![image](https://github.com/globlo/CNN/assets/49570577/2625c56a-001c-49bd-be99-d8aa5af716ab)



# Comparisons: 
• Which network is giving you best accuracy? And why ? 
As the result, using network VGG16 gaves the better accuracy (91.56%) over the CNN built from scratch in HW3 (71.14%). This is because the network has the standard designed layers that is well studied and has a lot of trials. It will be very difficult to design the best layers for CNN from scratch in short amount of time.

Other thing I’ve noticed is that applying convolution in the layers of size 3*3 with more layers has less computing cost over 5*5.
