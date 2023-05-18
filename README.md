Active Inactive Student Classification System
This system utilizes webcam data to classify students as active or inactive during an online class. It follows the following steps:

1. Eye Tracking: The system first detects and tracks 20 points of the user's eyes using computer vision techniques.

2. Data Storage: The eye tracking data is then stored in CSV format for further processing and analysis.

3. Classification: The stored eye tracking data is used as input for a Random Forest classifier. The classifier predicts whether the student is active or inactive based on the eye movement patterns.

4. Model Saving: Once the classification is completed, the trained Random Forest model is saved for future use.

5. Real-Life Deployment: The system can be deployed in real-life online classes to monitor student activity and provide insights on their engagement levels.


Requirements

opencv-python==4.7.0.72
mediapipe==0.9.3.0
scikit-learn==1.2.2



License
MIT License

Feel free to customize the above template to include additional details specific to your system. Make sure to replace the placeholders (e.g., shukur-alom, https://github.com/shukur-alom/Active-inactive-student-classification-by-webcam-in-an-online-class) with the appropriate information.