# E-Commerce Product Classification using Deep Learning

This project focuses on developing a robust deep learning architecture to classify products from an e-commerce platform into 8 distinct categories. The classification task is crucial for enhancing user experience, facilitating search functionalities, and optimizing product recommendations. Through a meticulous pipeline encompassing data collection, preprocessing, model training, validation, and deployment, the project aims to deliver an accurate and efficient solution.

## Data Collection and Preprocessing

The journey begins with the acquisition of data, primarily through screenshots from Slash application. However, a significant challenge arose due to the inherent imbalance in the collected dataset, which could potentially bias the model's learning process. To mitigate this issue, a two-pronged approach was adopted:

1. **Additional Data Scraping**: Leveraging web scraping techniques, supplementary images were sourced from the internet to augment the dataset, ensuring a more balanced representation across all categories.
   
2. **Data Augmentation**: Through various augmentation techniques such as rotation, flipping, and scaling, the dataset's diversity was enhanced. This augmentation process not only aids in addressing the class imbalance but also improves the model's generalization ability.

## Dataset Preparation

The collected dataset underwent meticulous preparation to ensure its suitability for training deep learning models:

1. **Labeling**: Each image was meticulously labeled with its corresponding category, laying the groundwork for supervised learning.
   
2. **Preprocessing**: Prior to feeding the data into the neural network, preprocessing steps such as resizing, cropping, and normalization were applied to standardize the input format and facilitate efficient training.
   
3. **Normalization**: Image pixel values were normalized to a standardized range, typically between 0 and 1, to stabilize the training process and accelerate convergence.
   
4. **Augmentation**: Augmentation techniques were judiciously applied to augment the dataset, injecting variability and robustness into the training samples.

## Model Architectures

A suite of deep learning architectures was explored to ascertain the most effective model for the classification task. The architectures investigated include:

1. **Basic CNN**: A fundamental Convolutional Neural Network (CNN) architecture, typically consisting of convolutional, pooling, and fully connected layers.
   
2. **Residual CNN**: Building upon the basic CNN, the Residual CNN architecture incorporates residual connections, enabling deeper networks to be trained more effectively.
   
3. **Fine-tuned VGG16**: VGG16, a pre-trained convolutional neural network known for its simplicity and effectiveness, was fine-tuned on the e-commerce dataset to adapt its features to the specific classification task.
   
4. **Fine-tuned ResNet50**: ResNet50, renowned for its deep architecture and skip connections, was fine-tuned to leverage its powerful feature extraction capabilities for e-commerce product classification.
   
5. **Fine-tuned InceptionV3**: InceptionV3, characterized by its inception modules and efficient use of computational resources, was fine-tuned to exploit its intricate feature representations.
   
6. **Fine-tuned Vision Transformer (ViT)**: Leveraging the breakthrough transformer architecture, ViT was fine-tuned to explore its efficacy in handling image classification tasks.

Through rigorous experimentation and evaluation, the Vision Transformer (ViT) emerged as the top performer, achieving an impressive accuracy of 98%.

## Model Deployment

With the trained model in hand, the focus shifted towards deployment, ensuring seamless integration into real-world applications. The deployment process involved:

1. **Model Saving**: The trained model and its weights were saved, encapsulating the learned representations and parameters for future use.
   
2. **Validation and Testing**: The saved model weights were loaded into another notebook for validation and testing, ensuring the model's robustness and efficacy across diverse datasets.
   
3. **Deployment using Streamlit**: The model was deployed using Streamlit, a user-friendly framework for building interactive web applications. This facilitated easy access and utilization of the model for inference tasks.
   
4. **Inference Testing**: The deployed model was subjected to rigorous inference testing using diverse datasets, validating its performance and ensuring its readiness for production use.

## Additional Resources

To delve deeper into the project's implementation and results, the following resources are available:

1. [Code Explanation](https://drive.google.com/file/d/1WTjTWTTW2LntxyUWlAmbOi5sDvM4IaJS/view?usp=sharing)
2. [Notebook Testing and Validation Video](https://drive.google.com/file/d/1u6YczEk59BMh3lEek6Atr-KCqB6vAPLA/view?usp=sharing)
3. [Deployment Video](https://drive.google.com/file/d/1EkK_CCFc3tQZbK1TPSrDBXcCjJPMKsCC/view?usp=sharing)
4. [Dataset Link](https://drive.google.com/drive/folders/1YU7dHxsTx-agUchYeBOhOVMmlzUyuECC?usp=sharing)
   
These resources provide comprehensive insights into the project's methodology, experimental setup, and outcomes, facilitating a thorough understanding of the undertaken endeavor.

## Contributors

- Ahmed Sameh Ahmed Abdelaziz
