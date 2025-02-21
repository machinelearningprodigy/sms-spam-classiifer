


# ✉️📱 Email/SMS Spam Classifier  

A **Machine Learning-powered** Email and SMS spam classification app built using **Python** and **Streamlit**. 🚀  

## 🔍 Overview  

This project classifies messages as either **"Spam"** or **"Not Spam"** based on their content. It utilizes **Natural Language Processing (NLP)** techniques to preprocess text before making predictions using a pre-trained model.  

## 📂 Project Structure  

The project consists of the following key components:  

- **🎨 Streamlit Application** – A user-friendly interface for entering and classifying messages.  
- **📝 Text Preprocessing** – Cleans and processes input text using NLP techniques.  
- **🤖 Machine Learning Model** – A trained model that predicts whether a message is spam or not.  
- **📊 Vectorizer** – Converts text data into a numerical format (Bag of Words) for processing.  

## ⚙️ How It Works  

1. **📝 Input:** The user enters a message (SMS or Email) into the provided text box in the Streamlit app.  
2. **🔄 Preprocessing:** The text undergoes:  
   - Lowercasing  
   - Tokenization  
   - Removal of non-alphanumeric characters & stopwords  
   - Stemming using the **Porter Stemmer**  
3. **🔢 Vectorization:** The cleaned text is transformed into a numerical format using a **pre-trained CountVectorizer**.  
4. **🤖 Prediction:** The vectorized text is fed into the model, which classifies it as either:  
   - **📩 Spam** – The message is likely spam.  
   - **✅ Not Spam** – The message is not spam.  
5. **📌 Output:** The result is displayed on the Streamlit app.  

## 🚀 Example Usage  

1. **Run the Streamlit app:**  
   ```bash
   streamlit run app.py
   ```  
2. Enter a message in the provided text area.  
3. Click the **"Predict"** button to check if the message is spam or not.  

### ✨ Try These Example Messages:  

✅ `"Hey, are we still on for dinner tonight?"`  
📩 `"Congratulations! You've won a free ticket to the Bahamas. Call now!"`  

## 📦 Dependencies  

To install required dependencies, run:  

```bash
pip install streamlit scikit-learn nltk
```  

Additionally, NLTK data packages `punkt` and `stopwords` need to be downloaded.  

## 📁 Files  

- **`app.py`** – The main script that runs the Streamlit app.  
- **`model.pkl`** – The pre-trained machine learning model for spam classification.  
- **`vectorizer.pkl`** – The pre-trained CountVectorizer for text transformation.  

## 🎯 Model Training  

The model was trained on a labeled dataset of SMS messages using common text classification techniques, including:  

- **Text Preprocessing** – Cleaning, tokenization, and stemming.  
- **Vectorization** – Converting text into a numerical format using **Bag of Words**.  
- **Model Selection** – A machine learning classifier was trained and optimized for accurate predictions.  

## 🎉 Conclusion  

This project showcases the power of **NLP** and **machine learning** in identifying spam messages. The **Streamlit app** provides a simple interface for testing the classifier with real-world examples.  

💡 **Feel free to explore, contribute, or extend this project. Happy coding!**  

## 📜 License  

This project is licensed under the **MIT License** – see the `LICENSE` file for details.  
