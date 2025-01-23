# DEPRESSION DETECTION

**GAZI UNIVERSITY**
**FACULTY OF ENGINEERING**
**COMPUTER ENGINEERING DEPARTMENT**

**Burak ÇETİN**

**DEPRESSION DETECTION AND SUPPORT SYSTEM**

**CENG373 – INTRODUCTION TO MACHINE LEARNING**

## 1. ABSTRACT

The increasing prevalence of mental health issues, particularly depression and suicidal ideation, has raised significant societal concerns. This project aims to develop an automated system to detect depression and suicidal tendencies from social media text using advanced natural language processing (NLP) techniques and machine learning. By analyzing the linguistic patterns and contextual nuances of social media posts, the system seeks to classify texts into three categories: neutral, depressive, and suicidal.

The methodology incorporates a hybrid approach that combines transformer-based embeddings with psycholinguistic and sentiment analysis features. The model architecture integrates a pre-trained transformer (BERT) for contextual language understanding with additional features, such as sentiment scores and first-person pronoun usage, which are extracted using tools like VADER and regular expression analysis. The data preprocessing steps include cleaning, tokenization, and addressing class imbalance through oversampling techniques to ensure fair representation across all categories.

The project employs advanced training techniques, including mixed-precision training and hyperparameter optimization, to enhance performance and efficiency. The model's evaluation leverages metrics such as accuracy, precision, recall, F1 score, and AUC-ROC to measure its effectiveness. Initial results demonstrate the model's ability to outperform traditional baselines like logistic regression and SVMs, particularly in identifying depressive and suicidal text.

This work contributes to the growing field of AI applications in mental health by providing a robust, scalable, and ethically mindful framework for early detection and intervention. Beyond its immediate technical outcomes, the project underscores the potential of AI in creating positive societal impact and fostering greater awareness and understanding of mental health issues.

## 2. INTRODUCTION

### 2.1 Problem Statement

The advent of social media has created a platform where individuals freely express their thoughts and emotions. While this openness has significant benefits, it also poses challenges in identifying early signs of mental health struggles, such as depression and suicidal ideation. Studies reveal that individuals experiencing these mental health issues often display linguistic patterns and emotional cues in their social media posts that could serve as indicators of their psychological state [1, 2]. However, due to the sheer volume of data and the subtlety of these signs, manual monitoring is neither feasible nor efficient. This project addresses the critical need for an automated system capable of detecting depressive and suicidal tendencies from text, providing a scalable and effective solution to this growing concern.

### 2.2 Motivation

Mental health issues, including depression and suicide, are pressing societal challenges, with suicide ranking among the leading causes of death worldwide [3]. Early detection and intervention are essential to reducing these numbers and saving lives. Social media platforms offer a unique opportunity for early diagnosis, as they often reflect users' emotional and mental states. However, existing tools are limited in their ability to analyze linguistic and contextual nuances in social media posts. The development of an advanced model to detect these indicators can facilitate timely feedback, allowing intervention by mental health professionals or support systems. This project has the potential to make a significant societal impact by fostering awareness, encouraging proactive mental health care, and ultimately reducing the stigma associated with seeking help.

### 2.3 Objectives

The primary objective of this project is to develop an automated system capable of accurately classifying social media posts into three categories: neutral, depressive, and suicidal. By integrating state-of-the-art transformer models with psycholinguistic and sentiment analysis features, the system aims to enhance the detection of depressive and suicidal content. Specific goals include:

- Building a robust preprocessing pipeline to clean and prepare social media text data for analysis.
- Incorporating contextual language embeddings with linguistic and sentiment-based features to improve classification accuracy.
- Addressing data imbalance issues to ensure fair representation across all classes.
- Evaluating the system's performance using comprehensive metrics, including precision, recall, F1 score, and AUC-ROC, to ensure its reliability.
- Providing timely insights that could assist mental health professionals and organizations in early 

## 3. METHODOLOGY

### 3.1 Dataset Preparation

Dataset preparation was a foundational step in this project, ensuring that the data used for training and evaluation was of high quality and relevant to the project's objectives. Several datasets were utilized, each bringing unique contributions to the study:

1.	**Reddit Dataset (r/Depression and r/SuicideWatch):** This dataset includes posts from Reddit communities specifically focused on mental health topics like depression and suicide. The data is invaluable for identifying patterns in textual content that signify mental health struggles [8].
2.	**Depression Reddit Cleaned Dataset:** This dataset consists of pre-processed and labeled Reddit posts related to depression. It reduced the need for extensive preprocessing, providing a cleaner foundation for analysis and model training [9].
3.	**Suicide and Depression Detection Dataset:** Sourced from Hugging Face, this dataset contains labeled data for detecting depression and suicidal tendencies. It aligns well with the project's focus on classification and analysis of mental health-related text [10].
4.	**Reddit Conversations Dataset:** This dataset contains general conversations from Reddit, offering diverse context and linguistic patterns to improve the robustness of the model [11].
5.	**Daily Dialog Dataset:** This dataset contains daily conversations, taken from lukasgarbas' nlp-text-emotion repository on GitHub [12].

**Data Cleaning and Preprocessing**

To ensure the datasets were suitable for training deep learning models, the following steps were taken:
1.	**Stopword Removal:** Words that do not contribute significant semantic meaning, such as "the" and "and," were removed to enhance the relevance of the text.
2.	**Tokenization:** Text was tokenized using the BERT tokenizer, which breaks text into smaller components (tokens) while preserving semantic meaning. This step is crucial for deep learning models to process text effectively.
3.	**Text Normalization:** All text was converted to lowercase, and non-alphanumeric characters were removed to maintain uniformity.
4.	**Label Mapping:** Text labels (neutral, Depression, SuicideWatch) were mapped to numerical values (0, 1, 2) to facilitate compatibility with machine learning models.
5.	**Data Reduction:** There was about 400,000 - 450,0000 pieces of data in total. But training this amount of data was very time consuming. Therefore, training was performed with about 150,000 data using 1/3 of the data. After various preprocessing operations, there are about 42,000 data per label (neutral, depression and SuicideWatch).

**Balancing Class Distribution**

The datasets exhibited class imbalance, with neutral posts significantly outnumbering depressive and suicidal posts. To mitigate this issue, the following strategies were employed:
- **Oversampling:** Depressive and suicidal posts were oversampled to match the number of neutral posts. This technique involved duplicating existing samples in the minority classes [9, 10].
- **Shuffling:** After oversampling, the dataset was shuffled to randomize the distribution and minimize training bias [8].

**Ethical Consideration**s

Given the sensitive nature of the data, ethical guidelines were followed throughout the project. Only publicly available datasets were used, and all personally identifiable information was excluded to protect individuals' privacy. This aligns with best practices for ethical research in mental health and artificial intelligence [4, 8].

**Summary**

In conclusion, dataset preparation included sourcing data from reliable repositories, applying rigorous cleaning and preprocessing methods, and implementing strategies to address class imbalance. These steps established a solid foundation for the subsequent phases of model training and evaluation. 

### 3.2 Feature Extraction

Feature extraction played a pivotal role in this project by allowing us to leverage diverse linguistic, psychological, and contextual features for detecting depression and suicidal tendencies. This multifaceted approach ensured the model could capture both the explicit and subtle markers of mental health conditions in text.

**Sentiment Analysis**

To gauge the emotional tone of the text, we utilized VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis. VADER is highly effective for social media and short-text contexts because it assigns scores for positive, negative, and neutral sentiments, along with an overall "compound" score. These scores provided a nuanced view of emotional polarity in the text. For instance, depressive or suicidal texts often exhibited lower compound scores due to their predominantly negative sentiment patterns [2, 6].

In addition, TextBlob, another sentiment analysis tool, was considered for its ability to assign polarity and subjectivity scores. TextBlob's simplicity and effectiveness in analyzing emotional valence provided complementary insights to VADER's metrics, enriching the feature set for classification [4].

**Psycholinguistic Features**

Psycholinguistic features provided an essential layer of interpretability by quantifying linguistic markers associated with mental health. Specifically, the following features were extracted:
1.	First-Person Pronouns: Texts containing frequent first-person pronouns such as "I," "me," and "mine" often indicate self-focused thought processes, a common trait in individuals experiencing depressive or suicidal ideation [1, 5].
2.	Emotional Words: A lexicon of emotional terms (e.g., "sad," "happy," "depressed," "joy") was used to calculate the frequency of such words in each text. Depressive texts typically contained more negative emotional terms, while suicidal texts often included words expressing hopelessness or despair [2, 4].

By quantifying these features, the model could link specific psycholinguistic patterns with the likelihood of depression or suicidal tendencies, addressing the interpretability gap often associated with deep learning approaches [6].

**Transformer Embeddings**

Transformer-based embeddings, specifically from BERT, were used to capture the contextual and semantic richness of the text. BERT generates dense, high-dimensional representations for each word or token, enabling the model to discern subtle differences in meaning and intent [5, 6].

For instance, a suicidal text like "I feel like there's no way out" might carry the same literal meaning as a neutral text discussing problem-solving. However, BERT’s embeddings can distinguish the intent and emotional context behind such phrases, a capability critical for this task [4, 8].

The final embeddings were pooled to obtain a comprehensive representation of the text, which was subsequently fused with sentiment and psycholinguistic features. This fusion approach allowed the model to incorporate both context-aware features and interpretable linguistic markers, creating a robust framework for classification.

**Summary of Feature Integration**

The combination of sentiment scores, psycholinguistic markers, and transformer embeddings provided a comprehensive understanding of the text. While sentiment analysis captured the emotional tone, psycholinguistic features offered interpretability by highlighting specific linguistic patterns. Transformer embeddings complemented these by preserving context and meaning, enabling the model to excel in distinguishing between neutral, depressive, and suicidal texts.

By integrating these diverse feature types, the project not only enhanced classification accuracy but also addressed the limitations of relying solely on deep learning or rule-based approaches, offering a balanced and interpretable solution to a critical societal challenge.

### 3.3 Model Architecture

The architecture of the Fusion Model was designed to integrate the power of transformer-based contextual embeddings with handcrafted linguistic features, enabling effective classification of text into neutral, depressive, or suicidal categories. This hybrid design leverages pre-trained transformer models for their ability to capture semantic and syntactic nuances, while the additional features enhance interpretability by incorporating psycholinguistic and sentiment-based insights.

#### 3.3.1 Transformer Layer

The foundation of the Fusion Model is a pre-trained transformer, specifically the BERT model (bert-base-uncased). This layer provides context-aware embeddings for input text.
- **Functionality:**
  - Processes tokenized input (input_ids) and attention masks (attention_mask) to generate a sequence of hidden states.
  - Extracts a fixed-length embedding from the [CLS] token, representing the entire input text's semantic meaning.
  - The transformer layer's parameters are fine-tuned during training to adapt to the specific task.

#### 3.3.2 What is BERT?

Traditional language models often rely on unidirectional context, either processing text from left-to-right or right-to-left. This approach limits the ability to fully utilize surrounding context for understanding words within sentences. BERT addresses this limitation by employing a bidirectional training mechanism, enabling the model to learn context from both the left and right sides of a word simultaneously [13].

**Key Innovations:**
- Deep Bidirectional Training: Unlike models such as OpenAI GPT, BERT uses a "masked language model" (MLM) objective, allowing the representation to incorporate context from all directions.
- Fine-Tuning Versatility: With minimal task-specific architecture modifications, BERT can be fine-tuned for a wide array of NLP tasks, such as question answering, sentiment analysis, and named entity recognition (NER).

##### 3.3.2.1 Architecture of BERT

BERT's architecture is based on the Transformer encoder introduced by Vaswani et al. (2017). It uses multiple layers of self-attention and feed-forward neural networks to capture relationships between words in a sentence [13].

**Model Configurations:**
- BERTBASE: 12 layers, 768 hidden units, 12 attention heads, 110M parameters.
- BERTLARGE: 24 layers, 1024 hidden units, 16 attention heads, 340M parameters.

**Input Representation:**

BERT’s input representation unifies single sentences and sentence pairs into a token sequence. Each sequence starts with a special token [CLS] and uses [SEP] to separate sentences. Additionally, token, segment, and position embeddings are combined to provide comprehensive context [13].


![image](https://github.com/user-attachments/assets/c4a77c6a-6315-42d0-90c4-90769985b517)

*Figure 3.3.2.1.1: BERT*

##### 3.3.2.2. Pre-Training Objectives

BERT is pre-trained on vast amounts of unlabeled text using two key tasks [13]:

**Masked Language Model (MLM):**
-A percentage of input tokens (typically 15%) are randomly masked.
- The model predicts the original tokens based on the bidirectional context.
- This approach forces the model to develop a deep understanding of language structure.

**Next Sentence Prediction (NSP):**
- BERT is trained to predict whether one sentence logically follows another.
- This task aids in understanding relationships between sentences, crucial for applications like question answering.

##### 3.3.2.3. Fine-Tuning

Fine-tuning BERT involves adding a task-specific output layer and training the entire model end-to-end on labeled data. For example [13]:
- Question Answering: Predicting start and end tokens of an answer in a passage.
- Text Classification: Using the [CLS] token's embedding for tasks like sentiment analysis.
- Sequence Labeling: Leveraging token-level embeddings for tasks like NER.

![image](https://github.com/user-attachments/assets/48a5a83f-c293-4fa4-9208-b6c66578ad80)

*Figure *.3.2.3.1: Fine-Tuning BERT*

**Fusion Layer**

The fusion layer is designed to combine the transformer’s output with additional handcrafted features, creating a comprehensive representation for classification.
- Input:
  - Transformer output (hidden state of the [CLS] token): A dense vector of size 768 (the hidden size of BERT).
  - Additional features: Psycholinguistic metrics (e.g., first-person pronouns and emotional word counts) and sentiment scores (positive, negative, neutral, and compound).
- Operation:
  - Concatenates the transformer output and additional features.
  - Passes the concatenated vector through a fully connected layer to reduce dimensionality and create a fused representation.

**Classifier**

The classifier is a multi-layer feedforward neural network designed to map the fused representation to the output classes.
- **Architecture:**
  - Input Layer: Accepts the output from the fusion layer.
  - Hidden Layers: Two fully connected layers interleaved with ReLU activation functions and dropout layers to prevent overfitting.
  - Output Layer: A final linear layer with three neurons (corresponding to neutral, depressive, and suicidal classes). The output is passed through a softmax function to generate class probabilities.

**Overall Workflow**

1.	**Input Data:**
  - Tokenized text (input_ids and attention_mask) is passed to the transformer layer.
  - Additional features are extracted using sentiment analysis (VADER) and psycholinguistic analysis.
2.	**Transformer Output:**
  - The hidden state of the [CLS] token is extracted as the transformer output.
3.	**Fusion:**
  - Transformer embeddings are concatenated with the additional features.
  - The concatenated vector is transformed into a fused representation by the fusion layer.
4.	**Classification:**
  - The fused representation is passed through the classifier to predict the probability distribution over the three classes.

**Model Summary**

The architecture efficiently integrates deep semantic features from the transformer with handcrafted linguistic insights, ensuring a holistic understanding of input text. This combination allows the model to identify subtle linguistic cues indicative of depression or suicidal ideation while maintaining the robustness of transformer-based representations.

**Advantages of the Architecture**
- Contextual Understanding: Leveraging BERT’s embeddings ensures that the model captures contextual relationships between words.
- Enhanced Features: Incorporating additional features improves performance, especially when specific linguistic patterns are not prominent in transformer embeddings.
- Flexibility: The modular structure of the model allows easy integration of other pre-trained transformers or additional features if needed in the future.

By integrating deep learning with domain-specific handcrafted features, the Fusion Model bridges the gap between advanced NLP techniques and interpretable linguistic cues, ensuring effective classification of mental health indicators.

### 3.4 Training and Hyperparameter Tuning
The training and hyperparameter tuning process combines state-of-the-art techniques such as AdamW optimization, linear warm-up scheduling, and mixed-precision training to ensure efficient and accurate learning. Systematic tuning through grid search and cross-validation enhances the model's generalizability, making it robust in detecting depression and suicidal ideation across diverse text inputs.


## 4. IMPLEMENTATION

### 4.1 Tools and Technologies

The project leveraged a robust suite of tools and technologies, carefully chosen to meet the requirements of handling and processing natural language data, building deep learning models, and performing efficient training and evaluation. The following sections detail the tools and technologies used, along with their roles in the project.

**Programming Language**
- Python: Python was the core programming language for this project due to its simplicity, versatility, and the extensive ecosystem of libraries for data analysis, natural language processing (NLP), and deep learning. Python's readability and active community support make it the ideal choice for both research and production-level AI applications.

**Frameworks**
- PyTorch: PyTorch was selected as the primary deep learning framework due to its dynamic computation graph, which allows for flexible model design and debugging. Its strong integration with GPU acceleration made it ideal for training large models like transformers. PyTorch's extensive ecosystem also includes tools like torch.nn for defining neural network layers and torch.optim for optimization algorithms.
- Hugging Face Transformers: The Hugging Face Transformers library was critical for integrating state-of-the-art pre-trained models like BERT. This library simplifies the usage of transformer models by providing:
  - Pre-trained tokenizers and models.
  - Support for various transformer architectures such as BERT, RoBERTa, and GPT.
  - Tools for fine-tuning pre-trained models on custom datasets. Its modular design enabled easy experimentation and customization for the project's specific needs, such as incorporating additional psycholinguistic and sentiment features.

Libraries
1.	**Machine Learning and Data Analysis:**
    - scikit-learn: Scikit-learn was used for implementing K-Fold cross-validation, which ensured robust evaluation by splitting the dataset into multiple folds for training and testing. Additionally, it provided utilities for calculating evaluation metrics such as precision, recall, and F1-score.
    - NumPy and Pandas:
      - NumPy facilitated efficient numerical computations, especially for handling multi-dimensional arrays.
      - Pandas enabled data manipulation and preprocessing, such as tokenization, feature extraction, and cleaning textual data from datasets.
2.	**Visualization:**
    - Matplotlib: Matplotlib was used to plot training and validation performance metrics over epochs, such as loss and accuracy trends. These visualizations provided insights into the model's convergence and overfitting behavior.
    - Seaborn: Seaborn was used for advanced visualizations, such as plotting the distribution of labels in the dataset. It offered aesthetically pleasing and easily interpretable graphics, aiding in exploratory data analysis.
3.	**Sentiment Analysis:**
    - VADER Sentiment Analyzer: The VADER (Valence Aware Dictionary for Sentiment Reasoning) sentiment analysis tool was integrated to extract sentiment scores (positive, negative, neutral, and compound) from textual data. These scores added an additional layer of semantic information to the model's input features.
    - TextBlob (Optional): TextBlob, a Python library for processing textual data, was considered as a complementary tool for performing sentiment analysis and computing features like subjectivity and polarity scores.

**Hardware**
- Google Colab (T100 GPU): The project utilized Google Colab, equipped with a T100 GPU (40GB of memory), to accelerate model training. The GPU support was crucial for handling large transformer models and reducing training time significantly. Additionally, Colab's cloud environment provided an easily accessible platform for experimentation.

**Integration and Workflow**

The combination of these tools enabled the following workflow:
1.	**Data Preprocessing:** Libraries like Pandas, NumPy, and Hugging Face Tokenizers ensured that the textual data was cleaned, tokenized, and converted into formats suitable for deep learning models.
2.	**Feature Engineering:** Tools like VADER extracted meaningful sentiment scores, while custom functions captured psycholinguistic features such as emotional word counts.
3.	**Model Training:** PyTorch and Hugging Face simplified the definition and training of a custom fusion model that combined transformer embeddings with handcrafted features.
4.	**Evaluation and Visualization:** Scikit-learn, Matplotlib, and Seaborn provided utilities for rigorous model evaluation and clear visualization of results.

## 5. RESULTS

### 5.1 Training and Validation Performance

The training and validation performance of the BERT-based model was evaluated over a series of epochs. The process revealed trends in loss reduction and accuracy improvement, highlighting the effectiveness of the model training.

**Training and Validation Loss Over Epochs**

The training and validation loss curves exhibited a steady decline across all epochs, reflecting the model's ability to minimize error during optimization. Initially, the validation loss was higher than the training loss, which is expected due to the model's adjustment to unseen data. As training progressed, the gap between training and validation loss reduced, indicating improved generalization.

**Accuracy Trends During Training**

The accuracy trends observed during training showed consistent improvement, stabilizing at a high level by the final epochs. This indicates that the model effectively learned to classify the data across all classes (Neutral, Depressive, and SuicideWatch) while avoiding overfitting.

**Key Observations:**
1.	The training accuracy increased steadily across all epochs, achieving an optimal balance of precision and recall by the final epoch.
2.	Validation accuracy followed a similar trend, confirming that the model generalized well to unseen data.
3.	The final training accuracy reached approximately 0.9252, with a validation loss of 0.2318, suggesting robust performance after 5 epochs (I couldn’t generate graphs of 5 epochs training due to some time issues on Google Colab).

**Conclusion**

The training and validation performance demonstrates that the model successfully converged, achieving high accuracy and low validation loss. These results highlight the effectiveness of the BERT-based fusion model in learning both linguistic and contextual signals from the dataset, even with the inclusion of additional psycholinguistic and sentiment features.

### 5.2 Evaluation Metrics

Evaluation metrics are critical in assessing the performance of a machine learning model, particularly in a project as sensitive as detecting depressive and suicidal tendencies. These metrics provide insights into the model's capability to generalize and classify instances correctly. For this project, accuracy, precision, recall, F1 score, specificity, and AUC-ROC were used to evaluate the model's effectiveness, and confusion matrices were constructed to analyze class-level performance.

**Accuracy**

Accuracy is the overall correctness of the model, calculated as the ratio of correctly classified samples to the total number of samples. While accuracy provides a general sense of model performance, it can be misleading in cases of imbalanced datasets, which is why additional metrics were analyzed.

![image](https://github.com/user-attachments/assets/407dd023-0412-4ab2-a06a-11294bd2a0c3)

*Figure 5.2.2: Accuracy (Accuracy is nearly 0.92 after 5 epochs)*

**Formula:**
 *Accuracy=(True Positives (TP)+True Negatives (TN))/Total Samples*

**Precision**

Precision measures the accuracy of positive predictions for each class. In this project, it reflects the model’s ability to correctly classify depressive or suicidal posts without falsely labeling neutral posts.

![image](https://github.com/user-attachments/assets/2cfa011c-d2c2-475c-8b72-fe0317806e59)

*Figure 5.2.3: Precision-Recall-F1_Score*

**Formula:**
 *Precision=TP/(TP+False Positives (FP))*

**Recall (Sensitivity)**

Recall quantifies the model's ability to identify all relevant instances. For example, in the case of suicidal posts, it measures how many true suicidal posts were correctly identified by the model.

**Formula:**
 *Recall=TP/(TP+False Negatives (FN))*

**Specificity**

Specificity focuses on the model's ability to identify negative instances correctly. For neutral posts, it measures how well the model avoids falsely classifying depressive or suicidal posts as neutral.

![image](https://github.com/user-attachments/assets/def6e6f2-dc0c-47a0-ac76-726774ffde08)

*Figure 5.2.4: Specificity*

**Formula:**
 *Specificity=TN/(TN+FP)*

**F1 Score**

The F1 score is the harmonic mean of precision and recall, balancing these two metrics. It is particularly useful in imbalanced datasets, where high precision or recall alone is insufficient.

**Formula:**
 *F1 Score=2⋅(Precision⋅Recall)/(Precision+Recall)*

**AUC-ROC**

The Area Under the Receiver Operating Characteristic Curve (AUC-ROC) evaluates the model's ability to distinguish between classes across all classification thresholds. A higher AUC-ROC indicates better separability between classes.

![image](https://github.com/user-attachments/assets/87fdbe19-7b0c-45b7-9866-3cf1e446e75c)

*Figure 5.2.5: AUC-ROC*

**Confusion Matrix**

Confusion matrices were used to examine the performance for each class: neutral, depressive, and suicidal. Each matrix displays True Positives, False Positives, False Negatives, and True Negatives, providing a detailed view of classification errors.

![image](https://github.com/user-attachments/assets/64d1d360-9a0b-48f6-b9eb-8fa5fd713a16)

*Figure 5.2.6: Confusion Matrix*

**Analysis of Confusion Matrix**

**The confusion matrix revealed:**
- Neutral posts were often misclassified as depressive, likely due to shared linguistic patterns between neutral and depressive text.
- Suicidal posts had fewer misclassifications, likely because of distinct patterns and keywords associated with this class.

**Discussion**

The evaluation metrics highlight the model's effectiveness in classifying text into neutral, depressive, and suicidal categories. However, further improvements could involve:
- Enhancing recall for the depressive class, as some depressive posts were misclassified as neutral.
- Refining the features to improve specificity for the neutral class.
- Experimenting with alternative loss functions, such as focal loss, to handle class imbalances more effectively.

### 5.3 Comparative Analysis

To evaluate the performance of the BERT-based fusion model, its results were compared with baseline models, including Logistic Regression and Support Vector Machines (SVM). The comparison focused on key metrics such as accuracy, precision, recall, F1-score, specificity, and the macro-averaged AUC-ROC.

![image](https://github.com/user-attachments/assets/37d14461-3583-4b1c-9661-5896685f78a5)

*Figure 5.3.1: Comparison with Logistic Regression*

![image](https://github.com/user-attachments/assets/212248c3-6ee2-4121-9f4f-da3c35dff386)

*Figure 5.3.2: Comparison of Specificity by Class*

**Baseline Model Performance**

The Logistic Regression model, used as a baseline, achieved an accuracy of 82.08% and a macro-averaged AUC-ROC of 93.56%. While these results demonstrate the model's effectiveness in basic classification tasks, its inability to fully capture linguistic and contextual nuances is evident in its lower recall and F1-scores compared to the BERT-based model.

**Performance of the BERT-Based Fusion Model**

The BERT-based fusion model, which integrates transformer embeddings with psycholinguistic and sentiment features, achieved an accuracy of 83.68% and a macro-averaged AUC-ROC of 94.79%. Notably, the model performed better in distinguishing between depressive and suicidal content, as reflected by its higher F1-score and recall for these classes.

**Key Metric Comparison**

![image](https://github.com/user-attachments/assets/7a0c8cf3-cb1c-4f69-ac2f-ef44e6d3dc26)

**Confusion Matrix Comparison**

The confusion matrix for the BERT model showed a more balanced performance across classes. For example:
- **Neutral Class:** High precision (95.75%) and specificity (97.82%), outperforming the baseline models.
- **Depressive Class:** Improved recall (73.68%) over Logistic Regression, indicating better sensitivity in identifying depressive posts.
- **SuicideWatch Class:** Higher F1-score (77.38%) compared to the baseline, demonstrating the model's enhanced ability to identify suicidal content.

**Analysis and Insights**
1.	Outperformance: The BERT model outperformed Logistic Regression and SVM across all key metrics. Its integration of transformer embeddings allowed it to capture the contextual and semantic nuances of the text more effectively than the baseline models.
2.	Complex Features: The inclusion of psycholinguistic and sentiment features further boosted its performance, particularly in distinguishing subtle differences between depressive and suicidal posts.
3.	Enhanced Generalization: The higher AUC-ROC and F1-scores demonstrate the model's ability to generalize well to unseen data while maintaining class-specific performance.

**Conclusion**
The comparative analysis confirms that the BERT-based fusion model significantly outperforms the baseline models. Its ability to incorporate linguistic, contextual, and additional psycholinguistic features makes it a superior choice for detecting depressive and suicidal ideation in social media posts. This highlights the importance of advanced architectures in tackling complex mental health-related text classification tasks.

## 6. CONCLUSION

This project aimed to create a robust text classification system capable of detecting depressive and suicidal ideation in social media posts. By leveraging state-of-the-art techniques in natural language processing (NLP) and machine learning, the project successfully demonstrated the effectiveness of a BERT-based fusion model in addressing this critical societal challenge.

**Key Findings and Achievements**
- **Model Performance:** The BERT-based fusion model achieved a high accuracy of 83.68% and a macro-averaged AUC-ROC of 94.79%, outperforming traditional baseline models such as Logistic Regression and Support Vector Machines (SVM).
- **Class-Specific Insights:** The model showed notable performance in distinguishing between neutral, depressive, and suicidal posts, with precision and recall scores consistently higher than the baselines. This highlights the model's ability to identify subtle nuances in mental health-related language.
- **Feature Integration:** The integration of transformer embeddings with psycholinguistic and sentiment features enhanced the model's ability to capture both contextual and emotional aspects of the text.
- **Balanced Dataset:** The use of advanced resampling techniques ensured a balanced dataset, reducing bias and improving the model's generalizability across classes.

**Societal Impact and Future Applications**
- **Mental Health Support:** This project underscores the potential of machine learning to assist mental health professionals by providing early warnings for depressive or suicidal tendencies based on social media activity. This could lead to timely interventions and potentially save lives.
- **Scalability:** The model can be scaled and adapted for other languages, demographics, or social media platforms, broadening its reach and impact.
- **Integration in Support Systems:** Future implementations could integrate the model into helpline systems, mobile apps, or social media platforms to provide real-time monitoring and support for at-risk individuals.

**Reflection on Success**

The project successfully met its objectives of designing and implementing a reliable classification system for detecting mental health signals in text. Despite challenges such as data imbalance and training complexity, the final model demonstrated exceptional performance and practical applicability. This project not only contributes to the growing field of AI-driven mental health detection but also lays a strong foundation for future research and development in this area.

**Future Directions**

To further enhance the project's impact:
1.	**Advanced Linguistic Features:** Incorporating features like sarcasm detection and emotion classification could improve the model's ability to handle complex text patterns.
2.	**Real-Time Deployment:** Implementing the model in real-world settings with real-time feedback mechanisms could make it more actionable.
3.	**Ethical Considerations:** Addressing ethical concerns around data privacy and model biases will be critical to building trust and ensuring responsible use of the technology.

In conclusion, this project demonstrated the transformative potential of AI in tackling mental health challenges, providing a valuable tool for both researchers and practitioners. By bridging the gap between technology and mental health, it paves the way for impactful advancements in this critical domain.

## 7. REFERENCES

[1] Al-Mosaiwi, M., & Johnstone, T. (2018). In an absolute state: Elevated use of absolutist words is a marker specific to anxiety, depression, and suicidal ideation. Clinical Psychological Science, 6(4), 529-542.

[2] Sawhney, R., Joshi, H., & Gandhi, U. (2020). A computational approach to detecting and analyzing suicidal tendencies on social media. Social Network Analysis and Mining, 10(1), 1-12.

[3] World Health Organization. (2021). Suicide. Retrieved from https://www.who.int/news-room/fact-sheets/detail/suicide

[4] Li, T. M. H., Chen, J., Law, F. O. C., Li, C., Chan, N. Y., Chan, J. W. Y., Chau, S. W. H., Liu, Y., Li, S. X., Zhang, J., Leung, K., & Wing, Y. (2023). Detection of suicidal ideation in clinical interviews for depression using natural language processing and machine learning: Cross-sectional study. JMIR Medical Informatics, 11, e50221. https://doi.org/10.2196/50221

[5] Tadesse, M. M., Lin, H., Xu, B., & Yang, L. (2019). Detection of depression-related posts in Reddit social media forum. IEEE Access, 7, 44883–44893. https://doi.org/10.1109/access.2019.2909180

[6] Benton, A., Mitchell, M., & Hovy, D. (2017). Multi-task learning for mental health using social media text. Proceedings of the 15th Conference of the EACL, 152–162. https://doi.org/10.48550/arXiv.1712.03538

[7] Seah, J. H. K., & Shim, K. J. (2018). Data mining approach to the detection of suicide in social media: A case study of Singapore. 2018 IEEE International Conference on Big Data (Big Data), Seattle, WA, December 10-13: Proceedings, 5442-5444. https://doi.org/10.1109/BigData.2018.8622528 

[8] Reddit Dataset (r/Depression and r/SuicideWatch). Retrieved from https://www.kaggle.com/datasets/xavrig/reddit-dataset-rdepression-and-rsuicidewatch

[9] Depression Reddit Cleaned Dataset. Retrieved from https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned

[10] Suicide and Depression Detection Dataset. Retrieved from https://huggingface.co/datasets/joshyii/suicide_depression_detection

[11] Reddit Conversations Dataset. Retrieved from https://www.kaggle.com/datasets/jerryqu/reddit-conversations

[12] Lukasgarbas. (n.d.). nlp-text-emotion/data/datasets/dailydialog.csv at master · lukasgarbas/nlp-text-emotion. GitHub. Retrieved from https://github.com/lukasgarbas/nlp-text-emotion/blob/master/data/datasets/dailydialog.csv 

[13] Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1810.04805 


# CONTACT
- **Name:** Burak Çetin
- **Mail:** brakcetin660@gmail.com
