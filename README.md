# tec-CHAT

### Natural Language Processing 

NLP code in NINT to run on the Tec1 using 
- https://github.com/SteveJustin1963/tec-SPEECH
- https://github.com/SteveJustin1963/tec-EAR

In 1966, program ELIZA was developed, which aimed at tricking it users by making them believe that they were having a conversation with a real human being. ELIZA was designed to imitate a therapist who would ask open-ended questions and even respond with follow-ups. It responds to questions with answers that sound like they were provided by a therapist. The program is designed to mimic human conversation, and it is often used to gull people into thinking they are talking to a real person. HAT DOES ELIZA DO? Using "'pattern matching" and substitution methodology, the program gives canned responses that made early users feel they were talking to someone who understood their input. The program was limited by the scripts that were in the program. (ELIZA was originally written in MAD-
Slip.) 


### what is NLP?
" Natural language processing (NLP) is a process of understanding texts, speeches,
and similar are used by computerized systems..(Chowdhury, 2003, p. 51). 
The Oxford Dictionary defines NLP as “...computational techniques to the analysis and synthesis of natural
language and speech” (Natural Language Processing, 2017). 
The main goal of...to realize a human-like language processing for several tasks or applications and to analyze the generated texts with computational techniques (Liddy, 2010, p. 3864)."

### what is NLP text  

NLP text is a type of text that can be processed by natural language processing algorithms. There are many different algorithms used for natural language processing, but some of the most common ones include 
- part-of-speech Tagging
- Parsing
- Sentence segmentation
- Tokenization
- Stemming
- Lemmatization

## Most advanced NLP is GPT-3, T5, Transformers

They are out fo reach for the tec1... 

GPT-3 is a computer system that is designed to generate human-like responses to questions. It is based on a neural network that has been trained on a large amount of data. The system is designed to generate responses that are similar to what a human would say. GPT-3 is the third generation GPT Natural Language Processing model created by OpenAI. It is the size that differentiates GPT-3 from its predecessors. The 175 billion parameters of GPT-3 make it 17 times as large as GPT-2. It also turns GPT-3 about ten times as large as Microsoft's Turing NLG model. [6]

This website [3] provides a simple way to generate text using the T5 transformer. You simply provide a prompt, and the transformer will generate text based on the prompt. The text generation is based on the T5 transformer, which is a neural network model that is trained on a large amount of text data. The transformer learn the relationships between the words in the text data, and can then generate new text that is similar to the training data. The T5 transformer is a neural network model that is trained on a large amount of text data. The transformer learn the relationships between the words in the text data, and can then generate new text that is similar to the training data.and can generate text on its own.it can also improve the quality of the text it generates by fine-tuning the model on specific domains or tasks and can generate text in multiple languages and multiple genres, that is both grammatically correct and fluent. 

Google is using the T5 text-to-text transfer learning algorithm to improve the performance of its natural language processing models. The T5 algorithm can be used to fine-tune a model to a specific task, and this can improve the performance of the model on that task. For example, the T5 algorithm was used to improve the performance of a Google Translate model. The T5 algorithm can also be used to improve the performance of other NLP tasks, such as question answering and text classification. The T5 algorithm is a powerful tool for transfer learning, and it can be used to improve the performance of many different types of NLP models. [4]

Text-to-text transfer transformer (T5) is a new approach for natural language understanding (NLU) that can be used to read and comprehend any text. It was developed by Google Research and is based on the transformer architecture. T5 is trained on a large amount of text data in order to learn the general knowledge about the world. This allows it to transfer that knowledge to any text, regardless of the domain or the task. For example, T5 can be used to generate summaries of news articles, generate descriptions of images, or answer questions based on a passage of text. T5 has shown promising results on a variety of NLU tasks, including question answering, text classification, and text generation. In addition, T5 is efficient to train and can be used on a variety of hardware platforms, including CPUs, GPUs, and TPUs. Overall, T5 is a promising new approach for NLU that has the potential to revolutionize the field.[5]


We for computers is ASCII code, eg letter ‘A’ is 65. ASCII is the traditional encoding system, which is based on the English characters. Collection of such characters is generally referred to as token in NLP. The easiest way to represent any text in an NLP pipeline is by one-hot encoded vector representation. If a sentence contains a certain word then the corresponding entry in the vector is represented as “1” otherwise it’s “0”. For example, let’s consider the following two sentences:

![image](https://user-images.githubusercontent.com/58069246/167066937-d52dd4c2-fe07-49ce-ab7c-95c7e7eec14f.png)

![image](https://user-images.githubusercontent.com/58069246/167066963-46192da6-9aa4-4ae1-8d7c-055b1a86fdaa.png)

representation words as binary vectors...every word is represented with zero except the current word with 1. a thousand unique words require a vector of 1000. NLP systems have a dictionary of words, where each word is given an integer representation. So the phrase “natural language processing is the best field!” could be represented as “2 13 6 34 12 22 90”. Text preprocessing is the process of cleaning and standardizing text data before it can be used for further analysis. This typically involves removing unwanted characters, such as punctuation and whitespace, and converting the text to a uniform format, such as lowercase. Text normalization is the process of transforming text into a canonical form that can be used for further processing. The goals of text normalization include reducing dimensionality, noise, and ambiguity, and improving the interpretability of text. Tokenization is the process of converting a piece of text into a list of words or special characters which have meaning in natural language. Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. They are a dense vector representation of words that captures semantic relationships between words. Word embeddings can be used to improve the performance of many NLP tasks such as text classification, part-of-speech tagging, named entity recognition, and machine translation. Models in machine learning pipeline are a sequence of mathematical operations, which can learn to estimate the output on unseen data.
Selecting your model is very crucial, as it can decide the performance of your system. different layers make up a model. The input layer is used to represent  a tensor (tensor is a generalization of vectors and matrices and is easily understood as a multidimensional array), which is then fed into the embedding layer,  that performs a dot product on the input sequence and embedding matrix, which converts each word index into a corresponding embedding vector. The dropout layer randomly drops the input by a given percentage, which helps to prevent overfitting. this is fed into a A recurrent neural network is a type of neural network that is designed to handle sequential data. This type of network is well suited for tasks such as speech recognition and language translation.
Transfer learning is the improvement of learning in a new task by make use of previous knowledge in our NLP system. popular method is Word Embeddings.
Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation.
They are a distributed representation for text that is perhaps one of the key breakthroughs for the impressive performance of deep learning models on challenging natural language processing problems.
For example, the word “bank” would have a similar vector representation as the word “river” since they are both related to water.
The key idea is that these word vectors are learned in a way that captures semantic relationships between words.
This is different from one-hot encoded vectors, which assigns a vector to each word where all elements are zero except for the element corresponding to the index of the word, which is set to 1.
One-hot encoded vectors are useful for representing words for classification tasks, but they don’t capture relationships between words.
Word embeddings are often used as the input layer for deep learning models.
They can be learned from scratch or pre-trained on large datasets such as Google’s word2vec.
Pre-trained word embeddings are typically used because they can be trained on much larger datasets than what is available for most NLP tasks. 

### Project

We have https://github.com/SteveJustin1963/tec-SPEECH that allows us to make phonetic sounds, but we have to write the sound strings to drive the chip. We need a very simple ai system. We could write code to call an API from a paid site like OpenAI.com to do the hard lifting, but thats not fun for the Z80, what NLP text generators are there for 8 bit computers?  We may even need Speech recognitionto convert audio into text, save typing.


## Ref 
1. https://en.wikipedia.org/wiki/History_of_natural_language_processing
2. https://en.wikipedia.org/wiki/ELIZA
3. https://web.njit.edu/~ronkowit/eliza.html
4. https://towardsdatascience.com/poor-mans-gpt-3-few-shot-text-generation-with-t5-transformer-51f1b01f843e
5. https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
6. https://github.com/google-research/text-to-text-transfer-transformer
7. https://www.itbusinessedge.com/development/what-is-gpt-3/
8. https://towardsai.net/p/nlp/natural-language-processing-nlp-with-python-tutorial-for-beginners-1f54e610a1a0
9. https://towardsdatascience.com/introduction-to-natural-language-processing-for-noobs-8f47d0a27fcc
10. https://hackaday.com/tag/tty/, https://www.youtube.com/watch?v=OYJti8dJMV4
11. https://en.wikipedia.org/wiki/MegaHAL
12. https://en.wikipedia.org/wiki/Hidden_Markov_model
13. 






