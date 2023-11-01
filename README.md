
# Fashion MNIST 

The goal of this project is to classify images of apparel from the Fashion MNIST dataset into sustainable and non-sustainable categories. 

## Model

A Convolutional Neural Network (CNN) model is developed using TensorFlow and Keras. The CNN is chosen for its effectiveness in image classification tasks.

## Instructions to Run the Code:

1. Clone or download the project:
Use `git clone` if you're cloning a repository:

```bash
git clone https://github.com/SadiulArefin/FashionMNIST.git
```

2. Environment setup:

 Create a virtual environment:

```bash
python3 -m venv venv
```

* Activate the virtual environment:
* Windows: `.\venv\Scripts\activate`
* Unix/MacOS: `source venv/bin/activate`

Install dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

3. Script execution:

 Run the `evaluate_model.py` script:

```bash
python codes/evaluate_model.py

```

4. Output:

After running the script, you will find an `output.txt` file  containing the model summary, evaluation metric.
