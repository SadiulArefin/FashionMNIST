import tensorflow as tf
from data_analysis import x_test,x_train,y_test,y_train

def load_model():
    return tf.keras.models.load_model('fashion_mnist_model.h5')


model = load_model()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

with open('output.txt', 'w') as f:
    f.write(f'Model\'s architecture summary:\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write(f'\nEvaluation metric (Accuracy): {test_acc}\n')

print("Modle evaluation completed and output.txt generated")

    
