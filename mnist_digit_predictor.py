# Imports
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist['data'], mnist['target']

# Convert target `y` to numeric type
y = y.astype(np.int8)

# Split the dataset into training and test sets
x_train, x_test = x[:60000], x[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# Shuffle the training set
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# Create a binary detector for the label '2'
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)

# Train a Logistic Regression model
clf = LogisticRegression(tol=0.1, max_iter=1000)
clf.fit(x_train, y_train_2)

def predict_digit(some_digit):
    """Predict if the digit is a '2' using the trained model."""
    example = pd.DataFrame([some_digit])  # Convert to DataFrame
    prediction = clf.predict(example)
    return prediction[0]

def random_number_generator():
    """Generate and return a random number between 0 and 9."""
    return np.random.randint(0, 10)  # Generate a number between 0 and 9

def show_random_number_image(random_number):
    """Display an image of the randomly generated number."""
    # Create an empty image of 28x28 pixels
    img = np.zeros((28, 28))

    # Fill in the pixels to represent the random number
    digit_image_index = np.where(y == random_number)[0]
    if len(digit_image_index) > 0:
        digit_image = x[digit_image_index[0]].reshape(28, 28)
        plt.imshow(digit_image, cmap=plt.cm.binary, interpolation='nearest')
        plt.title(f'Randomly Generated Number: {random_number}')
        plt.axis("off")
        plt.show()
    else:
        print("No image found for the generated number.")

def main_menu():
    """Display a menu for the user to select actions."""
    while True:
        print("\nSelect an option:")
        print("1. Generate a random number and predict if it's '2'")
        print("2. Generate and show a random number")
        print("3. Exit")

        choice = input("Enter the number of your choice: ")

        if choice == '1':
            random_number = random_number_generator()
            print("Randomly Generated Number:", random_number)

            # Prepare the digit image for prediction
            digit_image_index = np.where(y == random_number)[0]
            if len(digit_image_index) > 0:
                some_digit = x[digit_image_index[0]]
                prediction = predict_digit(some_digit)
                print("Is it the number '2'? ->", prediction)
            else:
                print("No image found for the generated number for prediction.")

        elif choice == '2':
            random_number = random_number_generator()
            print("Randomly Generated Number:", random_number)
            show_random_number_image(random_number)
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select a valid option.")

# Run the main menu
if __name__ == "__main__":
    main_menu()
