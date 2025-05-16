import numpy as np #library used for computations
import matplotlib.pyplot as plt #library used for data visualization
import json #to store the values of thetas to be used in prediction.py


#The second program will be used to train your model. It will read your dataset file
#and perform a linear regression on the data.

def load_data():
	'''This function loads the csv dataset.'''
	# Load the dataset
	data = np.loadtxt("data.csv", delimiter=",", skiprows=1) #skipping the first row that holds the titles
	mileage = data[:,0]
	price = data[:,1]
	return mileage, price

def normalize_data(mileage, price):
	'''This function normalizes the data.'''
	# Normalize the data
	mileage = (mileage - np.mean(mileage)) / np.std(mileage) #standardization
	price = (price - np.mean(price)) / np.std(price) #standardization
	return mileage, price

def gradient_descent(mileage, price):
	'''This function performs a gradient descent on the dataset.'''

	#initialize variables
	curr_loss = 1e9 #initialised with a high value
	prev_loss = 0.0
	learning_rate = 0.01 #learning rate
	epsilon = 0.0001 #Seuil pour la convergence
	iteration = 0
	max_iterations = 1000 #in case epsilon is not reached
	# thetas = Static.get_thetas()
	w = 0 #thetas[1] -> w
	b = 0 #thetas[0] -> b

	while(abs(curr_loss - prev_loss) > epsilon and iteration < max_iterations): #absolute value of the difference so that if it becomes negative we don't exit the loop
		prev_loss = curr_loss

		# Compute predictions
		predictions = w * mileage + b

		# Compute gradients
		gradient_w = np.sum((predictions - price) * mileage) / len(price)
		gradient_b = np.sum(predictions - price) / len(price)

		# Update thetas
		w = w - learning_rate * gradient_w
		b = b - learning_rate * gradient_b

		# Compute the loss AFTER updating predictions
		predictions = w * mileage + b  # Update predictions with new thetas
		curr_loss = np.sum((predictions - price) ** 2) / (2 * len(price))
		# print(f"Iteration {iteration}: Loss = {curr_loss}, Thetas = [{b}, {w}]")

		iteration += 1

	# Static.set_thetas([b, w])
	return [b, w]

def de_normalize_data(mileage, price, thetas):
	thetas[1] = thetas[1] * (np.std(price) / np.std(mileage))
	thetas[0] = thetas[0] * np.std(price) + np.mean(price) - thetas[1] * np.mean(mileage)
	# Static.set_thetas([thetas[1], thetas[0]])
	return thetas

def plot_data(mileage, price, thetas):
	plt.scatter(mileage, price, color='blue', label='Data Points')
	regression_line = thetas[1] * mileage + thetas[0]
	plt.plot(mileage, regression_line, color='red', label='Regression Line')
	plt.xlabel("Mileage")
	plt.ylabel("Price")
	plt.title("Price of a car for a given Mileage")
	plt.show()

def save_thetas(thetas):
	'''This function saves the thetas to a json file.'''
	# thetas = Static.get_thetas()
	with open('thetas.json', 'w') as f:
		json.dump(thetas, f)
	print("Thetas saved to thetas.json")


def main():
	mileage, price = load_data()
	normalized_mileage, normalized_price = normalize_data(mileage, price)
	normalized_thetas = gradient_descent(normalized_mileage, normalized_price)
	thetas = de_normalize_data(mileage, price, normalized_thetas)
	plot_data(mileage, price, thetas)
	save_thetas(thetas)#save thetas to a json file to be used in prediction.py




	


if __name__ == "__main__":
	main()


#https://www.youtube.com/watch?v=rcl_YRyoLIY
#https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/