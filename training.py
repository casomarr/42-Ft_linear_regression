import numpy as np #library used for computations
import matplotlib.pyplot as plt #library used for data visualization
import json #to store the values of thetas to be used in prediction.py


def load_data():
	'''This function loads the csv dataset.'''
	data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
	mileage = data[:,0]
	price = data[:,1]
	return mileage, price

def normalize_data(mileage, price):
	'''This function normalizes the data to have a mean of 0 and a standard deviation of 1.
	This is done to speed up the convergence of the gradient descent.'''
	mileage = (mileage - np.mean(mileage)) / np.std(mileage)
	price = (price - np.mean(price)) / np.std(price)
	return mileage, price

def gradient_descent(mileage, price):
	'''This function performs a gradient descent on the dataset.'''
	curr_loss = 1e9 #initialised with a high value
	prev_loss = 0.0
	learning_rate = 0.01 #learning rate
	epsilon = 0.0001 #convergence threshold
	iteration = 0
	max_iterations = 1000 #in case epsilon is not reached
	w = 0 #thetas[1] -> w
	b = 0 #thetas[0] -> b

	while(abs(curr_loss - prev_loss) > epsilon and iteration < max_iterations): #we use the absolute value 
		#of the difference so that if it becomes negative (and is therefore < epsilon) we don't exit the loop

		#we update the previous loss for the while loop condition
		prev_loss = curr_loss
		
		# we compute the prediction and the gradients
		predictions = w * mileage + b
		gradient_w = np.sum((predictions - price) * mileage) / len(price)
		gradient_b = np.sum(predictions - price) / len(price)

		#we update the thetas b and w
		w = w - learning_rate * gradient_w
		b = b - learning_rate * gradient_b

		#we update the prediction with the new thetas and compute the loss
		predictions = w * mileage + b
		curr_loss = np.sum((predictions - price) ** 2) / (2 * len(price))

		iteration += 1

	return [b, w]

def de_normalize_data(mileage, price, thetas):
	thetas[1] = thetas[1] * (np.std(price) / np.std(mileage))
	thetas[0] = thetas[0] * np.std(price) + np.mean(price) - thetas[1] * np.mean(mileage)
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
	with open('thetas.json', 'w') as f:
		json.dump(thetas, f)
	print("Thetas saved to thetas.json")


def main():
	mileage, price = load_data()
	normalized_mileage, normalized_price = normalize_data(mileage, price)
	normalized_thetas = gradient_descent(normalized_mileage, normalized_price)
	thetas = de_normalize_data(mileage, price, normalized_thetas)
	plot_data(mileage, price, thetas)
	save_thetas(thetas)#saves thetas to a json file to be used in prediction.py




	


if __name__ == "__main__":
	main()


#https://www.youtube.com/watch?v=rcl_YRyoLIY
#https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/