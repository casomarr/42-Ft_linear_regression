import numpy as np #library used for computations
import matplotlib.pyplot as plt #library used for data visualization



#The second program will be used to train your model. It will read your dataset file
#and perform a linear regression on the data.

# def static():
# 	'''This function mimicks a static variable in C by retaining thetas' value across calls.'''
# 	if not hasattr(static, "thetas"):
# 		# hasattr(object, name) checks if an object has an attribute with 
# 		# the specified name. It returns True if the attribute exists, and 
# 		# False otherwise. It ensures thetas will be initialized only once.
# 		static.thetas = [0,0] #thetas[0] -> b & thetas[1] -> w
# 	return static.thetas
#
#static.thetas[1]

class Static:
	'''This class mimics a static variable in C by retaining thetas' value across calls.'''
	thetas = [0, 0] #thetas[0] -> b & thetas[1] -> w

	@staticmethod
	def get_thetas():
		return Static.thetas

	@staticmethod
	def set_thetas(thetas):
		Static.thetas = thetas

def load_data():
	'''This function loads the csv dataset.'''
	# Load the dataset
	data = np.loadtxt("data.csv", delimiter=",")
	mileage = data[:,0]
	price = data[:,1]
	# Normalize the data
	mileage = (mileage - np.mean(mileage)) / np.std(mileage) #standardization
	price = (price - np.mean(price)) / np.std(price) #standardization
	return mileage, price

def gradient_descent(mileage, price):
	'''This function performs a gradient descent on the dataset.'''

	#initialize variables
	curr_loss = 1e9 #initialised with a high value
	prev_loss = 0.0
	thetas = Static.get_thetas()
	learning_rate = 0.01 #learning rate
	epsilon = 0.01; #Seuil pour la convergence


	while((curr_loss - prev_loss) > epsilon):
		prev_loss = curr_loss
		# Compute the loss
		curr_loss = np.sum((thetas[1] * mileage + thetas[0] - price) ** 2) / (2 * len(price))
		# Compute the gradient
		w1 = thetas[1] - learning_rate * ()
		b1 = thetas[0] - learning_rate * dérivée
		Static.set_thetas([b1, w1]) # Update the thetas
		




	Static.set_thetas(thetas)




def plot_data(mileage, price, thetas):
	plt.scatter(mileage, price, color='blue')
	plt.plot(mileage, thetas[0] * price + thetas[1] * mileage, color='red')
	plt.xlabel("Mileage")
	plt.ylabel("Price")
	plt.title("Price vs Mileage")
	plt.show()


def main():
	mileage, price = load_data()
	gradient_descent(mileage, price)
	plot_data(mileage, price. Static.get_thetas())



	


if __name__ == "__main__":
	main()


#https://www.youtube.com/watch?v=rcl_YRyoLIY
#https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/