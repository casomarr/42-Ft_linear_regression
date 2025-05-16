import sys
import json #pour récupérer les thetas du json file dans lequel training.py a sauvegardé leurs valeurs
import matplotlib.pyplot as plt #library used for data visualization
import numpy as np #library used for computations

def plot_data(mileage, thetas, predicted_price):
	data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
	dataset_mileage = data[:,0]
	dataset_price = data[:,1]
	plt.scatter(dataset_mileage, dataset_price, color='blue', label='Data Points')
	regression_line = thetas[1] * dataset_mileage + thetas[0]
	plt.plot(dataset_mileage, regression_line, color='red', label='Regression Line')
	plt.scatter(mileage, predicted_price, color='green', label='Predicted Price', marker='x', s=100, linewidths=2)
	plt.xlabel("Mileage")
	plt.ylabel("Price")
	plt.title("Predicted price of a car for a given mileage")
	plt.show()

def load_thetas():
	'''This function loads the thetas from the json file where training.py has saved them.'''
	try:
		with open('thetas.json', 'r') as f:
			thetas = json.load(f)
	except FileNotFoundError:
		print("Thetas file not found. Please run training.py first.")
		sys.exit()
	return thetas

def main():
	'''Predicts the price of a car for a given mileage'''

	try:
		mileage = float(input("Enter a mileage: "))
	except ValueError:
		print("Invalid mileage: should be a numeric value")
	if (mileage < 0):
		print("The mileage has to be positive.")
		sys.exit()
	if (mileage > 1000000):
		print("No car has such a high mileage.")
		sys.exit()

	thetas = load_thetas()
	predicted_price = thetas[1] * mileage + thetas[0]
	print(f"Estimated price : {predicted_price}$")
	plot_data(mileage, thetas, predicted_price)
	
	
if __name__ == "__main__":
	main()
