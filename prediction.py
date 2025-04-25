import sys
from training import Static


def main(): #clean code : no need for a main in python but increases readability
	'''Predicts the price of a car for a given mileage'''

	try:
		mileage = float(input("Enter a mileage: "))
	except ValueError:
		print("Invalid mileage: should be a numeric value")
	if (mileage < 0):
		print("The mileage has to be positive.")
		sys.exit() #equivalent of return 1 from main in C
	if (mileage > 1,000,000)
		print("No car has such a high mileage.")
		sys.exit()

	thetas = Static.get_thetas() #retrieves thetas from training.py

	predicted_price = thetas[1] * mileage + thetas[0]
	print(f"Estimated price : {predicted_price}$")
	
	

if __name__ == "__main__":
	main()
#good practice : ensures that the main() 
#function is only executed when the script is run directly, 
# not when it is imported as a module.