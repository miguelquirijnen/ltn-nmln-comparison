import os
fileName = input("Please enter the name of the file you'd like to use.")
while not os.path.isfile(fileName):
    fileName = input("Whoops! No such file! Please enter the name of the file you'd like to use.")
