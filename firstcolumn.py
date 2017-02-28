import csv
with open('file.txt', 'rb') as myfile:
	spamreader = csv.reader(myfile, delimiter=' ', )
	for row in spamreader:
		print row[0]
