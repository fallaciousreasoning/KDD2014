"""Just uses the first 300 rows for testing"""

def smallerise(filename, lines_count = 1000):
	lines = []
	with open(filename, 'r', errors='ignore') as f:
		for i in range(lines_count):
			try:
				lines.append(f.readline())
			except:
				print('uh oh, coudln\'t read line',i,'of', filename)

	parts = filename.split('.')
	small_filename = parts[0] + '_small.' + parts[1]

	with open(small_filename, 'w') as f:
		for line in lines:
			f.write(line)

filenames = [
	'donations.csv',
	'essays.csv',
	'projects.csv',
	'resources.csv'
]

for filename in filenames:
	try:
		smallerise(filename)
	except:
		print('couldn\'t read', filename)