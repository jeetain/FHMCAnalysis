import os

if __name__ == '__main__':
	files = os.listdir('./')
	tests = [f for f in files if '.py' in f and 'run_tests.py' != f and 'test.py' != f]
	for test in tests:
		print 14*'*****'+'\nRunning: '+test+'\n'+14*'*****'+'\n'
		os.system('python '+test)
