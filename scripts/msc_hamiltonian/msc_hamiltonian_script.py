import os
import time

'''
alpha = 0.0001
#extensions = [500, 1000, 2000, 4000, 8000, 16000]
extension = 500
sizes = [51, 101, 201, 301, 501, 751, 1001, 2001, 3001, 5001, 10001, 20001]

for size in sizes:
    os.system(f"python msc_hamiltonian.py --alpha={alpha} --position_grid_size={size} --position_grid_extension={extension}")

#os.system("python ../../results/data/msc_hamiltonian/take_3/rename_json.py ../../results/data/msc_hamiltonian/take_3/")

#os.system("shutdown")
'''

time.sleep(3600)
os.system("python msc_hamiltonian.py --alpha=0.0001 --position_grid_extension=100 --position_grid_size=20001")