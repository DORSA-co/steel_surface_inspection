import csv

csv_path = 'severstal-steel-defect-detection/train.csv'

def csv_reader( csv_path):
    with open( csv_path, newline='') as csvfile :
            csv_iter = csv.reader( csvfile)
            csv_file = list(csv_iter)
            return csv_file[1:]  



csv_list = csv_reader( csv_path)
m=0