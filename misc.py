import os, re
from prettytable import from_csv

dir = "./docs"


# check if a file exists in directory
def file_exist(dir, filename):
  
  filepath = os.path.join(dir, filename)
  
  if os.path.isfile(filepath):
    return True
  else:
    return False

def ls():
     list_of_files = os.listdir(dir) # get all files in dir
     for file in list_of_files:
          print(file) # print the name of file
          with open(os.path.join(dir, file), "r") as f: # open the file
               content = f.read() # read the content
               char_num = len(content.split()) # count the number of words
               print(char_num) # print the number of words


# find chunks in a specific directory
def find_files(dir, doc, regexp):
  
  matching_chunks = []
  
  for file in os.listdir(dir):
    # Check if the file matches the pattern "filename_regexp.txt" using re.match
    if re.match(f"{doc}_{regexp}\.txt", file):
      matching_chunks.append(file)
  
  if not matching_chunks:
    print("no file found")
  else:
    # Print the matching files
    for file in matching_chunks:
      print(file)

def pt():
     with open("kbd.csv") as fp:
          mytable = from_csv(fp)
     print(mytable)
     return mytable

def pt2file():
     mytable = pt()
     # Get the table output as a string
     table_txt = mytable.get_csv_string()
     # Open a file in the current directory in write mode
     with open("mytable.txt", "w") as fp:
          # Write the table string to the file
          fp.write(table_txt)


def dash2hash():
     print("d2hash")

     # Define the directory path
     dir_path = "./chunks"

     # Loop through the files in the directory
     for file in os.listdir(dir_path):
          # Find the last index of "_"
          last_underscore = file.rfind("_")
          
          # If there is an underscore, replace it with "#"
          if last_underscore != -1:
               new_file = file[:last_underscore] + "#" + file[last_underscore + 1:]
               # Rename the file
               os.rename(os.path.join(dir_path, file), os.path.join(dir_path, new_file))
               # Print the old and new file names
               print(f"Renamed {file} to {new_file}")

print("hello world!")

# list_of_files = os.listdir(dir)
# print(list_of_files)

#find_files("chunks", "AEO-FAQ", "\d+")

# bool = file_exist("docs", "AEO-FAQ")
# print(bool)