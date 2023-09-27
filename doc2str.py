file = open("a.txt", "r")
stext = file.read()
file.close()

lines = stext.splitlines()
first_three = lines[:5]
output = "\n".join(first_three)

#output = stext.split('\n')[3]

print(output)